"""
Baseline: Boundary-Only
Trained exclusively on within-window timing-discontinuity features —
no identity information, only signals that a session boundary might exist.

Features per window (3 scalars):
  ft_spike    – max absolute flight-time value in the window
                (session concatenation produces an anomalous FT at the join)
  ht_var_diff – |var(HT, first half) - var(HT, second half)|
                (typing rhythm variance shifts at a boundary)
  max_gap     – max flight-time value in the window
                (long inter-event gap indicates a pause/switch)

A 2-layer MLP is trained with BCE on StreamDataset windows (no encoder, no bank).
Under our shortcut-free protocol this should achieve AUSC ≈ 0.5.

Usage:
    cd <project_root>
    python -m experiments.AaltoDB.baselines.boundary_only.train [epochs]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import parse_runtime_config                   # noqa: E402
from data.AaltoDB.stage_2 import load_streams        # noqa: E402
from experiments.loaders import make_dataloader                  # noqa: E402
from utils.config import seed_training                                        # noqa: E402
from utils.wandb import setup_lightning_wandb                                # noqa: E402
from utils.logger import get_logger # noqa: E402

LOGGER = get_logger(__name__)

_BEST_MODELS_DIR = Path(__file__).resolve().parent / "best_models"
_CKPT_WARNING    = r"Checkpoint directory .* exists and is not empty\."

SEQ_LEN    = 50
BATCH_SIZE = 4096
LR         = 1e-3
N_FEATURES = 3


# ── feature extraction ────────────────────────────────────────────────────────

def boundary_features(window: np.ndarray) -> np.ndarray:
    """
    window: (seq_len, 3)  [hold_time, flight_time, key_id]
    Returns (3,) float32: [ft_spike, ht_var_diff, max_gap]
    """
    ht = window[:, 0].astype(np.float32)
    ft = window[:, 1].astype(np.float32)
    half = len(ht) // 2
    ht_var_diff = abs(float(ht[:half].var()) - float(ht[half:].var()))
    ft_spike    = float(np.abs(ft).max())
    max_gap     = float(ft.max())
    return np.array([ft_spike, ht_var_diff, max_gap], dtype=np.float32)


class BoundaryWindowDataset(Dataset):
    """Flat (features, label) dataset over all windows in the given streams."""

    def __init__(self, streams: list, seq_len: int):
        self._items: list[tuple[np.ndarray, float]] = []
        for stream in streams:
            events = stream["events"]
            labels = stream["labels"]
            n = len(events)
            if n < seq_len:
                continue
            for i in range(n - seq_len + 1):
                window = events[i: i + seq_len]
                label  = float(labels[i: i + seq_len].any())
                self._items.append((boundary_features(window), label))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        feat, label = self._items[idx]
        return torch.from_numpy(feat), torch.tensor(label)


# ── model ─────────────────────────────────────────────────────────────────────

class BoundaryMLP(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x).squeeze(-1))


# ── lightning module ──────────────────────────────────────────────────────────

class BoundaryModule(pl.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.model = BoundaryMLP()
        self.lr = lr

    def _step(self, batch):
        x, y = batch
        return F.binary_cross_entropy(self.model(x), y)

    def training_step(self, batch, _):
        loss = self._step(batch)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._step(batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


# ── entry point ───────────────────────────────────────────────────────────────

def train(epochs: int = 30):
    import warnings
    warnings.filterwarnings("ignore", message=_CKPT_WARNING, category=UserWarning)
    torch.set_float32_matmul_precision("high")

    project_cfg = json.loads((_PROJECT_ROOT / "config.json").read_text())
    config = parse_runtime_config(project_cfg)
    seed_training(config.seed)

    streams  = load_streams()
    train_ds = BoundaryWindowDataset(streams["train"], SEQ_LEN)
    val_ds   = BoundaryWindowDataset(streams["val"],   SEQ_LEN)
    LOGGER.info(f"  train windows={len(train_ds):,}  val windows={len(val_ds):,}")

    train_loader = make_dataloader(train_ds, BATCH_SIZE, shuffle=True,  config=config, seed_offset=0)
    val_loader   = make_dataloader(val_ds,   BATCH_SIZE, shuffle=False, config=config, seed_offset=1)

    module = BoundaryModule(lr=LR)
    _BEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = ModelCheckpoint(
        dirpath=str(_BEST_MODELS_DIR),
        filename="epoch_{epoch}_val_{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=1,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )
    callbacks: list[pl.Callback] = [ckpt]
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=epochs,
        callbacks=callbacks,
        logger=setup_lightning_wandb(_PROJECT_ROOT, module, project_cfg["runtime"]) or False,
        log_every_n_steps=max(1, min(config.log_every_n_steps, len(train_loader))),
        enable_model_summary=False, num_sanity_val_steps=0, precision="32-true",
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    LOGGER.info(f"\nBest checkpoint: {ckpt.best_model_path}")


if __name__ == "__main__":
    train(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 30)
