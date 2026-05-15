"""
Baseline: KVC-Top
The top KVC-onGoing entry re-trained on our split.
Implemented as KeystrokeModelBCE (transformer + per-token sigmoid head),
applied window-by-window with standard BCE — no enrollment bank.

The window label is 1 if any event in the window is post-onset (same as StreamDataset).

Usage:
    cd <project_root>
    python -m experiments.AaltoDB.baselines.kvc_top.train [epochs]
"""

import json
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from data.AaltoDB.prepare import load as load_aalto                              # noqa: E402
from experiments.config import parse_runtime_config                       # noqa: E402
from data.AaltoDB.stats import (                                        # noqa: E402
    aalto_feature_ranges, aalto_vocab_size,
)
from experiments.baselines.kvc_top.model import KeystrokeModel, KeystrokeModelBCE    # noqa: E402
from data.AaltoDB.stage_2 import StreamDataset, load_streams  # noqa: E402
from utils.config import seed_training, seed_worker                              # noqa: E402
from utils.wandb import setup_lightning_wandb                                    # noqa: E402
from utils.logger import get_logger # noqa: E402

LOGGER = get_logger(__name__)

_BEST_MODELS_DIR = Path(__file__).resolve().parent / "best_models"
_CKPT_WARNING    = r"Checkpoint directory .* exists and is not empty\."

SEQ_LEN    = 50
BATCH_SIZE = 2048
LR         = 1e-3
EMBED_DIM  = 4     # match existing KeystrokeModel default


# ── lightning module ──────────────────────────────────────────────────────────

class KVCTopModule(pl.LightningModule):
    """
    KeystrokeModelBCE outputs (B, T) per-token probabilities.
    We aggregate to window-level by taking the mean across T tokens,
    then apply BCE against the window label.
    """

    def __init__(self, model: KeystrokeModelBCE, lr: float):
        super().__init__()
        self.model = model
        self.lr    = lr
        _compiled  = torch.compile(self.model, dynamic=True)
        object.__setattr__(self, "_fwd", _compiled)

    def _step(self, batch):
        windows, labels = batch          # (B, L, F), (B,)
        token_probs = self._fwd(windows) # (B, L)
        p_window    = token_probs.mean(dim=1)  # (B,)
        return F.binary_cross_entropy(p_window, labels)

    def training_step(self, batch, _):
        loss = self._step(batch)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._step(batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                 weight_decay=1e-2)


# ── entry point ───────────────────────────────────────────────────────────────

def train(epochs: int = 30):
    import warnings
    warnings.filterwarnings("ignore", message=_CKPT_WARNING, category=UserWarning)
    torch.set_float32_matmul_precision("high")

    project_cfg = json.loads((_PROJECT_ROOT / "config.json").read_text())
    config = parse_runtime_config(project_cfg)
    seed_training(config.seed)

    train_data, val_data, _ = load_aalto()
    vocab_size = aalto_vocab_size(train_data)
    ranges     = aalto_feature_ranges(train_data)

    streams  = load_streams()
    train_ds = StreamDataset(streams["train"], seq_len=SEQ_LEN)
    val_ds   = StreamDataset(streams["val"],   seq_len=SEQ_LEN)
    LOGGER.info(f"  train windows={len(train_ds):,}  val windows={len(val_ds):,}")

    def _loader(ds, shuffle, seed_offset):
        g = torch.Generator()
        g.manual_seed(config.seed + seed_offset)
        return DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=shuffle,
            num_workers=config.num_workers, pin_memory=True,
            persistent_workers=config.num_workers > 0,
            worker_init_fn=seed_worker, generator=g,
            **({"prefetch_factor": config.prefetch_factor} if config.num_workers > 0 else {}),
        )

    backbone = KeystrokeModel(SEQ_LEN, vocab_size, EMBED_DIM, ranges)
    model    = KeystrokeModelBCE(backbone)
    module   = KVCTopModule(model, lr=LR)

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
        log_every_n_steps=max(1, min(config.log_every_n_steps, len(_loader(train_ds, True, 0)))),
        enable_model_summary=False, num_sanity_val_steps=0, precision="32-true",
    )
    trainer.fit(
        module,
        train_dataloaders=_loader(train_ds, shuffle=True,  seed_offset=0),
        val_dataloaders  =_loader(val_ds,   shuffle=False, seed_offset=1),
    )
    LOGGER.info(f"\nBest checkpoint: {ckpt.best_model_path}")


if __name__ == "__main__":
    train(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 30)
