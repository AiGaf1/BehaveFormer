"""
Stage 2: train the bank-aware decision module D_phi.

The encoder f_theta is loaded from the best stage-1 checkpoint and frozen.
D_phi is trained with a time-aware BCE loss that penalises delayed detections
exponentially after the attack onset t*.

Data flow (per batch):
  windows (B, L, F)  ──[frozen encoder]──> z_t (B, D)
  user_idx (B,)      ──[bank lookup]──────> bank (B, K, D)
  z_t + bank         ──[D_phi]───────────> p_t (B,)   in [0,1]
  p_t + labels + t_star + w_start  ──> L_TA loss

Banks are pre-computed once before training: for each user in train+val we
run enrollment windows through the frozen encoder and sample K embeddings.

Validation metric: AUSC (area under Usability–PTCR curve) on a subset of val
streams.  Early stopping monitors val_ausc (higher = better).

Usage:
    cd <project_root>
    python -m experiments.stage2_detector.train [epochs] [stage1_ckpt]
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from data.AaltoDB.prepare import load as load_aalto  # noqa: E402
from data.AaltoDB.stage_2 import StreamDataset, load_streams, stream_collate  # noqa: E402
from data.AaltoDB.stats import aalto_feature_ranges, aalto_vocab_size  # noqa: E402
from evaluation.metrics import Metric  # noqa: E402
from evaluation.stage2 import build_banks, score_stream  # noqa: E402
from experiments.config import parse_runtime_config  # noqa: E402
from experiments.loaders import make_dataloader  # noqa: E402
from experiments.stage1_encoder.model import Encoder  # noqa: E402
from experiments.stage1_encoder.train import (
    _BEST_MODELS_DIR as _STAGE1_BEST_MODELS_DIR,
)
from experiments.stage1_encoder.train import (  # noqa: E402
    DROPOUT,
    HEADS,
    KEY_EMB,
    LFF_FEATURES,
    NUM_LAYERS,
    SEQ_LEN,
)
from experiments.stage2_detector.loss import time_aware_bce  # noqa: E402
from experiments.stage2_detector.model import BankAwareDetector  # noqa: E402
from utils.config import seed_training  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.optimizers import MultiOptimizer, build_muon_hybrid  # noqa: E402
from utils.wandb import setup_lightning_wandb  # noqa: E402

LOGGER = get_logger(__name__)
_BEST_MODELS_DIR = Path(__file__).resolve().parent / "best_models"
_CKPT_WARNING    = r"Checkpoint directory .* exists and is not empty\."
_WORKERS_WARNING = r"The .* does not have many workers"
_EMPTY_WARNING   = r"Total length of `DataLoader` across ranks is zero"
_EVAL_WARNING    = r"Found \d+ module\(s\) in eval mode at the start of training"

# ── hyper-parameters ──────────────────────────────────────────────────────────
BANK_K           = 5      # enrollment embeddings per user in the bank
TOP_K_STATS      = 3      # top-k for ψ_t similarity stats
HIDDEN_DIM       = 128    # MLP hidden size
BATCH_SIZE       = 4096
LR               = 1e-3
VAL_STREAMS      = 2000   # max val streams for AUSC (None = all)


def _subsample_val_streams(streams: list, max_streams: int, seed: int = 0) -> list:
    """Subsample to keep validation tractable, balanced across attack/same_user."""
    rng = np.random.default_rng(seed)
    half = max_streams // 2
    attacks   = [s for s in streams if s["stream_type"] == "attack"]
    same_user = [s for s in streams if s["stream_type"] == "same_user"]
    if len(attacks) > half:
        attacks = [attacks[i] for i in rng.choice(len(attacks), half, replace=False)]
    if len(same_user) > half:
        same_user = [same_user[i] for i in rng.choice(len(same_user), half, replace=False)]
    return attacks + same_user


# ── Lightning module ──────────────────────────────────────────────────────────

class Stage2Module(pl.LightningModule):
    _fwd: Callable

    def __init__(
        self,
        encoder:   Encoder,
        detector:  BankAwareDetector,
        train_banks: dict[int, torch.Tensor],
        val_banks:   dict[int, torch.Tensor],
        val_streams: list,
        lr: float,
        config,
        loss_fn=None,
    ):
        super().__init__()
        self.encoder  = encoder
        self.detector = detector
        self.val_streams = val_streams
        self.lr      = lr
        self.config  = config
        self.loss_fn = loss_fn or time_aware_bce

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Pre-stack banks into dense (max_uid+1, K, D) tensors on device for fast
        # batched indexing — avoids per-sample CPU→GPU transfers in training_step.
        # `present` masks uids that have a bank (so we can assert on lookup
        # instead of silently scoring against zeros).
        self.train_banks_tensor, self._train_present = self._densify(train_banks)
        self.val_banks_tensor,   self._val_present   = self._densify(val_banks)
        # Keep dict only for stream-level eval (score_stream).
        self._val_banks = val_banks

        object.__setattr__(self, "_fwd", torch.compile(self.detector, dynamic=True))

    @staticmethod
    def _densify(banks: dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        max_uid = max(banks.keys())
        K, D = next(iter(banks.values())).shape
        out = torch.zeros(max_uid + 1, K, D)
        present = torch.zeros(max_uid + 1, dtype=torch.bool)
        for uid, b in banks.items():
            out[uid] = b
            present[uid] = True
        return out, present

    def on_fit_start(self):
        self.train_banks_tensor = self.train_banks_tensor.to(self.device)
        self.val_banks_tensor   = self.val_banks_tensor.to(self.device)
        self._train_present     = self._train_present.to(self.device)
        self._val_present       = self._val_present.to(self.device)
        # Encoder is frozen — keep it in eval() mode (no dropout / BN updates)
        # across all train/val steps. Lightning's .train()/.eval() propagation
        # would otherwise flip it back to train mode in training_step.
        self.encoder.eval()

    def on_train_epoch_start(self):
        self.encoder.eval()

    def training_step(self, batch, _):
        windows, labels, user_idx, t_star, w_start = batch
        uid = user_idx.to(self.device)
        assert bool(self._train_present[uid].all()), "stream references a user without a bank"
        bank = self.train_banks_tensor[uid]
        with torch.no_grad():
            z_t = self.encoder(windows)
        p_t  = self._fwd(z_t, bank)
        loss = self.loss_fn(p_t, labels, t_star.to(self.device), w_start.to(self.device))
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.encoder.eval()

    def validation_step(self, batch, _):
        windows, labels, user_idx, t_star, w_start = batch
        uid = user_idx.to(self.device)
        assert bool(self._val_present[uid].all()), "stream references a user without a bank"
        bank = self.val_banks_tensor[uid]
        z_t  = self.encoder(windows)
        p_t  = self._fwd(z_t, bank)
        loss = self.loss_fn(p_t, labels, t_star.to(self.device), w_start.to(self.device))
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

    def on_validation_epoch_end(self):
        score_fn = lambda s: score_stream(s, self.encoder, self.detector,  # noqa: E731
                                          self._val_banks, SEQ_LEN, self.device)
        subset = _subsample_val_streams(self.val_streams, VAL_STREAMS)
        m = Metric.evaluate_streams(score_fn, subset, single_streams=[])
        self.log("val_ausc",  m["ausc"],      on_epoch=True, prog_bar=True)
        self.log("val_ptcr",  m["ptcr"],      on_epoch=True, prog_bar=True)
        self.log("val_edd",   m["edd"],       on_epoch=True, prog_bar=True)
        self.log("val_usab",  m["usability"], on_epoch=True, prog_bar=True)
        self.log("val_tauop", m["tau_op"],    on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.detector.parameters(), lr=self.lr,
                weight_decay=self.config.weight_decay,
            )
        return MultiOptimizer(build_muon_hybrid(self.detector, self.lr, self.config))


# ── entry point ───────────────────────────────────────────────────────────────

def train(epochs: int = 30, stage1_ckpt: Path | None = None,
          loss_fn=None, best_models_dir: Path | None = None):
    import warnings
    warnings.filterwarnings("ignore", message=_CKPT_WARNING, category=UserWarning)
    warnings.filterwarnings("ignore", message=_WORKERS_WARNING)
    warnings.filterwarnings("ignore", message=_EMPTY_WARNING)
    warnings.filterwarnings("ignore", message=_EVAL_WARNING)
    torch.set_float32_matmul_precision("high")
    best_models_dir = best_models_dir or _BEST_MODELS_DIR

    project_cfg = json.loads((_PROJECT_ROOT / "config.json").read_text())
    config = parse_runtime_config(project_cfg)
    seed_training(config.seed)

    # ── load encoder ─────────────────────────────────────────────────────────
    if stage1_ckpt is None:
        ckpts = sorted(_STAGE1_BEST_MODELS_DIR.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No stage-1 checkpoints in {_STAGE1_BEST_MODELS_DIR}")
        stage1_ckpt = ckpts[-1]
    LOGGER.info(f"Stage-1 checkpoint: {stage1_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Loading Aalto features …")
    train_data, val_data, _ = load_aalto()
    vocab_size = aalto_vocab_size(train_data)
    ranges     = aalto_feature_ranges(train_data)

    encoder = Encoder(
        seq_len=SEQ_LEN, vocab_size=vocab_size, key_emb=KEY_EMB,
        feature_ranges=ranges, lff_features=LFF_FEATURES,
        num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT,
    ).to(device)
    s1     = torch.load(stage1_ckpt, map_location=device)
    state  = {k.removeprefix("encoder."): v for k, v in s1["state_dict"].items() if k.startswith("encoder.")}
    encoder.load_state_dict(state)
    encoder.eval()

    # ── pre-build banks ───────────────────────────────────────────────────────
    LOGGER.info("Building train banks …")
    train_banks = build_banks(encoder, train_data, SEQ_LEN, device, k=BANK_K)
    LOGGER.info("Building val banks …")
    val_banks   = build_banks(encoder, val_data,   SEQ_LEN, device, k=BANK_K)

    # ── datasets ──────────────────────────────────────────────────────────────
    LOGGER.info("Loading streams …")
    streams     = load_streams()
    # Skip streams whose user has no enrollment bank (short enrollment sessions).
    train_streams = [s for s in streams["train"] if s["user_idx"] in train_banks]
    val_streams   = [s for s in streams["val"]   if s["user_idx"] in val_banks]
    LOGGER.info(f"train streams: {len(train_streams):,} / {len(streams['train']):,}  "
                f"val streams: {len(val_streams):,} / {len(streams['val']):,}")
    train_ds = StreamDataset(train_streams, seq_len=SEQ_LEN, include_meta=True)
    val_ds   = StreamDataset(val_streams,   seq_len=SEQ_LEN, include_meta=True)

    train_loader = make_dataloader(train_ds, BATCH_SIZE, shuffle=True,
                                   config=config, seed_offset=0,
                                   collate_fn=stream_collate)
    val_loader   = make_dataloader(val_ds, BATCH_SIZE, shuffle=False,
                                   config=config, seed_offset=1,
                                   collate_fn=stream_collate)

    detector = BankAwareDetector(
        embed_dim=encoder.out_dim, top_k=TOP_K_STATS,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    )
    module = Stage2Module(
        encoder, detector,
        train_banks, val_banks, val_streams,
        lr=LR, config=config, loss_fn=loss_fn,
    )

    best_models_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=str(best_models_dir),
        filename="epoch_{epoch}_ausc_{val_ausc:.4f}",
        monitor="val_ausc",
        mode="max",
        save_top_k=1,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )
    callbacks: list[pl.Callback] = [ckpt]
    if project_cfg["runtime"].get("model_summary", False):
        callbacks.append(ModelSummary(max_depth=project_cfg["runtime"].get("model_summary_depth", 3)))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        callbacks=callbacks,
        logger=setup_lightning_wandb(_PROJECT_ROOT, module, project_cfg["runtime"]) or False,
        log_every_n_steps=max(1, min(config.log_every_n_steps, len(train_loader))),
        enable_model_summary=False,
        num_sanity_val_steps=0,
        precision="32-true",
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    LOGGER.info(f"Best checkpoint: {ckpt.best_model_path}")


if __name__ == "__main__":
    ckpt = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    train(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 30, stage1_ckpt=ckpt)
