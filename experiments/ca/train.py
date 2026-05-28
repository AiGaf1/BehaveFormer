"""
End-to-end single-model continuous authentication — NO Stage 1.

Borrows Personal VAD's *conditioning* (inject the enrolled user via a K-vector bank)
but replaces its per-frame decision with a causal GRU that *accumulates* evidence over
the stream (`ProfileConditionedAccumulator`). Encoder + detector are trained jointly
from scratch under time-aware BCE plus a soft monotonicity penalty, on full streams.

Enrollment banks are stored as RAW windows; at each TRAINING STEP the current batch's
users' banks are re-encoded fresh with the live encoder (detached), so the conditioning
profile never goes stale within an epoch. Validation banks are encoded once per val epoch
(the encoder is frozen there). Cost per step: the batch's stream windows (with grad) plus
its users' K bank windows (no grad, one encode per unique user).

Usage:
    cd <project_root>
    python -m experiments.ca.train [epochs] [seq_len]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, RandomSampler  # noqa: E402

from data.AaltoDB.prepare import load as load_aalto  # noqa: E402
from data.AaltoDB.stats import aalto_feature_ranges, aalto_vocab_size  # noqa: E402
from data.AaltoDB.streams import (  # noqa: E402
    StreamLevelDataset,
    build_streams_for_split,
    stream_session_collate,
)
from evaluation.ca_banks import build_raw_banks  # noqa: E402
from evaluation.metrics import Metric  # noqa: E402
from experiments.ca.model import ProfileConditionedAccumulator  # noqa: E402
from experiments.config import parse_runtime_config  # noqa: E402
from experiments.verification.model import Encoder  # noqa: E402
from utils.config import seed_training  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.optimizers import MultiOptimizer, split_optimizer_params  # noqa: E402
from utils.wandb import setup_lightning_wandb  # noqa: E402

LOGGER = get_logger(__name__)
_BEST_MODELS_DIR = Path(__file__).resolve().parent / "best_models"
_CKPT_WARNING    = r"Checkpoint directory .* exists and is not empty\."
_WORKERS_WARNING = r"The .* does not have many workers"
_EMPTY_WARNING   = r"Total length of `DataLoader` across ranks is zero"
_EVAL_WARNING    = r"Found \d+ module\(s\) in eval mode at the start of training"

# ── encoder architecture (trained from scratch; mean-pool + L2-norm) ─────────
KEY_EMB        = 4
LFF_FEATURES   = 16
NUM_LAYERS     = 2
HEADS          = 2
DROPOUT        = 0.1

# ── detector + training hyper-parameters ─────────────────────────────────────
USE_FILM          = True      # detector conditioning: True=FiLM, False=concat. Must
                              # match at train + inference (encoder ckpt depends on it).
BANK_K            = 5         # enrollment windows per user (distributed across sessions)
TAU               = 0.07      # cosine geometry (z & bank are L2-normalised)
GRAD_CLIP         = 1.0       # gradient norm clipping
BATCH_STREAMS     = 32        # streams per batch (forward encodes all their windows)
STREAMS_PER_EPOCH = 10_000    # cap train epoch length; sampled with replacement
VAL_STREAMS_LOSS  = 2_000     # cap val-loss epoch length
LR                = 1e-3
VAL_STREAMS       = 1_000     # streams scored by the full per-stream sigmoid pass


def _compute_pos_weight(streams: list, seq_len: int) -> float:
    """Per-window pos/neg ratio over a stream list (cumsum trick).
    A window starting at i is positive iff any event in [i, i+seq_len-1] is post-attack.
    """
    n_pos = n_neg = 0
    for s in streams:
        labels = np.asarray(s["labels"])
        if len(labels) < seq_len:
            continue
        cs = np.concatenate([[0], np.cumsum(labels)])
        window_sums = cs[seq_len:] - cs[:-seq_len]
        n_pos += int((window_sums > 0).sum())
        n_neg += int((window_sums == 0).sum())
    return n_neg / max(n_pos, 1)


def _subsample_val_streams(streams: list, max_streams: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    half = max_streams // 2
    attacks   = [s for s in streams if s["stream_type"] == "attack"]
    same_user = [s for s in streams if s["stream_type"] == "same_user"]
    if len(attacks) > half:
        attacks = [attacks[i] for i in rng.choice(len(attacks), half, replace=False)]
    if len(same_user) > half:
        same_user = [same_user[i] for i in rng.choice(len(same_user), half, replace=False)]
    return attacks + same_user


def _chunked_encode(encoder, windows: torch.Tensor, device, chunk: int = 8192) -> torch.Tensor:
    """Encode (N, L, F) raw windows → (N, D) embeddings, moving chunks to `device`.

    Grad flows when called outside no_grad (training); wrap in no_grad for banks.
    """
    if len(windows) == 0:
        return torch.zeros(0, encoder.out_dim, device=device)
    out = [encoder(windows[i: i + chunk].to(device)) for i in range(0, len(windows), chunk)]
    return torch.cat(out, dim=0)


# ── Lightning module ──────────────────────────────────────────────────────────

class Stage2E2EModule(pl.LightningModule):
    def __init__(
        self,
        encoder:  Encoder,
        detector: ProfileConditionedAccumulator,
        train_data: list,
        train_banks_raw: dict[int, torch.Tensor],
        val_banks_raw:   dict[int, torch.Tensor],
        val_streams: list,
        lr: float,
        config,
        seq_len: int = 50,
        pos_weight: float = 1.0,
        total_steps: int = 0,
        augment: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "detector", "train_data", "train_banks_raw", "val_banks_raw", "val_streams", "config"])
        self.encoder  = encoder
        self.detector = detector
        self.train_data  = train_data
        self.val_streams = val_streams
        self.seq_len = seq_len
        self.lr = lr
        self.config = config
        self.total_steps = total_steps
        self.augment = augment
        self._aug_epoch: int | None = None  # set by train_dataloader each epoch
        self.pos_weight: torch.Tensor
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

        self._train_banks_raw = train_banks_raw   # kept for bank filtering
        self._train_uids, self._train_raw, self._train_row = self._stack_banks(train_banks_raw)
        self._val_uids,   self._val_raw,   self._val_row   = self._stack_banks(val_banks_raw)
        self.val_bank_emb:   torch.Tensor | None = None  # (n, D) profile per user

    def train_dataloader(self):
        import random
        # Regenerate streams each epoch with a fresh seed → new t_star + impostor pairings.
        epoch = self.current_epoch if self.trainer is not None else 0
        self._aug_epoch = epoch if self.augment else None  # consumed by _fresh_train_banks
        streams, _ = build_streams_for_split(self.train_data, random.Random(epoch))
        streams = [s for s in streams
                   if s["user_idx"] in self._train_banks_raw and len(s["events"]) >= self.seq_len]
        attacks = [s for s in streams if s["stream_type"] == "attack"]
        mean_tstar = float(np.mean([s["t_star"] for s in attacks])) if attacks else 0.0
        LOGGER.info(f"[epoch {epoch}] regenerated {len(streams):,} train streams  "
                    f"mean t*={mean_tstar:.1f}  augment={self.augment}")
        ds = StreamLevelDataset(streams, seq_len=self.seq_len, augment_epoch=self._aug_epoch)
        sampler = RandomSampler(ds, replacement=True, num_samples=STREAMS_PER_EPOCH)
        return DataLoader(ds, batch_size=BATCH_STREAMS, sampler=sampler,
                          num_workers=0, drop_last=True, collate_fn=stream_session_collate)

    @staticmethod
    def _stack_banks(banks: dict[int, torch.Tensor]):
        uids = sorted(banks.keys())
        raw  = torch.stack([banks[u] for u in uids])             # (n, K, L, F) cpu
        row  = torch.full((uids[-1] + 1,), -1, dtype=torch.long)
        row[torch.tensor(uids)] = torch.arange(len(uids))
        return torch.tensor(uids), raw, row

    def on_fit_start(self):
        self._train_row = self._train_row.to(self.device)
        self._val_row   = self._val_row.to(self.device)

    @torch.no_grad()
    def _encode_banks(self, raw: torch.Tensor) -> torch.Tensor:
        """raw (n, K, L, F) cpu → (n, 2D) on device, detached profile vectors.

        Encodes each of the n·K bank windows, then summarises over K as
        [L2-normed mean, per-dim std]. The std half captures the within-bank
        distribution width (the signal that separates loose live signatures
        from tight Aalto ones).
        """
        was_training = self.encoder.training
        self.encoder.eval()
        n, K, L, Fdim = raw.shape
        z = _chunked_encode(self.encoder, raw.reshape(n * K, L, Fdim), self.device)
        if was_training:
            self.encoder.train()
        z = z.reshape(n, K, -1).detach()                          # (n, K, D)
        prof_mean = F.normalize(z.mean(dim=1), dim=-1)            # (n, D)
        prof_std  = z.std(dim=1)                                   # (n, D)
        return torch.cat([prof_mean, prof_std], dim=-1)          # (n, 2D)

    def _fresh_train_banks(self, uid_batch: torch.Tensor):
        """Re-encode THIS batch's users' banks → (U, D) profile vectors with the
        live encoder (detached). Recomputed each step so the conditioning profile
        tracks the current encoder state.

        When augmentation is on, applies the SAME per-(epoch, user) augmentation
        to each user's bank windows as was applied to their stream — so the
        bank↔stream relationship is preserved.
        """
        uids = torch.unique(uid_batch)                        # (U,) on device
        raw  = self._train_raw[self._train_row[uids].cpu()]   # (U, K, L, F) cpu
        if self._aug_epoch is not None:
            from data.augment.timing import _params_for, apply_np
            raw_np = raw.numpy()
            for i, uid in enumerate(uids.tolist()):
                p = _params_for(self._aug_epoch, uid)
                # Augment all K windows for this user with the same params.
                for k in range(raw_np.shape[1]):
                    raw_np[i, k] = apply_np(raw_np[i, k], p)
            raw = torch.from_numpy(raw_np)
        bank_emb = self._encode_banks(raw)                    # (U, D)
        row = torch.full((int(uids.max()) + 1,), -1, dtype=torch.long, device=self.device)
        row[uids] = torch.arange(len(uids), device=self.device)
        return bank_emb, row

    def on_validation_epoch_start(self):
        # Encoder is frozen during validation → encode val banks once per val epoch.
        self.val_bank_emb = self._encode_banks(self._val_raw)  # (n_val, D)

    def _stream_logits(self, z_all, uid, sid, bank_emb, row):
        """Per-stream accumulator pass. Returns logits (N,)."""
        n_streams = int(sid.max().item()) + 1
        logits = torch.empty(z_all.shape[0], device=self.device, dtype=z_all.dtype)
        for s in range(n_streams):
            mask    = (sid == s)
            z_s     = z_all[mask]                              # (T_s, D) time-ordered
            profile = bank_emb[row[uid[mask][0]]]              # (D,)
            logits[mask] = self.detector(z_s, profile)         # (T_s,)
        return logits

    def _shared_step(self, batch, training: bool):
        windows, labels, user_idx, *_, stream_id = batch
        uid = user_idx.to(self.device)
        sid = stream_id.to(self.device)
        z_all = _chunked_encode(self.encoder, windows, self.device)
        if training:
            bank_emb, row = self._fresh_train_banks(uid)
        else:
            bank_emb, row = self.val_bank_emb, self._val_row
        logits = self._stream_logits(z_all, uid, sid, bank_emb, row)
        return F.binary_cross_entropy_with_logits(
            logits, labels.to(self.device), pos_weight=self.pos_weight)

    def training_step(self, batch, _):
        loss = self._shared_step(batch, training=True)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._shared_step(batch, training=False)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

    @torch.no_grad()
    def _score_stream(self, stream, chunk: int = 4096) -> torch.Tensor:
        events = stream["events"].astype(np.float32)
        n = len(events)
        L = self.seq_len
        if n < L:
            return torch.zeros(0)
        windows = torch.from_numpy(
            np.stack([events[i: i + L] for i in range(n - L + 1)]))
        z_seq = _chunked_encode(self.encoder, windows, self.device, chunk=chunk)
        assert self.val_bank_emb is not None
        uid_row = int(self._val_row[stream["user_idx"]].item())
        profile = self.val_bank_emb[uid_row]                         # (D,)
        return torch.sigmoid(self.detector(z_seq, profile)).cpu()

    def on_validation_epoch_end(self):
        score_fn = lambda s: self._score_stream(s)  # noqa: E731
        subset = _subsample_val_streams(self.val_streams, VAL_STREAMS)
        m = Metric.evaluate_streams(score_fn, subset, single_streams=[])
        self.log("val_ausc",  m["ausc"],      on_epoch=True, prog_bar=True)
        self.log("val_ptcr",  m["ptcr"],      on_epoch=True, prog_bar=True)
        self.log("val_edd",   m["edd"],       on_epoch=True, prog_bar=True)
        self.log("val_wdd",   m["wdd"],       on_epoch=True, prog_bar=True)
        self.log("val_usab",  m["usability"], on_epoch=True, prog_bar=True)
        self.log("val_tauop", m["tau_op"],    on_epoch=True, prog_bar=True)

    def configure_optimizers(self):  # type: ignore[override]
        muon_params, adamw_params = [], []
        for model in (self.encoder, self.detector):
            m, a = split_optimizer_params(model)
            muon_params.extend(m)
            adamw_params.extend(a)
        opts = []
        if muon_params:
            opts.append(torch.optim.Muon(
                muon_params, lr=self.lr,
                weight_decay=self.config.weight_decay,
                momentum=self.config.muon_momentum,
                nesterov=self.config.muon_nesterov,
                ns_steps=self.config.muon_ns_steps,
                adjust_lr_fn=self.config.muon_adjust_lr_fn,
            ))
        if adamw_params:
            opts.append(torch.optim.AdamW(
                adamw_params, lr=self.lr, weight_decay=self.config.weight_decay))
        return MultiOptimizer(opts)


# ── entry point ───────────────────────────────────────────────────────────────

def train(epochs: int = 30, seq_len: int = 50, augment: bool = False):
    import warnings
    warnings.filterwarnings("ignore", message=_CKPT_WARNING, category=UserWarning)
    warnings.filterwarnings("ignore", message=_WORKERS_WARNING)
    warnings.filterwarnings("ignore", message=_EMPTY_WARNING)
    warnings.filterwarnings("ignore", message=_EVAL_WARNING)
    torch.set_float32_matmul_precision("high")

    project_cfg = json.loads((_PROJECT_ROOT / "config.json").read_text())
    config = parse_runtime_config(project_cfg)
    seed_training(config.seed)
    LOGGER.info(f"seq_len={seq_len}  epochs={epochs}  ckpt_dir={_BEST_MODELS_DIR.name}")

    LOGGER.info("Loading Aalto features …")
    train_data, val_data, _ = load_aalto()
    vocab_size = aalto_vocab_size(train_data)
    ranges     = aalto_feature_ranges(train_data)

    encoder = Encoder(
        seq_len=seq_len, vocab_size=vocab_size, key_emb=KEY_EMB,
        feature_ranges=ranges, lff_features=LFF_FEATURES,
        num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT,
    )
    detector = ProfileConditionedAccumulator(
        embed_dim=encoder.out_dim, dropout=DROPOUT, use_film=USE_FILM,
    )

    LOGGER.info("Building raw enrollment banks …")
    train_banks = build_raw_banks(train_data, seq_len, k=BANK_K, multi_session=True)
    val_banks   = build_raw_banks(val_data,   seq_len, k=BANK_K, multi_session=True)

    LOGGER.info("Loading val streams …")
    import random as _random
    val_streams_all, _ = build_streams_for_split(val_data, _random.Random(0))
    val_streams = [s for s in val_streams_all
                   if s["user_idx"] in val_banks and len(s["events"]) >= seq_len]
    LOGGER.info(f"val streams: {len(val_streams):,}")

    # Fixed val DataLoader — deterministic, no shuffling.
    val_loss_streams = _subsample_val_streams(val_streams, VAL_STREAMS_LOSS, seed=0)
    val_ds     = StreamLevelDataset(val_loss_streams, seq_len=seq_len)
    val_loader = DataLoader(val_ds, batch_size=BATCH_STREAMS, shuffle=False,
                            num_workers=0, collate_fn=stream_session_collate)

    # pos_weight estimated from epoch-0 streams (representative; doesn't change much epoch-to-epoch)
    epoch0_streams, _ = build_streams_for_split(train_data, _random.Random(0))
    epoch0_train = [s for s in epoch0_streams
                    if s["user_idx"] in train_banks and len(s["events"]) >= seq_len]
    pos_weight  = _compute_pos_weight(epoch0_train, seq_len)
    total_steps = epochs * (STREAMS_PER_EPOCH // BATCH_STREAMS)
    LOGGER.info(f"pos_weight={pos_weight:.2f}  total_steps={total_steps:,}")

    module = Stage2E2EModule(
        encoder, detector, train_data, train_banks, val_banks, val_streams,
        lr=LR, config=config, seq_len=seq_len,
        pos_weight=pos_weight, total_steps=total_steps,
        augment=augment,
    )

    _BEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=str(_BEST_MODELS_DIR),
        filename="epoch_{epoch}_ausc_{val_ausc:.4f}",
        monitor="val_ausc", mode="max", save_top_k=1,
        save_on_train_epoch_end=False, auto_insert_metric_name=False,
    )
    callbacks: list[pl.Callback] = [ckpt]
    if project_cfg["runtime"].get("model_summary", False):
        callbacks.append(ModelSummary(max_depth=project_cfg["runtime"].get("model_summary_depth", 3)))

    steps_per_epoch = STREAMS_PER_EPOCH // BATCH_STREAMS
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=epochs, callbacks=callbacks,
        logger=setup_lightning_wandb(_PROJECT_ROOT, module, project_cfg["runtime"]) or False,
        log_every_n_steps=max(1, min(config.log_every_n_steps, steps_per_epoch)),
        enable_model_summary=False, num_sanity_val_steps=0, precision="32-true",
        gradient_clip_val=GRAD_CLIP, reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(module, val_dataloaders=val_loader)
    LOGGER.info(f"Best checkpoint: {ckpt.best_model_path}")


if __name__ == "__main__":
    # Usage: python -m experiments.ca.train [epochs] [seq_len] [--augment]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    epochs  = int(args[0]) if len(args) > 0 else 3
    seq_len = int(args[1]) if len(args) > 1 else 25
    augment = "--augment" in flags
    train(epochs=epochs, seq_len=seq_len, augment=augment)
