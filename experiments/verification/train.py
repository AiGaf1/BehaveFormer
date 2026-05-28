"""
Stage 1: pretrain the encoder f_theta on identity-pure keystroke windows.

Loss: SupConLoss — encourages same-user windows to cluster on the unit sphere.
Validation: SupCon loss on a held-out subset of val users.
Early stopping: val_loss. Checkpoint: best val_loss saved to best_models/.

Usage:
    cd <project_root>
    python -m experiments.verification.train [epochs]
"""

import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*LeafSpec.*")
warnings.filterwarnings("ignore", message=r".*Dynamo does not know how to trace.*")

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary  # noqa: E402
from pytorch_metric_learning.losses import SupConLoss  # noqa: E402
from pytorch_metric_learning.samplers import MPerClassSampler  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from data.AaltoDB.prepare import load as load_aalto  # noqa: E402
from data.AaltoDB.stats import aalto_feature_ranges, aalto_vocab_size  # noqa: E402
from data.AaltoDB.windows import PureWindowDataset  # noqa: E402
from evaluation.metrics import Metric  # noqa: E402
from experiments.config import parse_runtime_config  # noqa: E402
from experiments.verification.model import Encoder  # noqa: E402
from utils.config import seed_training  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.optimizers import MultiOptimizer, build_muon_hybrid  # noqa: E402
from utils.wandb import setup_lightning_wandb  # noqa: E402

LOGGER = get_logger(__name__)
_BEST_MODELS_DIR = Path(__file__).resolve().parent / "best_models"
_CKPT_WARNING    = r"Checkpoint directory .* exists and is not empty\."
_WORKERS_WARNING = r"The .* does not have many workers"

SEQ_LEN          = 50
KEY_EMB          = 8
LFF_FEATURES     = 16
NUM_LAYERS       = 2
HEADS            = 2
DROPOUT          = 0.2
BATCH_SIZE       = 2048
LR               = 1e-3
VAL_USERS        = 512
VAL_EVERY_N      = 10     # compute EER every N epochs
TEMPERATURE      = 0.05
USERS_PER_EPOCH  = 8096   # users sampled per training epoch
SAMPLES_PER_USER = 8     # windows sampled per user per epoch (M-per-class)




class Stage1Module(pl.LightningModule):
    _fwd: Callable

    def __init__(self, encoder: Encoder, train_ds: PureWindowDataset,
                 train_eval_ds: PureWindowDataset, lr: float, config):
        super().__init__()
        self.encoder       = encoder
        self.train_ds      = train_ds
        self.train_eval_ds = train_eval_ds
        self.lr = lr
        self.config = config
        self.loss_fn = SupConLoss(temperature=TEMPERATURE)
        object.__setattr__(self, "_fwd", torch.compile(self.encoder, dynamic=True))

    def train_dataloader(self):
        self.train_ds.resample(seed=self.current_epoch)
        sampler = MPerClassSampler(
            self.train_ds.labels.numpy(), m=SAMPLES_PER_USER,
            batch_size=BATCH_SIZE, length_before_new_iter=len(self.train_ds),
        )
        return DataLoader(
            self.train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=0, pin_memory=True, drop_last=True,
        )

    def training_step(self, batch, _):
        windows, labels = batch
        loss = self.loss_fn(self._fwd(windows), labels)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_embs:   list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def validation_step(self, batch, _):
        windows, labels = batch
        with torch.no_grad():
            z = self._fwd(windows)
        self.log("val_loss", self.loss_fn(z, labels), on_epoch=True, on_step=False, prog_bar=True)
        self._val_embs.append(z.detach().cpu())
        self._val_labels.append(labels.cpu())

    def on_validation_epoch_end(self):
        eer = Metric.eer_from_embeddings(torch.cat(self._val_embs), torch.cat(self._val_labels))
        self.log("val_eer", eer, on_epoch=True, prog_bar=True)

        # Honest train EER: resample held-out train users, run in eval mode
        self.train_eval_ds.resample(seed=10_000 + self.current_epoch)
        with torch.no_grad():
            z = self._fwd(self.train_eval_ds.windows.to(self.device))
        train_eer = Metric.eer_from_embeddings(z.cpu(), self.train_eval_ds.labels)
        self.log("train_eer", train_eer, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(self.encoder.parameters(), lr=self.lr,
                                     weight_decay=self.config.weight_decay)
        return MultiOptimizer(build_muon_hybrid(self.encoder, self.lr, self.config))


def train(epochs: int = 300):
    warnings.filterwarnings("ignore", message=_CKPT_WARNING, category=UserWarning)
    warnings.filterwarnings("ignore", message=_WORKERS_WARNING)
    torch.set_float32_matmul_precision("high")

    project_cfg = json.loads((_PROJECT_ROOT / "config.json").read_text())
    config = parse_runtime_config(project_cfg)
    seed_training(config.seed)

    LOGGER.info("Loading Aalto features …")
    train_data, val_data, _ = load_aalto()
    vocab_size = aalto_vocab_size(train_data)
    ranges     = aalto_feature_ranges(train_data)
    LOGGER.info(f"train users={len(train_data)}  val users={len(val_data)}")

    train_ds      = PureWindowDataset(train_data, seq_len=SEQ_LEN,
                                      users_per_epoch=USERS_PER_EPOCH, pairs_per_user=SAMPLES_PER_USER)
    val_ds        = PureWindowDataset(val_data, seq_len=SEQ_LEN,
                                      users_per_epoch=VAL_USERS, pairs_per_user=SAMPLES_PER_USER,
                                      max_users=VAL_USERS)
    train_eval_ds = PureWindowDataset(train_data, seq_len=SEQ_LEN,
                                      users_per_epoch=VAL_USERS, pairs_per_user=SAMPLES_PER_USER)
    LOGGER.info(f"train windows={len(train_ds):,}  val windows={len(val_ds):,}")

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    encoder = Encoder(seq_len=SEQ_LEN, vocab_size=vocab_size, key_emb=KEY_EMB,
                      feature_ranges=ranges, lff_features=LFF_FEATURES,
                      num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT)
    module = Stage1Module(encoder, train_ds=train_ds, train_eval_ds=train_eval_ds,
                          lr=LR, config=config)

    _BEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = ModelCheckpoint(dirpath=str(_BEST_MODELS_DIR),
                           filename="epoch_{epoch}_eer_{val_eer:.4f}",
                           monitor="val_eer", mode="min", save_top_k=1,
                           save_on_train_epoch_end=False,
                           auto_insert_metric_name=False)
    callbacks: list[pl.Callback] = [ckpt]
    if project_cfg["runtime"].get("model_summary", False):
        callbacks.append(ModelSummary(max_depth=project_cfg["runtime"].get("model_summary_depth", 3)))

    n_train_batches = len(train_ds) // BATCH_SIZE
    pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=epochs, callbacks=callbacks,
        logger=setup_lightning_wandb(_PROJECT_ROOT, module, project_cfg["runtime"]) or False,
        log_every_n_steps=max(1, min(config.log_every_n_steps, n_train_batches)),
        check_val_every_n_epoch=VAL_EVERY_N,
        reload_dataloaders_every_n_epochs=1,
        enable_model_summary=False, num_sanity_val_steps=0, precision="32-true",
    ).fit(module, val_dataloaders=val_loader)
    LOGGER.info(f"Best checkpoint: {ckpt.best_model_path}")


if __name__ == "__main__":
    train(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 300)
