import json
import logging
import math
import os
import time
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiments.keystroke.common.loss import TripletLoss

_TREE_WARNING = r"`isinstance\(treespec, LeafSpec\)` is deprecated.*"
_LIGHTNING_LOGGER = logging.getLogger("pytorch_lightning.trainer.connectors.logger_connector.logger_connector")
_LITLOGGER_FILTER = None


class _LitLoggerSuggestionFilter(logging.Filter):
    def filter(self, record):
        return "try installing [litlogger]" not in record.getMessage()


def configure_lightning_environment() -> None:
    global _LITLOGGER_FILTER

    warnings.filterwarnings("ignore", message=_TREE_WARNING)
    if _LITLOGGER_FILTER is None:
        _LITLOGGER_FILTER = _LitLoggerSuggestionFilter()
    if _LITLOGGER_FILTER not in _LIGHTNING_LOGGER.filters:
        _LIGHTNING_LOGGER.addFilter(_LITLOGGER_FILTER)


def recommended_num_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 1) - 1))


def _move_optimizer_to_device(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _tensor_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _json_env(name: str, default):
    raw = os.getenv(name)
    try:
        return json.loads(raw) if raw else default
    except json.JSONDecodeError:
        return default


def load_resume_state(checkpoint_dir: Path, epoch_offset: int):
    if not epoch_offset:
        return math.inf, None, None

    checkpoint = torch.load(Path(checkpoint_dir) / f"training_{epoch_offset}.tar", map_location="cpu")
    return checkpoint["eer"], checkpoint["model_state_dict"], checkpoint["optimizer_state_dict"]


class KeystrokeDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size: int, num_workers: int, pin_memory: bool):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": num_workers > 0,
        }

    def _loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, **self.loader_kwargs)

    def train_dataloader(self):
        return self._loader(self.train_dataset)

    def val_dataloader(self):
        return self._loader(self.val_dataset)


class KeystrokeLightningModule(pl.LightningModule):
    def __init__(self, model_factory, learning_rate: float, compute_val_eer):
        super().__init__()
        self.model_factory = model_factory
        self.compute_val_eer = compute_val_eer
        self.learning_rate = learning_rate
        self.model = model_factory()
        self.loss_fn = TripletLoss()
        self.resume_optimizer_state = None
        self.validation_embeddings = []

    def training_step(self, batch, _):
        anchor, positive, negative = batch
        loss = self.loss_fn(*[self.model(item.float()) for item in (anchor, positive, negative)])
        self.log(
            "train_loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
            batch_size=anchor.size(0),
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=anchor.size(0),
        )
        return loss

    def on_validation_epoch_start(self):
        self.validation_embeddings = []

    def validation_step(self, batch, _):
        self.validation_embeddings.append(self.model(batch.float()).detach())

    def on_validation_epoch_end(self):
        if not self.validation_embeddings:
            return
        eer = self.compute_val_eer(torch.cat(self.validation_embeddings, dim=0))
        self.log("val_eer", eer, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def on_fit_start(self):
        if self.resume_optimizer_state is None:
            return
        optimizer = self.optimizers()
        optimizer.load_state_dict(self.resume_optimizer_state)
        _move_optimizer_to_device(optimizer, self.device)
        self.resume_optimizer_state = None

    def export_model(self):
        exported = self.model_factory()
        exported.load_state_dict({key: value.detach().cpu() for key, value in self.model.state_dict().items()})
        exported.train(self.model.training)
        return exported


class TrainingArtifactsCallback(Callback):
    def __init__(
        self,
        best_model_dir: Path,
        checkpoint_dir: Path,
        dataset_name: str,
        model_name: str = "keystroke",
        epoch_offset: int = 0,
        best_eer: float = math.inf,
        wandb_logger=None,
        resume_from_epoch: int | None = None,
    ):
        super().__init__()
        self.best_model_dir = Path(best_model_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.epoch_offset = epoch_offset
        self.best_eer = best_eer
        self.wandb_logger = wandb_logger
        self.resume_from_epoch = resume_from_epoch
        self._epoch_start = None
        self._last_epoch = epoch_offset
        self._saved_checkpoint_epochs = set()

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = _tensor_float(trainer.callback_metrics.get("train_loss"))
        eer = _tensor_float(trainer.callback_metrics.get("val_eer"))
        if loss is None or eer is None:
            return

        epoch = self.epoch_offset + trainer.current_epoch + 1
        self._last_epoch = epoch
        elapsed = 0.0 if self._epoch_start is None else time.time() - self._epoch_start
        print(f"------> Epoch No: {epoch} - Loss: {loss:>7f} - EER: {eer:>4f} - Time: {elapsed:>2f}")

        if eer < self.best_eer:
            model_path = self.best_model_dir / f"epoch_{epoch}_eer_{eer}.pt"
            torch.save(pl_module.export_model(), model_path)
            print(f"Model saved - EER improved from {self.best_eer} to {eer}")
            self.best_eer = eer
            self._log_artifact("model", model_path, epoch, eer, aliases=["best", f"epoch-{epoch}"])

        if epoch % 50 == 0:
            self._save_checkpoint(trainer, pl_module, epoch)

    def on_train_end(self, trainer, pl_module):
        if self._last_epoch > 0:
            self._save_checkpoint(trainer, pl_module, self._last_epoch)

    def _save_checkpoint(self, trainer, pl_module, epoch: int):
        if epoch in self._saved_checkpoint_epochs:
            return
        ckpt_path = self.checkpoint_dir / f"training_{epoch}"
        ckpt_file = ckpt_path.with_suffix(".ckpt")
        tar_file = ckpt_path.with_suffix(".tar")
        trainer.save_checkpoint(str(ckpt_file))
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": pl_module.model.state_dict(),
                "optimizer_state_dict": trainer.optimizers[0].state_dict(),
                "eer": self.best_eer,
            },
            tar_file,
        )
        self._saved_checkpoint_epochs.add(epoch)
        self._log_artifact(
            "checkpoint",
            ckpt_file,
            epoch,
            self.best_eer,
            aliases=["latest", f"epoch-{epoch}"],
            extra_file=tar_file,
        )

    def _log_artifact(
        self,
        artifact_type: str,
        path: Path,
        epoch: int,
        eer: float,
        aliases: list[str],
        extra_file: Path | None = None,
    ):
        if self.wandb_logger is None:
            return
        import wandb

        metadata = {
            "epoch": epoch,
            "val_eer": eer,
            "dataset": self.dataset_name,
            "model": self.model_name,
            "resume_from_epoch": self.resume_from_epoch,
        }
        artifact = wandb.Artifact(
            name=f"{self.dataset_name}-{self.model_name}-{artifact_type}-{self.wandb_logger.experiment.id}",
            type=artifact_type,
            metadata={key: value for key, value in metadata.items() if value is not None},
        )
        artifact.add_file(str(path))
        if extra_file is not None:
            artifact.add_file(str(extra_file))
        self.wandb_logger.experiment.log_artifact(artifact, aliases=aliases)


def setup_wandb(project_root: Path, model):
    if os.getenv("BEHAVEFORMER_WANDB_ENABLED") != "1":
        return None
    logger = WandbLogger(
        project=os.getenv("BEHAVEFORMER_WANDB_PROJECT") or "BehaveFormer",
        entity=os.getenv("BEHAVEFORMER_WANDB_ENTITY") or None,
        name=os.getenv("BEHAVEFORMER_WANDB_RUN_NAME") or None,
        tags=_json_env("BEHAVEFORMER_WANDB_TAGS_JSON", []),
        log_model=False,
        save_dir=str(project_root),
    )
    logger.experiment.config.update(_json_env("BEHAVEFORMER_WANDB_CONFIG_JSON", {}), allow_val_change=True)
    logger.experiment.config.update(
        {
            "model_class_name": model.__class__.__name__,
            "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        },
        allow_val_change=True,
    )
    logger.watch(model, log="all", log_freq=10)
    return logger


def build_trainer(use_gpu: bool, epochs: int, callback: Callback, wandb_logger):
    return pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        max_epochs=epochs,
        callbacks=[callback],
        logger=wandb_logger or False,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
