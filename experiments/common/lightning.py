import json
import os
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiments.common.loss import triplet_loss

_CHECKPOINT_DIR_WARNING = r"Checkpoint directory .* exists and is not empty\."
_VALID_WANDB_WATCH_MODES = {"none", "gradients", "parameters", "all"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _build_model_runner(model):
    compile_enabled = _env_flag("BEHAVEFORMER_TORCH_COMPILE", True)
    if compile_enabled and hasattr(torch, "compile"):
        return torch.compile(model, dynamic=True), True
    return model, False


def _resolve_wandb_watch_mode() -> str:
    watch_mode = os.getenv("BEHAVEFORMER_WANDB_WATCH", "none").strip().lower()
    if watch_mode not in _VALID_WANDB_WATCH_MODES:
        valid_modes = ", ".join(sorted(_VALID_WANDB_WATCH_MODES))
        raise ValueError(f"Unsupported BEHAVEFORMER_WANDB_WATCH={watch_mode!r}; expected one of: {valid_modes}")
    return watch_mode


def _configure_warning_filters():
    warnings.filterwarnings("ignore", message=_CHECKPOINT_DIR_WARNING, category=UserWarning)


class _KeystrokeLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_factory,
        learning_rate: float,
        compute_val_eer,
        train_dataset,
        loss_fn=None,
        compute_val_metrics=None,
        compute_train_metrics=None,
        train_eval_loader=None,
        metrics_every_n_epochs: int = 5,
    ):
        super().__init__()
        self.compute_val_eer = compute_val_eer
        self.learning_rate = learning_rate
        self.model = model_factory()
        self._model_runner, self.compile_enabled = _build_model_runner(self.model)
        self.loss_fn = loss_fn if loss_fn is not None else triplet_loss()
        self.validation_embeddings = []
        self.train_dataset = train_dataset
        self.compute_val_metrics = compute_val_metrics
        self.compute_train_metrics = compute_train_metrics
        self.train_eval_loader = train_eval_loader
        self.metrics_every_n_epochs = metrics_every_n_epochs

    def on_train_epoch_start(self):
        self.train_dataset.reshuffle()

    def training_step(self, batch, _):
        sequences, labels = batch
        embeddings = self._model_runner(sequences)
        loss = self.loss_fn(embeddings, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(labels))
        return loss

    def on_validation_epoch_start(self):
        self.validation_embeddings = []

    def validation_step(self, batch, _):
        self.validation_embeddings.append(self._model_runner(batch).detach())

    def on_validation_epoch_end(self):
        val_embeddings = torch.cat(self.validation_embeddings, dim=0)
        eer = self.compute_val_eer(val_embeddings)
        self.log("val_eer", eer, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        epoch = self.current_epoch + 1
        if epoch % self.metrics_every_n_epochs != 0:
            return

        if self.compute_val_metrics is not None:
            val_metrics = self.compute_val_metrics(val_embeddings)
            self.log_dict(
                {f"val_{k}": v for k, v in val_metrics.items() if k != "eer"},
                on_epoch=True,
                logger=True,
            )

        if self.compute_train_metrics is not None and self.train_eval_loader is not None:
            train_embeddings = self._collect_embeddings(self.train_eval_loader)
            train_metrics = self.compute_train_metrics(train_embeddings)
            self.log_dict({f"train_{k}": v for k, v in train_metrics.items()}, on_epoch=True, logger=True)

    @torch.no_grad()
    def _collect_embeddings(self, loader):
        self.model.eval()
        embeddings = [self._model_runner(batch.to(self.device)).detach() for batch in loader]
        self.model.train()
        return torch.cat(embeddings, dim=0)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)



def _setup_wandb(project_root: Path, model, compile_enabled: bool):
    if os.getenv("BEHAVEFORMER_WANDB_ENABLED") != "1":
        return None

    tags_raw = os.getenv("BEHAVEFORMER_WANDB_TAGS_JSON")
    tags = json.loads(tags_raw) if tags_raw else []
    
    config_raw = os.getenv("BEHAVEFORMER_WANDB_CONFIG_JSON")
    extra_config = json.loads(config_raw) if config_raw else {}

    logger = WandbLogger(
        project=os.getenv("BEHAVEFORMER_WANDB_PROJECT") or "BehaveFormer",
        entity=os.getenv("BEHAVEFORMER_WANDB_ENTITY") or None,
        name=os.getenv("BEHAVEFORMER_WANDB_RUN_NAME") or None,
        version=os.getenv("BEHAVEFORMER_WANDB_RUN_ID") or None,
        tags=tags,
        log_model=False,
        save_dir=str(project_root),
    )
    logger.experiment.config.update(extra_config, allow_val_change=True)
    logger.experiment.config.update(
        {
            "model_class_name": model.__class__.__name__,
            "model_parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "torch_compile_enabled": compile_enabled,
        },
        allow_val_change=True,
    )
    watch_mode = _resolve_wandb_watch_mode()
    if watch_mode != "none":
        if compile_enabled:
            raise RuntimeError(
                "BEHAVEFORMER_WANDB_WATCH requires eager execution. "
                "Re-run with BEHAVEFORMER_TORCH_COMPILE=0 to enable W&B watch safely."
            )
        logger.watch(model, log=watch_mode, log_freq=int(os.getenv("BEHAVEFORMER_WANDB_WATCH_FREQ", 100)))
    return logger


def run_keystroke_training(
    *,
    project_root,
    best_model_dir,
    train_dataset,
    val_dataset,
    batch_size,
    learning_rate,
    epochs,
    model_factory,
    compute_val_eer,
    loss_fn=None,
    compute_val_metrics=None,
    compute_train_metrics=None,
    train_eval_dataset=None,
    metrics_every_n_epochs: int = 5,
    check_val_every_n_epoch: int = 1,
):
    best_model_dir = Path(best_model_dir)
    best_model_dir.mkdir(exist_ok=True)
    _configure_warning_filters()

    torch.set_float32_matmul_precision("high")

    default_num_workers = max(1, min(8, (os.cpu_count() or 1) - 1))
    num_workers = int(os.getenv("BEHAVEFORMER_NUM_WORKERS", default_num_workers))
    prefetch_factor = int(os.getenv("BEHAVEFORMER_PREFETCH_FACTOR", 4)) if num_workers > 0 else None
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": prefetch_factor,

    }
    # shuffle=False: TrainDataset groups consecutive indices by user for guaranteed positives per batch
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    train_eval_loader = (
        DataLoader(train_eval_dataset, **loader_kwargs)
        if train_eval_dataset is not None
        else None
    )

    module = _KeystrokeLightningModule(
        model_factory,
        learning_rate,
        compute_val_eer,
        train_dataset=train_dataset,
        loss_fn=loss_fn,
        compute_val_metrics=compute_val_metrics,
        compute_train_metrics=compute_train_metrics,
        train_eval_loader=train_eval_loader,
        metrics_every_n_epochs=metrics_every_n_epochs,
    )
    wandb_logger = _setup_wandb(project_root, module.model, module.compile_enabled)

    best_ckpt = ModelCheckpoint(
        dirpath=str(best_model_dir),
        filename="epoch_{epoch}_eer_{val_eer:.4f}",
        monitor="val_eer",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    requested_log_every_n_steps = int(os.getenv("BEHAVEFORMER_LOG_EVERY_N_STEPS", 50))
    log_every_n_steps = max(1, min(requested_log_every_n_steps, max(1, len(train_loader))))

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        callbacks=[best_ckpt],
        logger=wandb_logger or False,
        log_every_n_steps=log_every_n_steps,
        enable_model_summary=False,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        precision="32-true",
        check_val_every_n_epoch=check_val_every_n_epoch,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
