import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from pytorch_metric_learning.losses import SupConLoss
from torch.utils.data import DataLoader

_CHECKPOINT_DIR_WARNING = r"Checkpoint directory .* exists and is not empty\."
_VALID_OPTIMIZERS = {"adamw", "muon_hybrid"}
_VALID_MUON_ADJUST_LR_FNS = {"match_rms_adamw", "original"}
_VALID_WANDB_WATCH_MODES = {"none", "gradients", "parameters", "all"}


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    compile_model: bool
    optimizer: str
    weight_decay: float
    gradient_clip_val: float
    muon_adjust_lr_fn: str
    muon_momentum: float
    muon_nesterov: bool
    muon_ns_steps: int
    num_workers: int
    prefetch_factor: int
    log_every_n_steps: int
    wandb_enabled: bool
    wandb_watch: str
    wandb_watch_freq: int


def seed_training(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _runtime_config(cfg: dict) -> RuntimeConfig:
    rt = cfg["runtime"]
    optimizer = rt["optimizer"]
    if optimizer not in _VALID_OPTIMIZERS:
        raise ValueError(f"optimizer={optimizer!r}; expected one of: {', '.join(sorted(_VALID_OPTIMIZERS))}")
    muon_adjust_lr_fn = rt["muon_adjust_lr_fn"]
    if muon_adjust_lr_fn not in _VALID_MUON_ADJUST_LR_FNS:
        raise ValueError(f"muon_adjust_lr_fn={muon_adjust_lr_fn!r}; expected one of: {', '.join(sorted(_VALID_MUON_ADJUST_LR_FNS))}")
    wandb_watch = rt["wandb_watch"]
    if wandb_watch not in _VALID_WANDB_WATCH_MODES:
        raise ValueError(f"wandb_watch={wandb_watch!r}; expected one of: {', '.join(sorted(_VALID_WANDB_WATCH_MODES))}")
    num_workers = rt["num_workers"]
    if num_workers < 0:
        num_workers = max(1, min(8, (os.cpu_count() or 1) - 1))
    return RuntimeConfig(
        seed=rt["seed"],
        compile_model=rt["compile_model"],
        optimizer=optimizer,
        weight_decay=rt["weight_decay"],
        gradient_clip_val=rt["gradient_clip_val"],
        muon_adjust_lr_fn=muon_adjust_lr_fn,
        muon_momentum=rt["muon_momentum"],
        muon_nesterov=rt["muon_nesterov"],
        muon_ns_steps=rt["muon_ns_steps"],
        num_workers=num_workers,
        prefetch_factor=rt["prefetch_factor"],
        log_every_n_steps=rt["log_every_n_steps"],
        wandb_enabled=rt["wandb_enabled"],
        wandb_watch=wandb_watch if rt["wandb_enabled"] else "none",
        wandb_watch_freq=rt["wandb_watch_freq"],
    )


def _compile_model(model, config):
    if config.compile_model and hasattr(torch, "compile"):
        return torch.compile(model, dynamic=True), True
    return model, False


def _split_optimizer_params(model):
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and name.endswith("weight") and "embedding" not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params


def _build_muon_hybrid(model, learning_rate, config):
    Muon = getattr(torch.optim, "Muon", None)
    if Muon is None:
        raise RuntimeError("Muon optimizer requested, but this PyTorch build does not provide torch.optim.Muon.")

    muon_params, adamw_params = _split_optimizer_params(model)
    optimizers = []
    if muon_params:
        optimizers.append(
            Muon(
                muon_params,
                lr=learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.muon_momentum,
                nesterov=config.muon_nesterov,
                ns_steps=config.muon_ns_steps,
                adjust_lr_fn=config.muon_adjust_lr_fn,
            )
        )
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=learning_rate, weight_decay=config.weight_decay))
    return optimizers[0] if len(optimizers) == 1 else optimizers


class KeystrokeLightningModule(pl.LightningModule):
    _forward_model: Any

    def __init__(
        self,
        model_factory,
        learning_rate: float,
        compute_val_eer,
        train_dataset,
        config,
        loss_fn=None,
        compute_val_metrics=None,
        compute_train_metrics=None,
        train_eval_loader=None,
        metrics_every_n_epochs: int = 5,
    ):
        super().__init__()
        self.model = model_factory()
        forward_model, self.compile_enabled = _compile_model(self.model, config)
        # Keep the compiled callable out of Lightning's child-module registry;
        # self.model is the single checkpointed/optimized module.
        object.__setattr__(self, "_forward_model", forward_model)
        self.automatic_optimization = config.optimizer == "adamw"

        self.config = config
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn or SupConLoss(temperature=0.07)
        self.train_dataset = train_dataset
        self.compute_val_eer = compute_val_eer
        self.compute_val_metrics = compute_val_metrics
        self.compute_train_metrics = compute_train_metrics
        self.train_eval_loader = train_eval_loader
        self.metrics_every_n_epochs = metrics_every_n_epochs
        self.validation_embeddings = []

    def on_train_epoch_start(self):
        self.train_dataset.reshuffle()

    def training_step(self, batch, _):
        sequences, labels = batch
        embeddings = self._forward_model(sequences)
        loss = self.loss_fn(embeddings, labels)

        if not self.automatic_optimization:
            self._step_optimizers(loss)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=self.logger is not None, batch_size=len(labels))
        return loss

    def _step_optimizers(self, loss):
        optimizers = self.optimizers(use_pl_optimizer=False)
        optimizers = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]

        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss)

        if self.config.gradient_clip_val:
            for optimizer in optimizers:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=self.config.gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )

        for optimizer in optimizers:
            optimizer.step()

    def on_validation_epoch_start(self):
        self.validation_embeddings = []

    def validation_step(self, batch, _):
        self.validation_embeddings.append(self._forward_model(batch).detach())

    def on_validation_epoch_end(self):
        val_embeddings = torch.cat(self.validation_embeddings, dim=0)
        eer = self.compute_val_eer(val_embeddings)
        has_logger = self.logger is not None
        self.log("val_eer", eer, on_step=False, on_epoch=True, prog_bar=True, logger=has_logger)

        if (self.current_epoch + 1) % self.metrics_every_n_epochs != 0:
            return

        if self.compute_val_metrics is not None:
            metrics = {f"val_{key}": value for key, value in self.compute_val_metrics(val_embeddings).items() if key != "eer"}
            self.log_dict(metrics, on_epoch=True, logger=has_logger)

        if self.compute_train_metrics is not None and self.train_eval_loader is not None:
            train_embeddings = self._collect_embeddings(self.train_eval_loader)
            metrics = {f"train_{key}": value for key, value in self.compute_train_metrics(train_embeddings).items()}
            self.log_dict(metrics, on_epoch=True, logger=has_logger)

    @torch.no_grad()
    def _collect_embeddings(self, loader):
        was_training = self.model.training
        self.model.eval()
        try:
            embeddings = [self._forward_model(batch.to(self.device)).detach() for batch in loader]
            return torch.cat(embeddings, dim=0)
        finally:
            self.model.train(was_training)

    def configure_optimizers(self):
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.config.weight_decay)
        return _build_muon_hybrid(self.model, self.learning_rate, self.config)


def _setup_wandb(project_root: Path, module: KeystrokeLightningModule, rt: dict):
    config = module.config
    if not config.wandb_enabled:
        return None
    if config.wandb_watch != "none" and module.compile_enabled:
        raise RuntimeError(
            "wandb_watch requires eager execution. "
            "Set compile_model=false in config.json to enable W&B watch safely."
        )

    logger = WandbLogger(
        project=rt.get("wandb_project") or "BehaveFormer",
        entity=rt.get("wandb_entity") or None,
        name=rt.get("wandb_run_name") or None,
        version=rt.get("wandb_run_id") or None,
        tags=rt.get("wandb_tags") or [],
        log_model=False,
        save_dir=str(project_root),
    )
    logger.experiment.config.update(rt.get("wandb_config") or {}, allow_val_change=True)
    logger.experiment.config.update(
        {
            "model_class_name": module.model.__class__.__name__,
            "model_parameter_count": sum(p.numel() for p in module.model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in module.model.parameters() if p.requires_grad),
            "torch_compile_enabled": module.compile_enabled,
            "optimizer_name": config.optimizer,
            "weight_decay": config.weight_decay,
            "gradient_clip_val": config.gradient_clip_val,
            "muon_adjust_lr_fn": config.muon_adjust_lr_fn,
            "muon_momentum": config.muon_momentum,
            "muon_nesterov": config.muon_nesterov,
            "muon_ns_steps": config.muon_ns_steps,
        },
        allow_val_change=True,
    )

    if config.wandb_watch != "none":
        logger.watch(module.model, log=config.wandb_watch, log_freq=config.wandb_watch_freq)
    return logger


def _seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _loader_kwargs(batch_size: int, config, seed_offset: int = 0):
    generator = torch.Generator()
    generator.manual_seed(config.seed + seed_offset)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
    return kwargs


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
    best_model_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", message=_CHECKPOINT_DIR_WARNING, category=UserWarning)
    torch.set_float32_matmul_precision("high")

    project_cfg = json.loads((Path(project_root) / "config.json").read_text())
    config = _runtime_config(project_cfg)
    seed_training(config.seed)
    train_loader = DataLoader(train_dataset, shuffle=False, **_loader_kwargs(batch_size, config, seed_offset=0))
    val_loader = DataLoader(val_dataset, **_loader_kwargs(batch_size, config, seed_offset=1))
    train_eval_loader = (
        DataLoader(train_eval_dataset, **_loader_kwargs(batch_size, config, seed_offset=2))
        if train_eval_dataset is not None
        else None
    )

    module = KeystrokeLightningModule(
        model_factory,
        learning_rate,
        compute_val_eer,
        train_dataset=train_dataset,
        config=config,
        loss_fn=loss_fn,
        compute_val_metrics=compute_val_metrics,
        compute_train_metrics=compute_train_metrics,
        train_eval_loader=train_eval_loader,
        metrics_every_n_epochs=metrics_every_n_epochs,
    )
    best_ckpt = ModelCheckpoint(
        dirpath=str(best_model_dir),
        filename="epoch_{epoch}_eer_{val_eer:.4f}",
        monitor="val_eer",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    rt = project_cfg["runtime"]
    callbacks: list[pl.Callback] = [best_ckpt]
    if rt.get("model_summary", False):
        callbacks.append(ModelSummary(max_depth=rt.get("model_summary_depth", 3)))

    trainer_kwargs = {}
    if config.optimizer == "adamw" and config.gradient_clip_val:
        trainer_kwargs["gradient_clip_val"] = config.gradient_clip_val

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        callbacks=callbacks,
        logger=_setup_wandb(Path(project_root), module, project_cfg["runtime"]) or False,
        log_every_n_steps=max(1, min(config.log_every_n_steps, max(1, len(train_loader)))),
        enable_model_summary=False,
        num_sanity_val_steps=0,
        precision="32-true",
        check_val_every_n_epoch=check_val_every_n_epoch,
        **trainer_kwargs,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
