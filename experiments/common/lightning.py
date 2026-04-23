import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiments.common.loss import TripletLoss


class _KeystrokeLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_factory,
        learning_rate: float,
        compute_val_eer,
        loss_fn=None,
        compute_val_metrics=None,
        compute_train_metrics=None,
        train_eval_loader=None,
        metrics_every_n_epochs: int = 5,
    ):
        super().__init__()
        self.model_factory = model_factory
        self.compute_val_eer = compute_val_eer
        self.learning_rate = learning_rate
        self.model = model_factory()
        self.loss_fn = loss_fn if loss_fn is not None else TripletLoss()
        self.validation_embeddings = []
        self.compute_val_metrics = compute_val_metrics
        self.compute_train_metrics = compute_train_metrics
        self.train_eval_loader = train_eval_loader
        self.metrics_every_n_epochs = metrics_every_n_epochs

    def training_step(self, batch, _):
        anchor, positive, negative = batch
        loss = self.loss_fn(*[self.model(item.float()) for item in (anchor, positive, negative)])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=anchor.size(0))
        return loss

    def on_validation_epoch_start(self):
        self.validation_embeddings = []

    def validation_step(self, batch, _):
        self.validation_embeddings.append(self.model(batch.float()).detach())

    def on_validation_epoch_end(self):
        if not self.validation_embeddings:
            return

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
        embeddings = [self.model(batch.float().to(self.device)).detach() for batch in loader]
        self.model.train()
        return torch.cat(embeddings, dim=0)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)



def _setup_wandb(project_root: Path, model):
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
        },
        allow_val_change=True,
    )
    logger.watch(model, log="gradients", log_freq=10)
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
):
    best_model_dir = Path(best_model_dir)
    best_model_dir.mkdir(exist_ok=True)

    torch.set_float32_matmul_precision("high")

    num_workers = max(1, min(8, (os.cpu_count() or 1) - 1))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
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
        loss_fn=loss_fn,
        compute_val_metrics=compute_val_metrics,
        compute_train_metrics=compute_train_metrics,
        train_eval_loader=train_eval_loader,
        metrics_every_n_epochs=metrics_every_n_epochs,
    )
    wandb_logger = _setup_wandb(project_root, module.model)

    best_ckpt = ModelCheckpoint(
        dirpath=str(best_model_dir),
        filename="epoch_{epoch}_eer_{val_eer:.4f}",
        monitor="val_eer",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        callbacks=[best_ckpt],
        logger=wandb_logger or False,
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)