from pathlib import Path

from pytorch_lightning.loggers import WandbLogger


def setup_lightning_wandb(project_root: Path, module, rt: dict):
    config = module.config
    if not config.wandb_enabled:
        return None

    logger = WandbLogger(
        project=rt.get("wandb_project") or "BehaveFormer",
        entity=rt.get("wandb_entity") or None,
        name=rt.get("wandb_run_name") or None,
        version=rt.get("wandb_run_id") or None,
        tags=rt.get("wandb_tags") or [],
        log_model="all",
        save_dir=str(project_root),
    )
    logger.experiment.config.update(rt.get("wandb_config") or {}, allow_val_change=True)
    logger.experiment.config.update(
        {
            "module_class_name":         module.__class__.__name__,
            "model_parameter_count":     sum(p.numel() for p in module.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "torch_compile_enabled":     True,
            "optimizer_name":            config.optimizer,
            "weight_decay":              config.weight_decay,
            "muon_adjust_lr_fn":         config.muon_adjust_lr_fn,
            "muon_momentum":             config.muon_momentum,
            "muon_nesterov":             config.muon_nesterov,
            "muon_ns_steps":             config.muon_ns_steps,
        },
        allow_val_change=True,
    )
    logger.watch(module, log="all", log_freq=config.wandb_watch_freq, log_graph=True)
    return logger
