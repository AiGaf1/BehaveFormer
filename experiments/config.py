import os
from dataclasses import dataclass

_VALID_OPTIMIZERS = {"adamw", "muon_hybrid"}
_VALID_MUON_ADJUST_LR_FNS = {"match_rms_adamw", "original"}
_VALID_WANDB_WATCH_MODES = {"none", "gradients", "parameters", "all"}


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    optimizer: str
    weight_decay: float
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


def parse_runtime_config(cfg: dict) -> RuntimeConfig:
    rt = cfg["runtime"]
    optimizer = rt["optimizer"]
    if optimizer not in _VALID_OPTIMIZERS:
        raise ValueError(f"optimizer={optimizer!r}; expected one of: {', '.join(sorted(_VALID_OPTIMIZERS))}")
    muon_adjust_lr_fn = rt["muon_adjust_lr_fn"]
    if muon_adjust_lr_fn not in _VALID_MUON_ADJUST_LR_FNS:
        raise ValueError(
            f"muon_adjust_lr_fn={muon_adjust_lr_fn!r}; expected one of: {', '.join(sorted(_VALID_MUON_ADJUST_LR_FNS))}"
        )
    wandb_watch = rt["wandb_watch"]
    if wandb_watch not in _VALID_WANDB_WATCH_MODES:
        raise ValueError(f"wandb_watch={wandb_watch!r}; expected one of: {', '.join(sorted(_VALID_WANDB_WATCH_MODES))}")
    num_workers = rt["num_workers"]
    if num_workers < 0:
        num_workers = max(1, min(8, (os.cpu_count() or 1) - 1))
    return RuntimeConfig(
        seed=rt["seed"],
        optimizer=optimizer,
        weight_decay=rt["weight_decay"],
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
