"""Shared checkpoint utilities."""

from pathlib import Path


def latest_ckpt(directory: Path) -> Path:
    ckpts = sorted(directory.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {directory}")
    return ckpts[-1]
