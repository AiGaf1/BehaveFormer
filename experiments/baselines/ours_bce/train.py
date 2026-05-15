"""
Baseline: Ours-BCE
Full bank-aware architecture (same as stage2) trained with standard BCE
instead of the time-aware loss L_TA.  Isolates the contribution of L_TA.

Usage:
    cd <project_root>
    python -m experiments.AaltoDB.baselines.ours_bce.train [epochs] [stage1_ckpt]
"""

import sys
from pathlib import Path

import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.AaltoDB.stage2_detector.train import train as _stage2_train  # noqa: E402

_BEST_MODELS_DIR = Path(__file__).resolve().parent / "best_models"


def _plain_bce(preds, labels, t_star, w_start):
    return F.binary_cross_entropy(preds, labels)


def train(epochs: int = 30, stage1_ckpt: Path | None = None):
    _stage2_train(epochs=epochs, stage1_ckpt=stage1_ckpt,
                  loss_fn=_plain_bce, best_models_dir=_BEST_MODELS_DIR)


if __name__ == "__main__":
    ckpt = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    train(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 30, stage1_ckpt=ckpt)
