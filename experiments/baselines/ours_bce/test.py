"""
Baseline: Ours-BCE — evaluation.

Usage:
    cd <project_root>
    python -m experiments.AaltoDB.baselines.ours_bce.test [stage2_ckpt] [stage1_ckpt]
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.AaltoDB.stage2_detector.test import test as _stage2_test  # noqa: E402

_BEST_MODELS_DIR = Path(__file__).resolve().parent / "best_models"


def test(stage2_ckpt: Path | None = None, stage1_ckpt: Path | None = None):
    if stage2_ckpt is None:
        ckpts = sorted(_BEST_MODELS_DIR.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints in {_BEST_MODELS_DIR}")
        stage2_ckpt = ckpts[-1]
    return _stage2_test(stage2_ckpt=stage2_ckpt, stage1_ckpt=stage1_ckpt)


if __name__ == "__main__":
    s2 = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    s1 = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    test(s2, s1)
