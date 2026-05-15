"""
Baseline: Boundary-Only — evaluation.
Expected result: AUSC ≈ 0.5 (chance), confirming the shortcut-free protocol.

Usage:
    cd <project_root>
    python -m experiments.AaltoDB.baselines.boundary_only.test [checkpoint.ckpt]
"""

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.metrics import Metric  # noqa: E402
from data.AaltoDB.stage_2 import load_streams                   # noqa: E402
from experiments.AaltoDB.baselines.boundary_only.train import (             # noqa: E402
from utils.logger import get_logger # noqa: E402

LOGGER = get_logger(__name__)
    BoundaryMLP, boundary_features, SEQ_LEN, _BEST_MODELS_DIR,
)


def make_score_fn(model: BoundaryMLP, device: torch.device):
    model.eval()

    @torch.no_grad()
    def score_fn(stream: dict) -> torch.Tensor:
        events = stream["events"]
        n = len(events)
        if n < SEQ_LEN:
            return torch.zeros(0)
        feats = np.stack([
            boundary_features(events[i: i + SEQ_LEN])
            for i in range(n - SEQ_LEN + 1)
        ])
        x = torch.from_numpy(feats).to(device)
        return model(x).cpu()

    return score_fn


def test(ckpt_path: Path | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ckpt_path is None:
        ckpts = sorted(_BEST_MODELS_DIR.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints in {_BEST_MODELS_DIR}")
        ckpt_path = ckpts[-1]
    LOGGER.info(f"Checkpoint: {ckpt_path}")

    ckpt   = torch.load(ckpt_path, map_location=device)
    model  = BoundaryMLP()
    model.load_state_dict(
        {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    )
    model.to(device)

    streams  = load_streams()
    score_fn = make_score_fn(model, device)
    results  = Metric.evaluate_streams(score_fn, streams["test"], streams["test_single"])
    Metric.print_results(results, label="Boundary-Only")
    return results


if __name__ == "__main__":
    test(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
