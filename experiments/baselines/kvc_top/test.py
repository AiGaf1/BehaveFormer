"""
Baseline: KVC-Top — evaluation.

Usage:
    cd <project_root>
    python -m experiments.AaltoDB.baselines.kvc_top.test [checkpoint.ckpt]
"""

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from data.AaltoDB.prepare import load as load_aalto                              # noqa: E402
from data.AaltoDB.stats import aalto_feature_ranges, aalto_vocab_size  # noqa: E402
from experiments.baselines.kvc_top.model import KeystrokeModel, KeystrokeModelBCE       # noqa: E402
from evaluation.metrics import Metric  # noqa: E402
from data.AaltoDB.stage_2 import load_streams            # noqa: E402
from experiments.AaltoDB.baselines.kvc_top.train import (            # noqa: E402
from utils.logger import get_logger # noqa: E402

LOGGER = get_logger(__name__)
    EMBED_DIM, SEQ_LEN, _BEST_MODELS_DIR,
)


def make_score_fn(model: KeystrokeModelBCE, device: torch.device, batch_size: int = 1024):
    model.eval()

    @torch.no_grad()
    def score_fn(stream: dict) -> torch.Tensor:
        events = stream["events"].astype(np.float32)
        n = len(events)
        if n < SEQ_LEN:
            return torch.zeros(0)
        windows = np.stack([events[i: i + SEQ_LEN] for i in range(n - SEQ_LEN + 1)])
        scores = []
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start: start + batch_size]).to(device)
            token_probs = model(batch)           # (B, L)
            scores.append(token_probs.mean(dim=1).cpu())
        return torch.cat(scores)

    return score_fn


def test(ckpt_path: Path | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ckpt_path is None:
        ckpts = sorted(_BEST_MODELS_DIR.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints in {_BEST_MODELS_DIR}")
        ckpt_path = ckpts[-1]
    LOGGER.info(f"Checkpoint: {ckpt_path}")

    _, _, test_data = load_aalto()
    vocab_size = aalto_vocab_size(test_data)
    ranges     = aalto_feature_ranges(test_data)

    backbone = KeystrokeModel(SEQ_LEN, vocab_size, EMBED_DIM, ranges)
    model    = KeystrokeModelBCE(backbone)
    ckpt     = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(
        {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    )
    model.to(device)

    streams  = load_streams()
    score_fn = make_score_fn(model, device)
    results  = Metric.evaluate_streams(score_fn, streams["test"], streams["test_single"])
    Metric.print_results(results, label="KVC-Top")
    return results


if __name__ == "__main__":
    test(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
