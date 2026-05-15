"""
Stage 2 full evaluation on the test split — Stage 1 baseline + Stage 2 detector.

Reports both rows for the paper's ablation:
  - Stage 1 baseline: cosine distance to enrollment template (no detector, no bank)
  - Stage 2 (ours):   bank-aware detector

Usage:
    cd <project_root>
    python -m experiments.stage2_detector.test [stage2_ckpt] [stage1_ckpt]
"""

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from data.AaltoDB.stage_2 import ENROLL_END, load_streams  # noqa: E402
from evaluation.metrics import Metric  # noqa: E402
from evaluation.stage2 import (  # noqa: E402
    build_banks,
    build_user_template,
    score_stream,
    score_stream_baseline,
)
from experiments.loaders import latest_ckpt, load_detector, load_encoder  # noqa: E402
from experiments.stage1_encoder.train import SEQ_LEN  # noqa: E402
from experiments.stage2_detector.train import BANK_K, _BEST_MODELS_DIR  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__)


def test(stage2_ckpt: Path | None = None, stage1_ckpt: Path | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage2_ckpt = stage2_ckpt or latest_ckpt(_BEST_MODELS_DIR)
    LOGGER.info(f"Stage-2 checkpoint: {stage2_ckpt}")

    encoder, test_data = load_encoder(stage1_ckpt, device)
    detector           = load_detector(stage2_ckpt, device, embed_dim=encoder.out_dim)

    LOGGER.info("Loading streams …")
    streams = load_streams()

    # ── Stage 1 baseline: cosine to enrollment template ──────────────────────
    LOGGER.info("Building per-user enrollment templates (Stage 1 baseline) …")
    rng = np.random.default_rng(0)
    templates = {
        uid: build_user_template(encoder, sessions[:ENROLL_END], SEQ_LEN, device, rng=rng)
        for uid, sessions in enumerate(test_data)
    }
    baseline_fn = lambda s: score_stream_baseline(s, encoder, templates, SEQ_LEN, device)  # noqa: E731
    baseline    = Metric.evaluate_streams(baseline_fn, streams["test"], streams["test_single"])
    Metric.print_results(baseline, label="Stage 1 baseline (cosine to template)")

    # ── Stage 2: bank-aware detector ─────────────────────────────────────────
    LOGGER.info("Building test banks …")
    banks = build_banks(encoder, test_data, SEQ_LEN, device, k=BANK_K)
    stage2_fn = lambda s: score_stream(s, encoder, detector, banks, SEQ_LEN, device)  # noqa: E731
    results   = Metric.evaluate_streams(stage2_fn, streams["test"], streams["test_single"])
    Metric.print_results(results, label="Stage 2 (ours)")
    return {"baseline": baseline, "stage2": results}


if __name__ == "__main__":
    s2 = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    s1 = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    test(s2, s1)
