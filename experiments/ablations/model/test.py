"""
Model ablation — Table 4 in the paper.

Evaluates all detector variants against the same frozen encoder and test streams.

Structural ablations (no retraining needed — components zeroed at inference):
  no_bank    – skip all bank interaction; feed [z_t, 0, 0, 0] to MLP
  no_psi     – zero out similarity stats ψ_t
  no_attn    – zero out attention readout v_t and residual z_t-v_t

Bank-size ablations (same trained model, different bank K at inference):
  k_1        – K=1  (single prototype)
  k_small    – K=K_SMALL (smaller than default BANK_K)
  k_large    – K=K_LARGE (larger than default BANK_K)

Loss ablation:
  bce        – load Ours-BCE checkpoint from baselines/ours_bce/best_models/

Usage:
    cd <project_root>
    python -m experiments.AaltoDB.ablations.model.test
"""

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.metrics import Metric  # noqa: E402
from data.AaltoDB.stage_2 import load_streams                    # noqa: E402
from experiments.AaltoDB.loaders import latest_ckpt, load_encoder, load_detector  # noqa: E402
from experiments.AaltoDB.stage1_encoder.train import SEQ_LEN                # noqa: E402
from experiments.AaltoDB.stage2_detector.train import (                     # noqa: E402
from utils.logger import get_logger # noqa: E402

LOGGER = get_logger(__name__)
    BANK_K, build_banks,
    _BEST_MODELS_DIR as _STAGE2_DIR,
)

_BCE_DIR = Path(__file__).resolve().parents[2] / "baselines" / "ours_bce" / "best_models"

K_SMALL = max(1, BANK_K // 2)
K_LARGE = BANK_K * 2


def _make_score_fn(encoder, detector, banks, device, ablation=None):
    @torch.no_grad()
    def score_fn(stream: dict) -> torch.Tensor:
        events = stream["events"].astype(np.float32)
        n = len(events)
        if n < SEQ_LEN:
            return torch.zeros(0)
        windows = np.stack([events[i: i + SEQ_LEN] for i in range(n - SEQ_LEN + 1)])
        uid  = stream["user_idx"]
        bank = banks[uid].to(device).unsqueeze(0).expand(len(windows), -1, -1)
        z    = encoder(torch.from_numpy(windows).to(device))
        return detector(z, bank, ablation=ablation).cpu()
    return score_fn


def _slice_banks(banks: dict, k: int) -> dict:
    return {uid: embs[:k] for uid, embs in banks.items()}


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, test_data = load_encoder(None, device)
    detector           = load_detector(latest_ckpt(_STAGE2_DIR), device)

    streams = load_streams()
    test_streams = streams["test"]
    single       = streams["test_single"]

    # build once with K_LARGE, then slice for smaller variants
    LOGGER.info("Building banks …")
    banks_large   = build_banks(encoder, test_data, SEQ_LEN, device, k=K_LARGE)
    banks_default = _slice_banks(banks_large, BANK_K)
    banks_small   = _slice_banks(banks_large, K_SMALL)
    banks_k1      = _slice_banks(banks_large, 1)

    # load BCE-trained detector
    try:
        det_bce = load_detector(latest_ckpt(_BCE_DIR), device)
    except FileNotFoundError:
        det_bce = None
        LOGGER.info(f"  [warning] No Ours-BCE checkpoint found in {_BCE_DIR} — skipping")

    variants = [
        ("Full model (ours)",              detector, banks_default, None),
        ("z_t only (no bank)",             detector, banks_default, "no_bank"),
        ("No similarity stats ψ_t",        detector, banks_default, "no_psi"),
        ("No attention readout & residual", detector, banks_default, "no_attn"),
        ("Single prototype (K=1)",         detector, banks_k1,      None),
        (f"Smaller bank (K={K_SMALL})",    detector, banks_small,   None),
        (f"Larger bank (K={K_LARGE})",     detector, banks_large,   None),
    ]
    if det_bce is not None:
        variants.append(("BCE loss (no L_TA)", det_bce, banks_default, None))

    rows = {}
    for label, det, banks, ablation in variants:
        LOGGER.info(f"\n── {label} ──")
        score_fn = _make_score_fn(encoder, det, banks, device, ablation=ablation)
        results  = Metric.evaluate_streams(score_fn, test_streams, single)
        rows[label] = results

    # print Table 4
    LOGGER.info("\n\nModel ablation (Table 4)")
    LOGGER.info(f"{'Variant':<40}  {'AUSC':>6}  {'PTCR':>6}  {'EDD':>7}")
    LOGGER.info("─" * 60)
    for label, r in rows.items():
        bold = " *" if label == "Full model (ours)" else "  "
        LOGGER.info(f"{label:<40}  {r['ausc']:6.4f}  {r['ptcr']:6.4f}  {r['edd']:7.1f}{bold}")

    return rows


if __name__ == "__main__":
    test()
