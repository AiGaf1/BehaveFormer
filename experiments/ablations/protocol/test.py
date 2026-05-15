"""
Protocol ablation — Table 2 in the paper.

For each of the 4 protocol conditions (full + 3 variants), evaluates:
  - Boundary-Only AUSC  (should be ≈0.5 only under the full protocol)
  - Ours AUSC           (should be stable across all variants)

Requires:
  - build_streams_variants.py run first
  - Trained boundary_only checkpoint
  - Trained stage1 + stage2 checkpoints

Usage:
    cd <project_root>
    python -m experiments.AaltoDB.ablations.protocol.test
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PREP_DIR     = Path(__file__).resolve().parent / "prepared"
sys.path.insert(0, str(_PROJECT_ROOT))

from data.AaltoDB.prepare import load as load_aalto                              # noqa: E402
from data.AaltoDB.stats import aalto_feature_ranges, aalto_vocab_size  # noqa: E402
from evaluation.metrics import Metric         # noqa: E402
from data.AaltoDB.stage_2 import load_streams            # noqa: E402
from experiments.AaltoDB.stage1_encoder.model import Encoder         # noqa: E402
from experiments.AaltoDB.stage1_encoder.train import (               # noqa: E402
    DROPOUT, EMBED_DIM, HEADS, LFF_FEATURES, NUM_LAYERS, SEQ_LEN,
    _BEST_MODELS_DIR as _STAGE1_DIR,
)
from experiments.AaltoDB.stage2_detector.model import BankAwareDetector  # noqa: E402
from experiments.AaltoDB.stage2_detector.train import (              # noqa: E402
    BANK_K, HIDDEN_DIM, TOP_K_STATS, build_banks,
    _BEST_MODELS_DIR as _STAGE2_DIR,
)
from experiments.AaltoDB.baselines.boundary_only.train import (      # noqa: E402
from utils.logger import get_logger # noqa: E402

LOGGER = get_logger(__name__)
    BoundaryMLP, boundary_features, SEQ_LEN as B_SEQ_LEN,
    _BEST_MODELS_DIR as _BOUNDARY_DIR,
)


def _load_boundary_score_fn(device):
    ckpts = sorted(_BOUNDARY_DIR.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No boundary_only checkpoints in {_BOUNDARY_DIR}")
    ckpt  = torch.load(ckpts[-1], map_location=device)
    model = BoundaryMLP()
    model.load_state_dict(
        {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()
         if k.startswith("model.")}
    )
    model.to(device).eval()

    @torch.no_grad()
    def score_fn(stream: dict) -> torch.Tensor:
        events = stream["events"]
        n = len(events)
        if n < B_SEQ_LEN:
            return torch.zeros(0)
        feats = np.stack([boundary_features(events[i: i + B_SEQ_LEN]) for i in range(n - B_SEQ_LEN + 1)])
        return model(torch.from_numpy(feats).to(device)).cpu()

    return score_fn


def _load_ours_score_fn(encoder, detector, banks, device):
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
        return detector(z, bank).cpu()
    return score_fn


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_data = load_aalto()
    vocab_size = aalto_vocab_size(test_data)
    ranges     = aalto_feature_ranges(test_data)

    # load encoder + detector (trained on full-protocol streams)
    s1_ckpt = sorted(_STAGE1_DIR.glob("*.ckpt"))[-1]
    s2_ckpt = sorted(_STAGE2_DIR.glob("*.ckpt"))[-1]

    encoder = Encoder(
        seq_len=SEQ_LEN, vocab_size=vocab_size, embed_dim=EMBED_DIM,
        feature_ranges=ranges, lff_features=LFF_FEATURES,
        num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT,
    ).to(device)
    s1 = torch.load(s1_ckpt, map_location=device)
    encoder.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in s1["state_dict"].items() if k.startswith("encoder.")}
    )
    encoder.eval()

    detector = BankAwareDetector(
        embed_dim=EMBED_DIM, top_k=TOP_K_STATS,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    ).to(device)
    s2 = torch.load(s2_ckpt, map_location=device)
    detector.load_state_dict(
        {k.removeprefix("detector."): v for k, v in s2["state_dict"].items() if k.startswith("detector.")}
    )
    detector.eval()

    banks = build_banks(encoder, test_data, SEQ_LEN, device, k=BANK_K)

    boundary_fn = _load_boundary_score_fn(device)
    ours_fn     = _load_ours_score_fn(encoder, detector, banks, device)

    # full protocol streams (single-session for FPR not needed here)
    full_streams = load_streams()["test"]

    conditions = {
        "Full protocol (ours)":               full_streams,
        "Fixed t*=L_A (no random onset)":     None,
        "Single-session legit (no same-user)": None,
        "Both removed (naive)":               None,
    }
    variant_keys = {
        "Fixed t*=L_A (no random onset)":     "fixed_onset",
        "Single-session legit (no same-user)": "no_same_user",
        "Both removed (naive)":               "naive",
    }

    # load variant streams
    for label, vkey in variant_keys.items():
        pkl = _PREP_DIR / f"{vkey}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"{pkl} not found – run build_streams_variants.py first")
        with open(pkl, "rb") as f:
            conditions[label] = pickle.load(f)["test"]

    # dummy single-session list (Op.FPR not the focus here)
    single = load_streams()["test_single"]

    rows = {}
    for label, streams in conditions.items():
        LOGGER.info(f"\nEvaluating: {label}")
        b_ausc = Metric.evaluate_streams(boundary_fn, streams, single)["ausc"]
        o_ausc = Metric.evaluate_streams(ours_fn,     streams, single)["ausc"]
        rows[label] = (b_ausc, o_ausc)

    # print Table 2
    LOGGER.info("\n\nProtocol ablation (Table 2)")
    LOGGER.info(f"{'Condition':<42}  {'B.-Only':>7}  {'Ours':>7}")
    LOGGER.info("─" * 60)
    for label, (b, o) in rows.items():
        dagger = "†" if label == "Full protocol (ours)" else " "
        LOGGER.info(f"{label:<42}  {b:7.4f}{dagger}  {o:7.4f}")

    return rows


if __name__ == "__main__":
    test()
