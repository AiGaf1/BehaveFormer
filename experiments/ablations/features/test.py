"""
Feature ablation — Table 3 in the paper.

Evaluates three feature configurations on separately trained models:
  ht_ft      – (HT, FT) only        columns=[0, 1], key_id padded with 0
  ht_ft_key  – (HT, FT, k_i)        columns=None  (all 3, headline)
  key_only   – k_i only              columns=[2],  timing padded with 0

Each variant requires its own stage-1 encoder + stage-2 detector trained with
the corresponding feature set.  Checkpoints are expected at:
  ablations/features/<variant>/stage1/best_models/*.ckpt
  ablations/features/<variant>/stage2/best_models/*.ckpt

For the headline variant (ht_ft_key) the stage2 best_models from
stage2_detector/best_models/ are used as the default.

Usage:
    cd <project_root>
    python -m experiments.AaltoDB.ablations.features.test
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_HERE         = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data.AaltoDB.prepare import load as load_aalto  # noqa: E402
from data.AaltoDB.stage_2 import load_streams  # noqa: E402
from data.AaltoDB.stats import aalto_feature_ranges, aalto_vocab_size  # noqa: E402
from evaluation.metrics import Metric  # noqa: E402
from experiments.stage1_encoder.model import Encoder  # noqa: E402
from experiments.stage1_encoder.train import (
    _BEST_MODELS_DIR as _STAGE1_DEFAULT,
)
from experiments.stage1_encoder.train import (  # noqa: E402
    DROPOUT,
    EMBED_DIM,
    HEADS,
    LFF_FEATURES,
    NUM_LAYERS,
    SEQ_LEN,
)
from experiments.stage2_detector.model import BankAwareDetector  # noqa: E402
from experiments.stage2_detector.train import (
    _BEST_MODELS_DIR as _STAGE2_DEFAULT,
)
from experiments.stage2_detector.train import (  # noqa: E402
from utils.logger import get_logger # noqa: E402

LOGGER = get_logger(__name__)
    BANK_K,
    HIDDEN_DIM,
    TOP_K_STATS,
    build_banks,
)

# columns fed to encoder for each feature mode
# HT=col 0, FT=col 1, key_id=col 2
# 'None' means all 3 columns unchanged
FEATURE_MODES = {
    "ht_ft":     [0, 1],   # pad key_id=0 at runtime
    "ht_ft_key": None,     # all 3, no padding
    "key_only":  [2],      # pad HT=FT=0 at runtime
}


def _pad_window(window: np.ndarray, mode: str) -> np.ndarray:
    """Return a (seq_len, 3) array with unused features zeroed."""
    w = window.copy()
    if mode == "ht_ft":
        w[:, 2] = 0.0          # zero key_id → embedding index 0 (padding)
    elif mode == "key_only":
        w[:, 0] = 0.0          # zero HT
        w[:, 1] = 0.0          # zero FT
    return w


def _load_encoder(ckpt_path: Path, vocab_size: int, ranges: dict, device) -> Encoder:
    enc = Encoder(
        seq_len=SEQ_LEN, vocab_size=vocab_size, embed_dim=EMBED_DIM,
        feature_ranges=ranges, lff_features=LFF_FEATURES,
        num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in ckpt["state_dict"].items()
         if k.startswith("encoder.")}
    )
    enc.eval()
    return enc


def _load_detector(ckpt_path: Path, device) -> BankAwareDetector:
    det = BankAwareDetector(
        embed_dim=EMBED_DIM, top_k=TOP_K_STATS,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    det.load_state_dict(
        {k.removeprefix("detector."): v for k, v in ckpt["state_dict"].items()
         if k.startswith("detector.")}
    )
    det.eval()
    return det


def _make_score_fn(encoder, detector, banks, seq_len, device, mode):
    @torch.no_grad()
    def score_fn(stream: dict) -> torch.Tensor:
        events = stream["events"].astype(np.float32)
        n = len(events)
        if n < seq_len:
            return torch.zeros(0)
        windows = np.stack([
            _pad_window(events[i: i + seq_len], mode)
            for i in range(n - seq_len + 1)
        ])
        uid  = stream["user_idx"]
        bank = banks[uid].to(device).unsqueeze(0).expand(len(windows), -1, -1)
        z    = encoder(torch.from_numpy(windows).to(device))
        return detector(z, bank).cpu()
    return score_fn


def run_variant(mode: str, s1_ckpt: Path, s2_ckpt: Path, test_data, vocab_size, ranges,
                streams, device):
    LOGGER.info(f"\n── {mode} ──")
    encoder  = _load_encoder(s1_ckpt, vocab_size, ranges, device)
    detector = _load_detector(s2_ckpt, device)
    banks    = build_banks(encoder, test_data, SEQ_LEN, device, k=BANK_K)
    score_fn = _make_score_fn(encoder, detector, banks, SEQ_LEN, device, mode)
    results  = Metric.evaluate_streams(score_fn, streams["test"], streams["test_single"])
    Metric.print_results(results, label=mode)
    return results


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_data = load_aalto()
    vocab_size = aalto_vocab_size(test_data)
    ranges     = aalto_feature_ranges(test_data)
    streams    = load_streams()

    all_results = {}
    for mode in FEATURE_MODES:
        # look for per-variant checkpoints; fall back to default stage1/stage2
        s1_dir = _HERE / mode / "stage1" / "best_models"
        s2_dir = _HERE / mode / "stage2" / "best_models"
        s1_ckpt = sorted(s1_dir.glob("*.ckpt"))[-1] if s1_dir.exists() and list(s1_dir.glob("*.ckpt")) else sorted(_STAGE1_DEFAULT.glob("*.ckpt"))[-1]
        s2_ckpt = sorted(s2_dir.glob("*.ckpt"))[-1] if s2_dir.exists() and list(s2_dir.glob("*.ckpt")) else sorted(_STAGE2_DEFAULT.glob("*.ckpt"))[-1]
        all_results[mode] = run_variant(mode, s1_ckpt, s2_ckpt, test_data, vocab_size, ranges, streams, device)

    # summary table
    LOGGER.info("\n\nFeature ablation summary (Table 3)")
    LOGGER.info(f"{'Mode':<14}  {'AUSC':>6}  {'PTCR':>6}  {'EDD':>7}")
    LOGGER.info("─" * 38)
    for mode, r in all_results.items():
        LOGGER.info(f"{mode:<14}  {r['ausc']:6.4f}  {r['ptcr']:6.4f}  {r['edd']:7.1f}")

    return all_results


if __name__ == "__main__":
    test()
