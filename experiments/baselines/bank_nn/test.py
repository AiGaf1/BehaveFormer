"""
Baseline: Bank-NN
Score = 1 - max_k cos(z_t, z_k^u)  — nearest-neighbour in the bank, no learned D_phi.

Usage:
    cd <project_root>
    python -m experiments.AaltoDB.baselines.bank_nn.test [stage1_ckpt]
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from data.AaltoDB.prepare import load as load_aalto                              # noqa: E402
from data.AaltoDB.stats import aalto_feature_ranges, aalto_vocab_size  # noqa: E402
from evaluation.metrics import Metric  # noqa: E402
from data.AaltoDB.stage_2 import load_streams            # noqa: E402
from experiments.AaltoDB.stage1_encoder.model import Encoder         # noqa: E402
from experiments.AaltoDB.stage1_encoder.train import (               # noqa: E402
    DROPOUT, EMBED_DIM, HEADS, LFF_FEATURES, NUM_LAYERS, SEQ_LEN,
    _BEST_MODELS_DIR as _STAGE1_BEST_MODELS_DIR,
)
from experiments.AaltoDB.stage2_detector.train import build_banks, BANK_K  # noqa: E402
from utils.logger import get_logger # noqa: E402

LOGGER = get_logger(__name__)


@torch.no_grad()
def make_score_fn(encoder: Encoder, banks: dict, seq_len: int, device: torch.device):
    """Returns a score_fn(stream) -> (n_windows,) using nearest-bank-entry scoring."""
    banks_dev = {uid: embs.to(device) for uid, embs in banks.items()}

    def score_fn(stream: dict) -> torch.Tensor:
        events = stream["events"].astype(np.float32)
        n = len(events)
        if n < seq_len:
            return torch.zeros(0)
        windows = torch.from_numpy(
            np.stack([events[i: i + seq_len] for i in range(n - seq_len + 1)])
        ).to(device)
        z    = encoder(windows)                               # (W, D)
        bank = banks_dev[stream["user_idx"]]                  # (K, D)
        # cosine similarity between each window and each bank entry: (W, K)
        sims = F.cosine_similarity(z.unsqueeze(1), bank.unsqueeze(0), dim=-1)
        return (1.0 - sims.max(dim=1).values).cpu()

    return score_fn


def test(stage1_ckpt: Path | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if stage1_ckpt is None:
        ckpts = sorted(_STAGE1_BEST_MODELS_DIR.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No stage-1 checkpoints in {_STAGE1_BEST_MODELS_DIR}")
        stage1_ckpt = ckpts[-1]
    LOGGER.info(f"Stage-1 checkpoint: {stage1_ckpt}")

    _, _, test_data = load_aalto()
    vocab_size = aalto_vocab_size(test_data)
    ranges     = aalto_feature_ranges(test_data)

    encoder = Encoder(
        seq_len=SEQ_LEN, vocab_size=vocab_size, embed_dim=EMBED_DIM,
        feature_ranges=ranges, lff_features=LFF_FEATURES,
        num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT,
    ).to(device)
    ckpt = torch.load(stage1_ckpt, map_location=device)
    encoder.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
    )
    encoder.eval()

    LOGGER.info("Building test banks …")
    banks = build_banks(encoder, test_data, SEQ_LEN, device, k=BANK_K)

    streams  = load_streams()
    score_fn = make_score_fn(encoder, banks, SEQ_LEN, device)
    results  = Metric.evaluate_streams(score_fn, streams["test"], streams["test_single"])
    Metric.print_results(results, label="Bank-NN")
    return results


if __name__ == "__main__":
    test(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
