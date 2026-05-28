"""
CA pipeline evaluation on the test split.

Loads the trained Encoder + ProfileConditionedAccumulator from a `train.py`
checkpoint, builds raw enrollment banks for test users, encodes them once, and
scores test streams with the per-stream sigmoid pass.

Usage:
    cd <project_root>
    python -m experiments.ca.test [ckpt]
"""

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from data.AaltoDB.prepare import load as load_aalto  # noqa: E402
from data.AaltoDB.streams import load_streams  # noqa: E402
from data.AaltoDB.stats import aalto_feature_ranges, aalto_vocab_size  # noqa: E402
from evaluation.metrics import Metric  # noqa: E402
from evaluation.ca_banks import build_raw_banks  # noqa: E402
from experiments.loaders import latest_ckpt  # noqa: E402
from experiments.verification.model import Encoder  # noqa: E402
from experiments.ca.model import ProfileConditionedAccumulator  # noqa: E402
from experiments.ca.train import (  # noqa: E402
    BANK_K, DROPOUT, HEADS, KEY_EMB, LFF_FEATURES, NUM_LAYERS, USE_FILM,
    _BEST_MODELS_DIR, _chunked_encode,
)
from utils.logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__)


def _load_models(ckpt_path: Path, seq_len: int, device: torch.device):
    train_data, _, _ = load_aalto()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder = Encoder(
        seq_len=seq_len, vocab_size=aalto_vocab_size(train_data), key_emb=KEY_EMB,
        feature_ranges=aalto_feature_ranges(train_data), lff_features=LFF_FEATURES,
        num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT,
    )
    detector = ProfileConditionedAccumulator(embed_dim=encoder.out_dim, dropout=DROPOUT, use_film=USE_FILM)
    sd = ckpt["state_dict"]
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
    detector.load_state_dict({k[len("detector."):]: v for k, v in sd.items() if k.startswith("detector.")})
    return encoder.to(device).eval(), detector.to(device).eval()


@torch.no_grad()
def _encode_banks(encoder, banks_raw: dict[int, torch.Tensor], device
                  ) -> dict[int, torch.Tensor]:
    """For each user: encode K bank windows → [L2-normed mean, per-dim std] (2D,)."""
    out: dict[int, torch.Tensor] = {}
    for uid, raw in banks_raw.items():       # raw: (K, L, F)
        z = _chunked_encode(encoder, raw, device)            # (K, D)
        prof_mean = torch.nn.functional.normalize(z.mean(dim=0), dim=-1)
        prof_std  = z.std(dim=0)
        out[uid] = torch.cat([prof_mean, prof_std])          # (2D,)
    return out


@torch.no_grad()
def _score_stream(stream, encoder, detector, banks_emb, seq_len, device) -> torch.Tensor:
    events = stream["events"].astype(np.float32)
    if len(events) < seq_len:
        return torch.zeros(0)
    windows = torch.from_numpy(
        np.stack([events[i: i + seq_len] for i in range(len(events) - seq_len + 1)]))
    z_seq = _chunked_encode(encoder, windows, device)
    return torch.sigmoid(detector(z_seq, banks_emb[stream["user_idx"]])).cpu()


def test(ckpt_path: Path | None = None, seq_len: int = 25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ckpt_path or latest_ckpt(_BEST_MODELS_DIR)
    LOGGER.info(f"Checkpoint: {ckpt_path}")

    encoder, detector = _load_models(ckpt_path, seq_len, device)

    LOGGER.info("Loading test data + streams …")
    _, _, test_data = load_aalto()
    streams = load_streams()

    LOGGER.info("Building raw test banks …")
    banks_raw = build_raw_banks(test_data, seq_len, k=BANK_K, multi_session=True)
    banks_emb = _encode_banks(encoder, banks_raw, device)

    test_streams   = [s for s in streams["test"]        if s["user_idx"] in banks_emb]
    single_streams = [s for s in streams["test_single"] if s["user_idx"] in banks_emb]
    LOGGER.info(f"Eval users: {len(banks_emb):,} / {len(test_data):,}  "
                f"streams: test={len(test_streams):,}  single={len(single_streams):,}")

    score_fn = lambda s: _score_stream(s, encoder, detector, banks_emb, seq_len, device)  # noqa: E731
    results = Metric.evaluate_streams(score_fn, test_streams, single_streams)
    Metric.print_results(results, label="Stage 2 e2e (ours)")
    return results


if __name__ == "__main__":
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    test(ckpt)
