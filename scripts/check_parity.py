"""Numeric parity check: score the same test stream via AuthEngine (inference)
and via the experiments.ca.test path (offline eval), and report any divergence.

Run with:
    python -m scripts.check_parity
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from app.engine import AuthEngine  # noqa: E402
from data.AaltoDB.prepare import load as load_aalto  # noqa: E402
from data.AaltoDB.streams import load_streams  # noqa: E402
from evaluation.ca_banks import build_raw_banks  # noqa: E402
from experiments.ca.train import BANK_K, _chunked_encode  # noqa: E402

SEQ_LEN = 25


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    engine = AuthEngine(seq_len=SEQ_LEN, device=str(device))
    encoder, detector = engine.encoder, engine.detector

    _, _, test_data = load_aalto()
    streams = load_streams()

    banks_raw = build_raw_banks(test_data, SEQ_LEN, k=BANK_K, multi_session=True)
    test_streams = [s for s in streams["test"]
                    if s["user_idx"] in banks_raw and s["stream_type"] == "attack"]
    if not test_streams:
        print("No usable attack stream found"); return
    stream = test_streams[0]
    uid = stream["user_idx"]
    print(f"Stream: user_idx={uid}  n_events={len(stream['events'])}  t*={stream['t_star']}")

    # Bank profile: test path uses _chunked_encode + [mean, std]; AuthEngine does the same.
    raw_bank = banks_raw[uid]
    with torch.no_grad():
        z_bank = _chunked_encode(encoder, raw_bank, device)             # (K, D)
        profile_test = torch.cat([
            F.normalize(z_bank.mean(dim=0), dim=-1), z_bank.std(dim=0)])  # (2D,)
    engine.set_bank_from_windows(raw_bank.numpy())
    profile_engine = engine.bank

    print(f"\nProfile max-diff: {(profile_engine - profile_test).abs().max().item():.3e}")

    # Score via both paths.
    p_engine = engine.score(stream["events"].astype(np.float32))
    events = stream["events"].astype(np.float32)
    windows = torch.from_numpy(
        np.stack([events[i: i + SEQ_LEN] for i in range(len(events) - SEQ_LEN + 1)]))
    with torch.no_grad():
        z_seq = _chunked_encode(encoder, windows, device)
        p_test = torch.sigmoid(detector(z_seq, profile_test)).cpu().numpy()

    print(f"Score lengths: engine={len(p_engine)}  test={len(p_test)}")
    diff = np.abs(p_engine - p_test)
    print(f"Score max-diff:  {diff.max():.3e}")
    print(f"Score mean-diff: {diff.mean():.3e}")
    print(f"First 5 engine: {p_engine[:5]}")
    print(f"First 5 test  : {p_test[:5]}")

    if diff.max() < 1e-5:
        print("\n[OK] PARITY: AuthEngine inference == test-path eval")
    else:
        print("\n[FAIL] DIVERGENCE")


if __name__ == "__main__":
    main()
