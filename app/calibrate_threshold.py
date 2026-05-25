"""Calibrate the impostor decision threshold for a specific user's bank.

Reports score distributions and recommended thresholds at EER, FRR=1%, FRR=5%, FRR=10%.

Usage:
    python app/calibrate_threshold.py app/banks/bank_20260522_1842.npz [n_users]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inference import AuthEngine, score_distributions  # noqa: E402
from data.AaltoDB.prepare import load as load_aalto  # noqa: E402

SEQ_LEN = 25


def summarize(name, scores):
    s = np.asarray(scores)
    print(f"  {name:14s}  n={len(s):>5}  mean={s.mean():.3f}  median={np.median(s):.3f}"
          f"  p10={np.percentile(s,10):.3f}  p90={np.percentile(s,90):.3f}")


def main(bank_path: str, n_users: int = 30) -> None:
    eng = AuthEngine(seq_len=SEQ_LEN)
    print(f"Checkpoint: {eng.__class__.__name__}  device: {eng.device}\n")

    user_bank_raw = np.load(bank_path)["windows"]
    print(f"User bank: {bank_path}  shape={user_bank_raw.shape}")

    _, val, _ = load_aalto()
    aalto_users = [u for u in val
                   if sum(1 for s in u if len(s) >= SEQ_LEN) >= 6][:n_users]
    print(f"Using {len(aalto_users)} Aalto val users for calibration\n")

    genuine, impostor = score_distributions(eng, user_bank_raw, aalto_users)

    print("Score distributions:")
    summarize("genuine",  genuine)
    summarize("impostor", impostor)

    thresholds = np.linspace(0.001, 0.999, 999)
    frr = np.array([(genuine  >  t).mean() for t in thresholds])
    far = np.array([(impostor <= t).mean() for t in thresholds])

    print("\nRecommended thresholds:")
    eer_i = int(np.argmin(np.abs(frr - far)))
    eer_r = (frr[eer_i] + far[eer_i]) / 2
    print(f"  τ at EER         = {thresholds[eer_i]:.3f}   (FRR=FAR={eer_r*100:.2f}%)")

    for target_frr in (0.01, 0.05, 0.10):
        i = int(np.argmin(np.abs(frr - target_frr)))
        print(f"  τ at FRR={target_frr*100:>2.0f}%      = {thresholds[i]:.3f}   "
              f"(actual FRR={frr[i]*100:.2f}%,  FAR={far[i]*100:.2f}%)")

    g_p90 = float(np.percentile(genuine, 90))
    i_p10 = float(np.percentile(impostor, 10))
    print(f"\n  Genuine 90th percentile:   {g_p90:.3f}")
    print(f"  Impostor 10th percentile:  {i_p10:.3f}")
    gap = i_p10 - g_p90
    if gap > 0:
        print(f"  Clean margin of {gap:.3f} — any τ in [{g_p90:.3f}, {i_p10:.3f}] is safe.")
        print(f"  Suggested τ (middle):      {(g_p90 + i_p10) / 2:.3f}")
    else:
        print(f"  Distributions overlap by {-gap:.3f} — use the EER threshold above.")


if __name__ == "__main__":
    bank = sys.argv[1] if len(sys.argv) > 1 else "app/banks/bank_20260522_1842.npz"
    n    = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    main(bank, n)
