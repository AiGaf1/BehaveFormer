"""Diagnose why two different users' banks don't separate.

Tests whether the detector VERIFIES identity (B's typing vs A's bank → high p)
or only detects within-stream DRIFT (needs a genuine→impostor transition to fire).

Uses real Aalto val users (clean format) so the keystroke encoding is exactly
what the model trained on.
"""

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from app.inference import AuthEngine, build_bank_windows  # noqa: E402
from data.AaltoDB.prepare import load as load_aalto  # noqa: E402

SEQ_LEN = 25


def stats(name, p):
    if len(p) == 0:
        print(f"  {name:28s}  (empty)")
        return
    print(f"  {name:28s}  mean={p.mean():.3f}  median={np.median(p):.3f}  "
          f"max={p.max():.3f}  min={p.min():.3f}  n={len(p)}")


def main():
    _, val, _ = load_aalto()
    eng = AuthEngine(seq_len=SEQ_LEN)
    print(f"checkpoint: {eng.__class__.__name__} loaded, device={eng.device}\n")

    # Two distinct users with enough data.
    A, B = 0, 1
    a_sessions = [s for s in val[A] if len(s) >= SEQ_LEN]
    b_sessions = [s for s in val[B] if len(s) >= SEQ_LEN]

    # Bank from user A's first sessions (mirror enrollment: K=5 across sessions).
    a_bank_windows = build_bank_windows(a_sessions[:5], seq_len=SEQ_LEN, k=5)
    eng.set_bank_from_windows(a_bank_windows)

    # Held-out streams (use sessions NOT in the bank).
    a_stream = a_sessions[-1]          # genuine A, vs A bank
    b_stream = b_sessions[-1]          # pure impostor B, vs A bank (no genuine prefix)

    # Attack stream: A prefix (first half) + B suffix (transition mid-stream).
    half = len(a_stream) // 2
    attack = np.concatenate([a_stream[:half], b_stream[:half]], axis=0)
    t_star_window = max(0, half - SEQ_LEN + 1)

    print("=== SCORING (bank = user A) ===")
    p_genuine  = eng.score(a_stream)
    p_impostor = eng.score(b_stream)
    p_attack   = eng.score(attack)

    stats("genuine  (A vs A-bank)",  p_genuine)
    stats("impostor (B vs A-bank)",  p_impostor)
    stats("attack   (A→B vs A-bank)", p_attack)

    print(f"\n  attack t* at window ~{t_star_window}")
    if len(p_attack) > t_star_window + 3:
        before = p_attack[:t_star_window].mean() if t_star_window > 0 else float("nan")
        after  = p_attack[t_star_window:].mean()
        print(f"  attack p before t*: {before:.3f}   after t*: {after:.3f}")

    # The decisive comparison: does pure-impostor separate from genuine?
    print("\n=== VERDICT ===")
    if len(p_genuine) and len(p_impostor):
        sep = p_impostor.mean() - p_genuine.mean()
        print(f"  impostor.mean - genuine.mean = {sep:+.3f}")
        if sep < 0.1:
            print("  → Pure-impostor does NOT separate from genuine.")
            print("    The model detects within-stream DRIFT, not absolute identity.")
        else:
            print("  → Pure-impostor separates. Identity verification works.")

    # Cross-check: encoder geometry. Are A and B even separable in embedding space?
    print("\n=== ENCODER GEOMETRY (cosine sims to A-bank) ===")
    za = eng._encode_windows(np.stack([a_stream[i:i+SEQ_LEN]
                                       for i in range(min(50, len(a_stream)-SEQ_LEN+1))]))
    zb = eng._encode_windows(np.stack([b_stream[i:i+SEQ_LEN]
                                       for i in range(min(50, len(b_stream)-SEQ_LEN+1))]))
    sims_a = (za @ eng.bank.t()).cpu().numpy()  # (n, K)
    sims_b = (zb @ eng.bank.t()).cpu().numpy()
    print(f"  A-windows vs A-bank cosine: mean={sims_a.mean():.3f}  max={sims_a.max(1).mean():.3f}")
    print(f"  B-windows vs A-bank cosine: mean={sims_b.mean():.3f}  max={sims_b.max(1).mean():.3f}")
    print(f"  separation (A_max - B_max): {sims_a.max(1).mean() - sims_b.max(1).mean():+.3f}")


if __name__ == "__main__":
    main()
