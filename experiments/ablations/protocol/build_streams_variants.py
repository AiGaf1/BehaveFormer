"""
Build protocol-ablation stream variants — Table 2 in the paper.

Three conditions that disable one or both shortcut-prevention mitigations:
  fixed_onset   – t* fixed to L_A (end of prefix) instead of random
  no_same_user  – same-user streams replaced with additional attack streams
                  (legitimate class is single-session only, not same-user)
  naive         – both: fixed t* AND no same-user streams

Under our full protocol, Boundary-Only should achieve AUSC ≈ 0.5.
Each variant re-introduces one or both shortcuts; Boundary-Only AUSC should rise.

Output: ablations/protocol/prepared/<variant>.pkl
Each file has the same format as data/prepared/streams.pkl.

Usage:
    cd <project_root>
    python -m experiments.AaltoDB.ablations.protocol.build_streams_variants
"""

import pickle
import random
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from data.AaltoDB.prepare import load as load_aalto  # noqa: E402
from data.AaltoDB.stage_2 import (  # noqa: E402
from utils.logger import get_logger # noqa: E402

LOGGER = get_logger(__name__)
    ENROLL_END, PREFIX_END, SEED, _build_stream,
)

_OUT_DIR = Path(__file__).resolve().parent / "prepared"


def _build_fixed_onset(prefix_sess, suffix_sess, stream_type, user_idx):
    """t* = L_A (attack starts exactly at the session boundary)."""
    t_star = len(prefix_sess)
    return _build_stream(prefix_sess, suffix_sess, t_star, stream_type, user_idx)


def build_variant(split_data: list, rng: random.Random, variant: str) -> list:
    """
    Build streams for one split under the given protocol variant.

    variant: 'fixed_onset' | 'no_same_user' | 'naive'
    """
    n_users = len(split_data)
    streams = []

    fixed = variant in ("fixed_onset", "naive")
    drop_same_user = variant in ("no_same_user", "naive")

    for user_idx, user_sessions in enumerate(split_data):
        prefix_pool = user_sessions[ENROLL_END:PREFIX_END]
        suffix_pool = user_sessions[PREFIX_END:]

        for p_idx, prefix_sess in enumerate(prefix_pool):
            L_A    = len(prefix_sess)
            t_star = L_A if fixed else rng.randint(0, L_A)

            impostor_idx    = (user_idx + 1 + rng.randint(0, n_users - 2)) % n_users
            impostor_suffix = rng.choice(split_data[impostor_idx][PREFIX_END:])
            streams.append(_build_stream(prefix_sess, impostor_suffix, t_star, "attack", user_idx))

            if drop_same_user:
                # replace same-user stream with another attack stream (different attacker)
                impostor_idx2    = (user_idx + 2 + rng.randint(0, n_users - 2)) % n_users
                impostor_suffix2 = rng.choice(split_data[impostor_idx2][PREFIX_END:])
                streams.append(_build_stream(prefix_sess, impostor_suffix2, t_star, "attack", user_idx))
            else:
                same_suffix = suffix_pool[(p_idx + 1) % len(suffix_pool)]
                streams.append(_build_stream(prefix_sess, same_suffix, t_star, "same_user", user_idx))

    rng.shuffle(streams)
    return streams


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [v for v in ("fixed_onset", "no_same_user", "naive")
               if not (_OUT_DIR / f"{v}.pkl").exists()]
    if not missing:
        LOGGER.info("All variant pickles already exist.  Delete to rebuild.")
        return

    LOGGER.info("Loading Aalto features …")
    train_data, val_data, test_data = load_aalto()

    for variant in missing:
        LOGGER.info(f"Building variant: {variant} …")
        rng = random.Random(SEED)
        result = {
            "train": build_variant(train_data, rng, variant),
            "val":   build_variant(val_data,   rng, variant),
            "test":  build_variant(test_data,  rng, variant),
        }
        out = _OUT_DIR / f"{variant}.pkl"
        with open(out, "wb") as f:
            pickle.dump(result, f)

        for split, items in result.items():
            attack    = sum(1 for s in items if s["stream_type"] == "attack")
            same_user = sum(1 for s in items if s["stream_type"] == "same_user")
            LOGGER.info(f"  {split:6s}  total={len(items):7d}  attack={attack:6d}  same_user={same_user:6d}")
        LOGGER.info(f"  Saved → {out}")


if __name__ == "__main__":
    main()
