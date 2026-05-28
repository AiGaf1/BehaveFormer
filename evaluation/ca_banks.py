"""CA enrollment-bank construction.

Builds per-user RAW-window banks consumed by the e2e detector pipeline.
Bank embedding and per-stream scoring live in `experiments/ca/train.py` and
`app/inference.py`; the cosine-template baseline lives in `ca_baseline.py`.
"""

import numpy as np
import torch

from data.AaltoDB.streams import ENROLL_END


def _nonoverlap_starts(n: int, n_take: int, seq_len: int) -> list[int]:
    """Evenly-spaced window start positions — non-overlapping when n >= n_take*seq_len,
    maximally spread otherwise."""
    if n_take == 1:
        return [0]
    if n >= n_take * seq_len:
        stride = n // n_take
    else:
        stride = max(1, (n - seq_len) // (n_take - 1))
    return [min(i * stride, n - seq_len) for i in range(n_take)]


def build_raw_banks(
    aalto_split: list,
    seq_len: int,
    k: int,
    seed: int = 42,
    multi_session: bool = False,
) -> dict[int, torch.Tensor]:
    """For each user, return a (k, seq_len, n_features) cpu tensor of RAW enrollment windows.

    multi_session=False (default): sample k DISTINCT windows uniformly from
        any valid enrollment window. Users with fewer than k windows skipped.
    multi_session=True: distribute k windows across the user's valid
        enrollment sessions as evenly as possible (floor(k/n) or
        ceil(k/n) per session). Enforces session diversity. Skipped only if
        the user has zero valid enrollment sessions.
    """
    from utils.logger import get_logger
    rng = np.random.default_rng(seed)
    banks: dict[int, torch.Tensor] = {}
    skipped_no_windows = skipped_short = 0

    for user_idx, user_sessions in enumerate(aalto_split):
        if multi_session:
            valid = [s.astype(np.float32) for s in user_sessions[:ENROLL_END] if len(s) >= seq_len]
            if not valid:
                skipped_no_windows += 1
                continue
            base, extra = divmod(k, len(valid))
            sample: list[np.ndarray] = []
            for i, sess in enumerate(valid):
                n_take = base + (1 if i < extra else 0)
                if n_take == 0:
                    continue
                for s in _nonoverlap_starts(len(sess), n_take, seq_len):
                    sample.append(sess[s: s + seq_len])
            banks[user_idx] = torch.from_numpy(np.stack(sample))
        else:
            windows: list[np.ndarray] = []
            for sess in user_sessions[:ENROLL_END]:
                arr = sess.astype(np.float32)
                if len(arr) < seq_len:
                    continue
                for i in range(len(arr) - seq_len + 1):
                    windows.append(arr[i: i + seq_len])
            if not windows:
                skipped_no_windows += 1
                continue
            if len(windows) < k:
                skipped_short += 1
                continue
            chosen_idx = rng.choice(len(windows), size=k, replace=False)
            banks[user_idx] = torch.from_numpy(np.stack([windows[i] for i in chosen_idx]))

    if skipped_no_windows or skipped_short:
        get_logger(__name__).info(
            "build_raw_banks: kept %d / %d users  (skipped no-windows=%d, <%d-windows=%d)",
            len(banks), len(aalto_split), skipped_no_windows, k, skipped_short,
        )
    return banks
