"""Bank construction and enrollment-session I/O.

`build_bank_windows` slices a session into enrollment windows;
`save_session_csv`/`load_session_csv` persist/read the raw `key,hold,flight` CSV.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np

from app.keymap import key_code


def build_bank_windows(session: np.ndarray, seq_len: int, k: int) -> np.ndarray:
    """Split one enrollment session into k evenly-spread (seq_len, 3) windows.

    Windows are non-overlapping when the session is long enough (n >= k*seq_len),
    maximally spread (with overlap) otherwise. session: (n, 3) feature array.
    Returns (k, seq_len, 3).
    """
    n = len(session)
    if n < seq_len:
        raise ValueError(f"Need at least {seq_len} keystrokes; got {n}")
    if n < k * seq_len:
        warnings.warn(
            f"Session has {n} keystrokes (< k*seq_len = {k * seq_len}); windows will "
            f"overlap. More enrollment typing improves authentication accuracy.",
            stacklevel=2,
        )
    # Spread k window-start positions evenly across the session. With enough
    # keystrokes the windows tile the session without overlap (stride = one full
    # chunk); otherwise they slide by the largest gap that still fits k windows.
    if n >= k * seq_len:
        stride = n // k
    else:
        stride = max(1, (n - seq_len) // max(1, k - 1))

    starts = [min(i * stride, n - seq_len) for i in range(k)]
    windows = [session[start: start + seq_len] for start in starts]
    return np.stack(windows).astype(np.float32)


def save_session_csv(path: Path, key_names: list[str],
                     hold_ms: list[float], flight_ms: list[float]) -> None:
    """Persist a full enrollment session as a plain `key,hold_ms,flight_ms` CSV.

    Normalization-agnostic and directly reusable as an experiment dataset sample:
    `load_session_csv` reads it straight back into the model's three features.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "hold_ms", "flight_ms"])
        for k, h, fl in zip(key_names, hold_ms, flight_ms):
            w.writerow([k, f"{h:.3f}", f"{fl:.3f}"])


def load_session_csv(path: Path) -> np.ndarray:
    """Read a `key,hold_ms,flight_ms` CSV → (n, 3) [hold_ms, flight_ms, keycode].

    Keys are mapped to JS keyCodes via `app.keymap.key_code`. Featurize with
    `app.engine.holdflight_to_features`.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = np.zeros((len(rows), 3), dtype=np.float64)
    for i, r in enumerate(rows):
        out[i] = (float(r["hold_ms"]), float(r["flight_ms"]), key_code(r["key"]))
    return out
