"""
Stage 2 data — detector training.

Two responsibilities:
  1. Build CA streams from raw Aalto sessions and save them as a pickle (run once).
  2. Provide StreamDataset and EnrollmentDataset for training and evaluation.

Per-user session roles (15 sessions each):
  sessions[0:5]   → enrollment  (bank building in stage 2)
  sessions[5:10]  → prefix pool (s_A in stream construction)
  sessions[10:15] → suffix pool (s_B)

Stream types:
  attack     – prefix of A  +  suffix of B≠A   labels: 0 before t*, 1 from t* onward
  same_user  – prefix of A  +  another A suffix labels: 0 throughout
  single     – one full session of A (eval only) labels: 0 throughout

Output pickle keys: 'train', 'val', 'test', 'test_single'
Each stream dict:
  {
    'events':      ndarray(N, 3)  – [hold_time, flight_time, key_id]
    'labels':      ndarray(N,)    – int8, 0 or 1
    't_star':      int            – attack onset (0 for same_user / single)
    'user_idx':    int
    'stream_type': str            – 'attack' | 'same_user' | 'single'
  }
"""

import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from data.AaltoDB.prepare import load as load_aalto  # noqa: E402
from utils.logger import get_logger  # noqa: E402

LOGGER = get_logger(__name__)

_OUT_PATH = Path(__file__).resolve().parent / "prepared" / "streams.pkl"

ENROLL_END = 5   # sessions[0:5]  → enrollment
PREFIX_END = 10  # sessions[5:10] → prefix pool
SEED       = 42


# ── stream construction ───────────────────────────────────────────────────────

def _build_stream(prefix_events, suffix_events, t_star, stream_type, user_idx) -> dict:
    combined = np.concatenate([prefix_events[:t_star], suffix_events], axis=0)
    labels = np.zeros(len(combined), dtype=np.int8)
    if stream_type == "attack":
        labels[t_star:] = 1
    return {"events": combined, "labels": labels, "t_star": t_star,
            "user_idx": user_idx, "stream_type": stream_type}


def build_streams_for_split(split_data: list, rng: random.Random) -> tuple[list, list]:
    """Return (streams, single_streams) for one split."""
    n_users = len(split_data)
    streams, singles = [], []

    for user_idx, user_sessions in enumerate(split_data):
        prefix_pool = user_sessions[ENROLL_END:PREFIX_END]
        suffix_pool = user_sessions[PREFIX_END:]

        singles.append({"events": suffix_pool[0],
                        "labels": np.zeros(len(suffix_pool[0]), dtype=np.int8),
                        "t_star": 0, "user_idx": user_idx, "stream_type": "single"})

        for p_idx, prefix_sess in enumerate(prefix_pool):
            t_star = rng.randint(0, len(prefix_sess))

            same_suffix = suffix_pool[(p_idx + 1) % len(suffix_pool)]
            streams.append(_build_stream(prefix_sess, same_suffix, t_star, "same_user", user_idx))

            impostor_idx = (user_idx + 1 + rng.randint(0, n_users - 2)) % n_users
            impostor_suffix = rng.choice(split_data[impostor_idx][PREFIX_END:])
            streams.append(_build_stream(prefix_sess, impostor_suffix, t_star, "attack", user_idx))

    rng.shuffle(streams)
    return streams, singles


def build(out_path: Path = _OUT_PATH) -> None:
    """Build all splits and save to pickle. Skips if file already exists."""
    if out_path.exists():
        LOGGER.info("Already exists: %s  (delete to rebuild)", out_path)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Loading Aalto features …")
    train_data, val_data, test_data = load_aalto()

    # Per-split RNG so subsetting one split doesn't shift t* / impostor pairings
    # in the other two (matters for controlled ablations).
    train_streams, _         = build_streams_for_split(train_data, random.Random(SEED + 0))
    val_streams,   _         = build_streams_for_split(val_data,   random.Random(SEED + 1))
    test_streams,  test_sing = build_streams_for_split(test_data,  random.Random(SEED + 2))

    result = {"train": train_streams, "val": val_streams,
              "test": test_streams, "test_single": test_sing}

    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    for split, items in result.items():
        attack    = sum(1 for s in items if s["stream_type"] == "attack")
        same_user = sum(1 for s in items if s["stream_type"] == "same_user")
        lengths   = [len(s["events"]) for s in items]
        LOGGER.info(
            "  %-12s  total=%7d  attack=%6d  same_user=%6d  len μ=%.0f  min=%d  max=%d",
            split, len(items), attack, same_user, np.mean(lengths), min(lengths), max(lengths),
        )
    LOGGER.info("Saved → %s", out_path)


def load_streams(path: Path = _OUT_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── datasets ──────────────────────────────────────────────────────────────────

class StreamDataset(Dataset):
    """Sliding-window dataset over CA streams for detector training.

    Each item: (window, label) or (window, label, meta) when include_meta=True.
    label = 1 if any event in the window is post attack onset, else 0.
    """

    def __init__(self, streams: list, seq_len: int,
                 columns: list | None = None, include_meta: bool = False):
        self.seq_len      = seq_len
        self.columns      = columns
        self.include_meta = include_meta
        self._streams     = streams
        self._index: list[tuple[int, int]] = [
            (s_idx, w)
            for s_idx, stream in enumerate(streams)
            if len(stream["events"]) >= seq_len
            for w in range(len(stream["events"]) - seq_len + 1)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        s_idx, w_start = self._index[idx]
        stream = self._streams[s_idx]
        events = stream["events"][w_start: w_start + self.seq_len]
        if self.columns is not None:
            events = events[:, self.columns]
        window = torch.from_numpy(events.astype(np.float32))
        label  = torch.tensor(float(stream["labels"][w_start: w_start + self.seq_len].any()))
        if not self.include_meta:
            return window, label
        return window, label, {
            "t_star":      stream["t_star"],
            "w_start":     w_start,
            "user_idx":    stream["user_idx"],
            "stream_type": stream["stream_type"],
        }


def stream_collate(batch):
    """Collate (window, label, meta) triples from StreamDataset(include_meta=True)."""
    windows  = torch.stack([w for w, _, _ in batch])
    labels   = torch.stack([l for _, l, _ in batch])
    user_idx = torch.tensor([m["user_idx"] for _, _, m in batch], dtype=torch.long)
    t_star   = torch.tensor([m["t_star"]   for _, _, m in batch], dtype=torch.long)
    w_start  = torch.tensor([m["w_start"]  for _, _, m in batch], dtype=torch.long)
    return windows, labels, user_idx, t_star, w_start


class StreamLevelDataset(Dataset):
    """Each item = one full stream's windows + per-window labels + meta.

    __getitem__(i) returns dict with:
      windows  : (n_i, seq_len, F) float32
      labels   : (n_i,) float        — 1 if any event in window is post-attack
      w_starts : (n_i,) long
      t_star   : int
      user_idx : int
    Streams with fewer than seq_len events are filtered out at __init__.

    If `augment_epoch` is set, applies per-user timing augmentation (hold/flight
    scale + IME strip) keyed by (epoch, user_idx). Banks in the training step
    must apply the SAME (epoch, user_idx) params or the bank↔stream
    relationship breaks. See `data.augment.timing`.
    """

    def __init__(self, streams: list, seq_len: int, columns: list | None = None,
                 augment_epoch: int | None = None):
        self.seq_len = seq_len
        self.columns = columns
        self.augment_epoch = augment_epoch
        self._streams = [s for s in streams if len(s["events"]) >= seq_len]

    def __len__(self) -> int:
        return len(self._streams)

    def __getitem__(self, idx: int) -> dict:
        stream = self._streams[idx]
        events = stream["events"]
        if self.columns is not None:
            events = events[:, self.columns]
        events = events.astype(np.float32)
        if self.augment_epoch is not None:
            from data.augment.timing import _params_for, apply_np
            params = _params_for(self.augment_epoch, int(stream["user_idx"]))
            events = apply_np(events, params)
        n = len(events) - self.seq_len + 1
        windows = np.stack([events[i: i + self.seq_len] for i in range(n)])
        labels_arr = stream["labels"]
        labels = np.array(
            [labels_arr[i: i + self.seq_len].any() for i in range(n)],
            dtype=np.float32,
        )
        return {
            "windows":  torch.from_numpy(windows),
            "labels":   torch.from_numpy(labels),
            "w_starts": torch.arange(n, dtype=torch.long),
            "t_star":   int(stream["t_star"]),
            "user_idx": int(stream["user_idx"]),
        }


def stream_session_collate(batch):
    """Flatten a list of B stream dicts into a single batch.

    Returns:
      windows   : (total_windows, seq_len, F)
      labels    : (total_windows,)
      user_idx  : (total_windows,)   — repeated per window
      t_star    : (total_windows,)   — repeated per window
      w_start   : (total_windows,)
      stream_id : (total_windows,)   — 0..B-1, groups windows by stream
    """
    windows  = torch.cat([item["windows"]  for item in batch], dim=0)
    labels   = torch.cat([item["labels"]   for item in batch], dim=0)
    w_start  = torch.cat([item["w_starts"] for item in batch], dim=0)
    user_idx = torch.cat([
        torch.full((len(item["windows"]),), item["user_idx"], dtype=torch.long)
        for item in batch
    ])
    t_star = torch.cat([
        torch.full((len(item["windows"]),), item["t_star"], dtype=torch.long)
        for item in batch
    ])
    stream_id = torch.cat([
        torch.full((len(item["windows"]),), i, dtype=torch.long)
        for i, item in enumerate(batch)
    ])
    return windows, labels, user_idx, t_star, w_start, stream_id


if __name__ == "__main__":
    build()
