"""
Stage 1 data — encoder pretraining.

Wraps raw Aalto sessions into (window, user_id) pairs for SupConLoss.
Call resample(seed) at the start of each epoch to draw a fresh set of
users and windows.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PureWindowDataset(Dataset):
    """Each epoch: sample `users_per_epoch` users, `pairs_per_user` windows each."""

    def __init__(
        self,
        data: list,
        seq_len: int,
        users_per_epoch: int,
        pairs_per_user: int,
        max_users: int | None = None,
    ):
        if max_users is not None and len(data) > max_users:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(len(data), size=max_users, replace=False))
            data = [data[i] for i in idx]

        self.seq_len         = seq_len
        self.users_per_epoch = users_per_epoch
        self.pairs_per_user  = pairs_per_user

        # pre-extract all valid windows per user (random start each time via resample)
        self.sessions: list[list[np.ndarray]] = [
            [s.astype(np.float32) for s in user_sessions if len(s) >= seq_len]
            for user_sessions in data
        ]
        self._valid_uids = [uid for uid, s in enumerate(self.sessions) if s]
        self.resample(seed=0)

    def resample(self, seed: int) -> None:
        rng  = np.random.default_rng(seed)
        n    = min(self.users_per_epoch, len(self._valid_uids))
        uids = rng.choice(self._valid_uids, size=n, replace=False)

        pairs: list[tuple[np.ndarray, int]] = []
        for uid in uids:
            sessions = self.sessions[uid]
            n_sess   = len(sessions)
            # one window per distinct session if possible; else sample with replacement
            replace  = n_sess < self.pairs_per_user
            sess_idx = rng.choice(n_sess, size=self.pairs_per_user, replace=replace)
            for si in sess_idx:
                sess = sessions[si]
                max_start = len(sess) - self.seq_len
                start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                pairs.append((sess[start : start + self.seq_len], int(uid)))

        windows = np.stack([w for w, _ in pairs])
        labels  = np.array([u for _, u in pairs], dtype=np.int64)
        self.windows = torch.from_numpy(windows)
        self.labels  = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.windows[idx], self.labels[idx]
