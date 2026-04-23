import numpy as np
from torch.utils.data import Dataset


def _select(sequence, columns, seq_len=None):
    ks = sequence if isinstance(sequence, np.ndarray) else sequence[0]
    if columns is not None:
        ks = ks[:, list(columns)]
    if seq_len is not None:
        out = np.zeros((seq_len, ks.shape[1]), dtype=np.float32)
        length = min(len(ks), seq_len)
        out[:length] = ks[:length]
        return out
    return ks.astype(np.float32)


class TrainDataset(Dataset):
    """Returns (sequence, user_id) pairs with guaranteed positives per batch.

    Indices are grouped: every block of `sequences_per_user` consecutive indices
    maps to the same user, so the DataLoader's sequential batching always contains
    multiple samples per user. Call reshuffle() between epochs to get new user assignments.
    batch_size must equal users_per_batch * sequences_per_user.
    """

    def __init__(self, data, batch_size, epoch_batch_count, columns=None, seq_len=None, sequences_per_user=4):
        self.data = data
        self.sequences_per_user = sequences_per_user
        users_per_batch = batch_size // sequences_per_user
        n_slots = users_per_batch * epoch_batch_count
        self.columns = columns
        self.seq_len = seq_len
        # each slot maps to a user; sequences_per_user consecutive indices share a slot
        self._user_ids = np.repeat(np.random.randint(len(data), size=n_slots), sequences_per_user)

    def __len__(self):
        return len(self._user_ids)

    def reshuffle(self):
        n_slots = len(self._user_ids) // self.sequences_per_user
        self._user_ids = np.repeat(np.random.randint(len(self.data), size=n_slots), self.sequences_per_user)

    def __getitem__(self, idx):
        user_id = int(self._user_ids[idx])
        session = np.random.randint(len(self.data[user_id]))
        seq     = np.random.randint(len(self.data[user_id][session]))
        return _select(self.data[user_id][session][seq], self.columns, self.seq_len), user_id


class EvalDataset(Dataset):
    """Flat user × session × sequence index over nested eval data."""

    def __init__(self, data, columns=None, seq_len=None):
        self.data = data
        self.n_sessions = len(data[0])
        self.n_seqs = len(data[0][0])
        self.columns = columns
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) * self.n_sessions * self.n_seqs

    def __getitem__(self, idx):
        user    = idx // (self.n_sessions * self.n_seqs)
        session = (idx // self.n_seqs) % self.n_sessions
        seq     = idx % self.n_seqs
        return _select(self.data[user][session][seq], self.columns, self.seq_len)


def scale_features(data, keystroke_scale_map, imu_scale_map):
    """Divide keystroke and IMU columns in-place according to scale maps."""
    for user in data:
        for session in user:
            for i, sequence in enumerate(session):
                ks  = sequence[0].astype(np.float64, copy=True)
                imu = sequence[1].astype(np.float64, copy=True)
                for col, divisor in keystroke_scale_map.items():
                    ks[:, col] /= divisor
                for col, divisor in imu_scale_map.items():
                    imu[:, col] /= divisor
                session[i][0] = ks
                session[i][1] = imu


def sample_user_subset(data, max_users, seed=0):
    if max_users is None or len(data) <= max_users:
        return data
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(data), size=max_users, replace=False))
    return [data[i] for i in indices]
