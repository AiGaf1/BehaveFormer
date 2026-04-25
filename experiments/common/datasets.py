import numpy as np
from torch.utils.data import Dataset


class KeystrokeData:
    """Adapter for the active Aalto and HMOG keystroke layouts.

    Aalto stores one keystroke array per session:
        users[user][session] -> keystroke_array

    HMOG stores many windowed sequences per session:
        users[user][session][sequence] -> [keystroke_array, imu_array]
    """

    def __init__(self, users, columns=None, seq_len=None):
        self.users = users
        self.columns = columns
        self.seq_len = seq_len
        self.sessions_per_user = len(users[0])
        self.sequences_per_session = len(self._session_sequences(users[0][0]))

    def __len__(self):
        return len(self.users)

    def iter_sequences(self):
        for user in self.users:
            for session in user:
                for sequence in self._session_sequences(session):
                    yield self.to_model_input(sequence)

    def random_sequence(self, user_id):
        user = self.users[user_id]
        session = user[np.random.randint(len(user))]
        sequences = self._session_sequences(session)
        return self.to_model_input(sequences[np.random.randint(len(sequences))])

    def eval_sequence(self, idx):
        user = idx // (self.sessions_per_user * self.sequences_per_session)
        session = (idx // self.sequences_per_session) % self.sessions_per_user
        sequence = idx % self.sequences_per_session
        return self.to_model_input(self._session_sequences(self.users[user][session])[sequence])

    def eval_count(self):
        return len(self) * self.sessions_per_user * self.sequences_per_session

    def key_vocab_size(self):
        return int(max(sequence[:, -1].max() for sequence in self.iter_sequences())) + 1

    def feature_ranges(self):
        timing = np.concatenate([sequence[:, :-1] for sequence in self.iter_sequences()])
        abs_timing = np.abs(timing)
        return {
            f"timing_{idx}": {
                "min": float(abs_timing[:, idx][abs_timing[:, idx] > 0].min()),
                "max": float(abs_timing[:, idx].max()),
            }
            for idx in range(abs_timing.shape[1])
        }

    def to_model_input(self, sequence):
        keystrokes = self._keystroke_array(sequence)
        if self.columns is not None:
            keystrokes = keystrokes[:, list(self.columns)]
        if self.seq_len is None:
            return keystrokes.astype(np.float32)

        out = np.zeros((self.seq_len, keystrokes.shape[1]), dtype=np.float32)
        length = min(len(keystrokes), self.seq_len)
        out[:length] = keystrokes[:length]
        return out

    @staticmethod
    def _session_sequences(session):
        return (session,) if isinstance(session, np.ndarray) else session

    @staticmethod
    def _keystroke_array(sequence):
        return sequence if isinstance(sequence, np.ndarray) else sequence[0]


class TrainDataset(Dataset):
    """Returns (sequence, user_id) pairs with guaranteed positives per batch.

    Indices are grouped: every block of `sequences_per_user` consecutive indices
    maps to the same user, so the DataLoader's sequential batching always contains
    multiple samples per user. Call reshuffle() between epochs to get new user assignments.
    batch_size must equal users_per_batch * sequences_per_user.
    """

    def __init__(self, data, batch_size, epoch_batch_count, columns=None, seq_len=None, sequences_per_user=4):
        self.data = KeystrokeData(data, columns, seq_len)
        self.sequences_per_user = sequences_per_user
        users_per_batch = batch_size // sequences_per_user
        n_slots = users_per_batch * epoch_batch_count
        # each slot maps to a user; sequences_per_user consecutive indices share a slot
        self._user_ids = np.repeat(np.random.randint(len(self.data), size=n_slots), sequences_per_user)

    def __len__(self):
        return len(self._user_ids)

    def reshuffle(self):
        n_slots = len(self._user_ids) // self.sequences_per_user
        self._user_ids = np.repeat(np.random.randint(len(self.data), size=n_slots), self.sequences_per_user)

    def __getitem__(self, idx):
        user_id = int(self._user_ids[idx])
        return self.data.random_sequence(user_id), user_id


class EvalDataset(Dataset):
    """Flat user x session x sequence index over eval data."""

    def __init__(self, data, columns=None, seq_len=None):
        self.data = KeystrokeData(data, columns, seq_len)

    def __len__(self):
        return self.data.eval_count()

    def __getitem__(self, idx):
        return self.data.eval_sequence(idx)


def sample_user_subset(data, max_users, seed=0):
    if max_users is None or len(data) <= max_users:
        return data
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(data), size=max_users, replace=False))
    return [data[i] for i in indices]
