import numpy as np


def compute(sequence: np.ndarray) -> np.ndarray:
    press, release, key = sequence.T
    features = np.zeros((len(sequence), 3), dtype=np.float32)
    features[:, 0] = (release - press) / 1000
    features[:-1, 1] = (press[1:] - release[:-1]) / 1000
    features[:, 2] = key + 1  # 0 reserved for padding
    return features


def apply(raw_data: list) -> list:
    for user_sessions in raw_data:
        for idx, session in enumerate(user_sessions):
            user_sessions[idx] = compute(session)
    return raw_data
