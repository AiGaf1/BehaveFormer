import numpy as np

from data.AaltoDB.stats import FLIGHT_MU, FLIGHT_STD, HOLD_MU, HOLD_STD


def compute(sequence: np.ndarray) -> np.ndarray:
    """Convert raw [press, release, key] events to model features.

    Returns (n, 3) float32 with columns:
      0: asinh(hold_time_s)  z-scored against frozen Aalto train stats
      1: asinh(flight_time_s) z-scored against frozen Aalto train stats
      2: key + 1  (0 reserved for padding)

    asinh compresses the heavy tail; the per-feature z-score (frozen Aalto
    train stats) calibrates the global scale so genuine self-scores sit near
    zero. The per-window median-of-abs norm in the encoder handles residual
    per-window scale. (A3 ablation: dropping the z-score raised self-scores
    0.03→0.35 and lowered AUSC, so it is kept.)
    """
    press, release, key = sequence.T
    features = np.zeros((len(sequence), 3), dtype=np.float32)
    features[:,  0] = (release - press)             / 1000  # hold (s)
    features[:-1, 1] = (press[1:] - release[:-1])   / 1000  # flight (s)
    features[:,  2] = key + 1                              # 0 reserved for padding
    features[:, 0] = (np.arcsinh(features[:, 0]) - HOLD_MU)   / HOLD_STD
    features[:, 1] = (np.arcsinh(features[:, 1]) - FLIGHT_MU) / FLIGHT_STD
    return features


def apply(raw_data: list) -> list:
    for user_sessions in raw_data:
        for idx, session in enumerate(user_sessions):
            user_sessions[idx] = compute(session)
    return raw_data
