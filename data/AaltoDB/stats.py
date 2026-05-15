import numpy as np


def aalto_vocab_size(data):
    return int(max(session[:, -1].max() for user in data for session in user)) + 1


def aalto_feature_ranges(data):
    timing = np.concatenate([session[:, :-1] for user in data for session in user])
    abs_timing = np.abs(timing)
    return {
        f"timing_{i}": {
            "min": float(abs_timing[:, i][abs_timing[:, i] > 0].min()),
            "max": float(abs_timing[:, i].max()),
        }
        for i in range(abs_timing.shape[1])
    }
