import numpy as np

# ── Frozen z-score stats over asinh-transformed Aalto training features ──
# Computed once from the 80% training split (~29 M events).
# Applied per-feature inside features.compute: (asinh(x) - μ) / σ.
HOLD_MU    = 0.024208
HOLD_STD   = 0.052522
FLIGHT_MU  = 0.291422
FLIGHT_STD = 0.325895


def aalto_vocab_size(data):
    return int(max(session[:, -1].max() for user in data for session in user)) + 1


def aalto_feature_ranges(data):
    """Per-feature (min, max) used by LFF to build its frequency grid.

    Uses the 1st / 99th percentile of |x| > 0 rather than absolute min/max so
    rare outliers (e.g. a 135-day flight time, asinh+zscored to ~50) don't
    blow up the high-frequency end of the Fourier basis. Periods spanning
    [p1, p99] of typical magnitudes give a usable frequency range across the
    distribution body.
    """
    timing = np.concatenate([session[:, :-1] for user in data for session in user])
    abs_timing = np.abs(timing)
    ranges = {}
    for i in range(abs_timing.shape[1]):
        nz = abs_timing[:, i][abs_timing[:, i] > 0]
        if len(nz) == 0:
            ranges[f"timing_{i}"] = {"min": 1e-3, "max": 1.0}
            continue
        ranges[f"timing_{i}"] = {
            "min": float(np.percentile(nz, 1)),
            "max": float(np.percentile(nz, 99)),
        }
    return ranges
