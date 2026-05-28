"""Multi-device augmentation for cross-capture-environment generalization.

The Aalto training data was captured via browser keystroke events (web-JS),
producing median hold times ~14 ms and a high proportion of IME tokens (key 229).
Live deployment via tkinter on a desktop OS produces median hold ~100 ms and
zero IME tokens. Models trained on Aalto without augmentation collapse all
tkinter-style inputs to a single region of embedding space (verified across 7
architectures and 3 distinct live typists).

This module provides per-stream augmentation that simulates the tkinter regime:
  - Multiplicative scaling of hold / flight (anisotropic, wide range)
  - IME removal (replace key 229 tokens with random non-IME keys)

Augmentation parameters are keyed by (epoch, user_idx) so a user's stream and
their bank windows receive the same perturbation within a training step — this
keeps the (stream, bank) conditioning relationship consistent.

Important: features in the cached pickles are already asinh+z-scored (see
data/AaltoDB/features.py). To apply a multiplicative scale to the RAW timing,
we undo the z-score+asinh, multiply, then re-apply both. The closed-form
shortcut for large |x| (additive shift in z-score space) is wrong for the body
of the distribution where asinh ≈ x; the explicit round-trip is exact.
"""

import numpy as np

from data.AaltoDB.stats import FLIGHT_MU, FLIGHT_STD, HOLD_MU, HOLD_STD

# Per-stream scale ranges (cover the live hold-mean shift Aalto +0.04 → live +1.6).
HOLD_SCALE_RANGE   = (0.3, 10.0)
FLIGHT_SCALE_RANGE = (0.2, 3.0)
JITTER_STD         = 0.0
IME_STRIP_PROB     = 0.7
# Variance-compression range: shrink each window's timing toward its mean.
# Measured live banks reach within-window hold std 0.02 (vs Aalto floor ~0.04);
# compressing to ~0.1× of Aalto's spread simulates the "metronomic typist" regime
# that the two hardest-to-separate live banks (1337, 1344_A) occupy.
VAR_COMPRESS_RANGE = (0.1, 1.0)

# IME key code in stored features is 229 + 1 = 230 (features.compute applies +1).
_IME_KEY = 230

# Common non-IME keys in Aalto (stored, +1-offset). Used as replacement candidates
# when stripping IME from a stream. Letters E, T, A, O, I, N, S, R, H, L, plus Space.
_REPLACEMENT_KEYS = np.array(
    [33, 70, 85, 66, 80, 79, 74, 78, 84, 73, 83, 76, 77],
    dtype=np.float32,
)


def _params_for(epoch: int, user_idx: int) -> dict:
    """Deterministic per-(epoch, user) augmentation params."""
    seed = (epoch * 10_000_019 + user_idx) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return {
        "hold_scale":    float(rng.uniform(*HOLD_SCALE_RANGE)),
        "flight_scale":  float(rng.uniform(*FLIGHT_SCALE_RANGE)),
        "var_compress":  float(rng.uniform(*VAR_COMPRESS_RANGE)),
        "strip_ime":     bool(rng.random() < IME_STRIP_PROB),
        "noise_seed":    int(rng.integers(0, 2**32 - 1)),
    }


def _scale_z(z: np.ndarray, scale: float, mu: float, std: float) -> np.ndarray:
    """Apply a multiplicative scale `scale` in RAW space to z-scored values.

    z → asinh(scale · sinh(z·std + mu)) − mu, then divide by std.
    Exact round-trip; no small-x / large-x approximation.
    """
    x_raw = np.sinh(z * std + mu)
    x_raw_aug = scale * x_raw
    return (np.arcsinh(x_raw_aug) - mu) / std


def apply_np(events: np.ndarray, params: dict) -> np.ndarray:
    """Apply augmentation to a (n, 3) z-scored feature array.

    Returns a NEW array (input not mutated). Pipeline:
      1. Hold / flight multiplicative scale (per-stream constant) — mean shift.
      2. Variance compression toward per-window mean — simulates tight typists.
      3. Per-position Gaussian jitter (optional).
      4. IME stripping (probabilistic, per-stream).
    """
    out = events.copy().astype(np.float32)
    rng = np.random.default_rng(params["noise_seed"])
    # 1. Hold/flight multiplicative scale (in raw space) — covers the live mean shift.
    out[:, 0] = _scale_z(out[:, 0], params["hold_scale"], HOLD_MU, HOLD_STD)
    out[:, 1] = _scale_z(out[:, 1], params["flight_scale"], FLIGHT_MU, FLIGHT_STD)
    # 2. Variance compression: shrink each timing channel toward its mean. Simulates
    #    the low within-window variance of metronomic live typists (banks 1337/1344_A).
    a = params["var_compress"]
    if a < 1.0:
        for ch in (0, 1):
            m = out[:, ch].mean()
            out[:, ch] = m + a * (out[:, ch] - m)
    # 3. Jitter on z-scored timings (post-scale, simulates OS-level timestamp noise).
    if JITTER_STD > 0:
        out[:, 0] += rng.normal(0, JITTER_STD, size=len(out)).astype(np.float32)
        out[:, 1] += rng.normal(0, JITTER_STD, size=len(out)).astype(np.float32)
    # 4. Strip IME: replace IME tokens with random common non-IME keys.
    if params["strip_ime"]:
        ime_mask = out[:, 2] == _IME_KEY
        n_ime = int(ime_mask.sum())
        if n_ime:
            out[ime_mask, 2] = rng.choice(_REPLACEMENT_KEYS, size=n_ime)
    return out
