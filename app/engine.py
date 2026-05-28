"""Authentication engine — model loading + per-window scoring.

Wraps the trained end-to-end model (Stage-1 encoder + ProfileConditionedAccumulator
detector) for real-time scoring of a keystroke stream against an enrolled user's
bank. Bank construction and threshold calibration live in `app.calibration`.

Pipeline (mirrors training/eval exactly):
  raw keystrokes  →  features [hold_s, flight_s, key_id+1]  (data.AaltoDB.features)
  features         →  sliding windows of (seq_len, 3)
  windows          →  encoder  →  L2-normed embeddings z_t  (D,)
  z_seq + bank     →  detector →  per-window logit → sigmoid → p_t  (impostor prob)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from data.AaltoDB.features import compute as compute_features  # noqa: E402
from data.AaltoDB.prepare import load as load_aalto  # noqa: E402
from data.AaltoDB.stats import aalto_feature_ranges, aalto_vocab_size  # noqa: E402
from experiments.ca.model import ProfileConditionedAccumulator  # noqa: E402
from experiments.ca.train import (  # noqa: E402
    DROPOUT,
    HEADS,
    KEY_EMB,
    LFF_FEATURES,
    NUM_LAYERS,
    USE_FILM,
)
from experiments.loaders import latest_ckpt  # noqa: E402
from experiments.verification.model import Encoder  # noqa: E402

_CKPT_DIR = _PROJECT_ROOT / "experiments" / "ca" / "best_models"
BANK_DIR  = Path(__file__).resolve().parent / "banks"
BANK_PATH = BANK_DIR / "user_bank.npz"


def _default_ckpt() -> Path:
    return latest_ckpt(_CKPT_DIR)


def keystrokes_to_features(press_ms: list[float], release_ms: list[float],
                           key_ids: list[int]) -> np.ndarray:
    """Convert raw press/release timestamps (ms) + key codes to model features.

    Returns (n, 3) float32: [hold_time_s, flight_time_s, key_id+1] — identical to
    data.AaltoDB.features.compute, which expects [press, release, key_code].
    """
    raw = np.stack([np.asarray(press_ms, dtype=np.float64),
                    np.asarray(release_ms, dtype=np.float64),
                    np.asarray(key_ids, dtype=np.float64)], axis=1)
    return compute_features(raw)


class AuthEngine:
    """Loads the trained encoder + detector and scores keystroke streams."""

    def __init__(self, ckpt_path: Path | None = None, seq_len: int = 25,
                 device: str | None = None):
        self.seq_len = seq_len
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt_path = ckpt_path or _default_ckpt()

        # Encoder construction needs vocab_size + feature_ranges; derive from the
        # dataset stats (the same the model was trained with).
        train_data, _, _ = load_aalto()
        vocab_size = aalto_vocab_size(train_data)
        ranges     = aalto_feature_ranges(train_data)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        encoder = Encoder(
            seq_len=seq_len, vocab_size=vocab_size, key_emb=KEY_EMB,
            feature_ranges=ranges, lff_features=LFF_FEATURES,
            num_layers=NUM_LAYERS, heads=HEADS, dropout=DROPOUT,
        )
        detector = ProfileConditionedAccumulator(embed_dim=encoder.out_dim, dropout=DROPOUT, use_film=USE_FILM)

        sd = ckpt["state_dict"]
        encoder.load_state_dict({k[len("encoder."):]: v for k, v in sd.items()
                                 if k.startswith("encoder.")})
        detector.load_state_dict({k[len("detector."):]: v for k, v in sd.items()
                                  if k.startswith("detector.")})

        self.encoder  = encoder.to(self.device).eval()
        self.detector = detector.to(self.device).eval()
        self.bank: torch.Tensor | None = None  # (D,) mean-pooled, L2-normed profile
        # Z-norm calibration: per-bank mean/std of genuine p_t. None until set
        # by load_bank (from cal_mu/cal_sigma in the npz) or calibrate_threshold.
        self.znorm_mu: float | None = None
        self.znorm_sigma: float | None = None

    # ── bank (enrollment) ────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_windows(self, windows: np.ndarray) -> torch.Tensor:
        """(N, seq_len, 3) raw features → (N, D) L2-normed embeddings on device."""
        batch = torch.from_numpy(windows.astype(np.float32)).to(self.device)
        return self.encoder(batch)

    @torch.no_grad()
    def set_bank_from_windows(self, raw_windows: np.ndarray) -> None:
        """raw_windows: (K, seq_len, 3) → encode K windows and store the active
        profile as [L2-normed mean, per-dim std] (shape 2D).

        Clears any previously-set znorm calibration (μ, σ apply to a specific bank).
        """
        z = self._encode_windows(raw_windows)                          # (K, D)
        prof_mean = torch.nn.functional.normalize(z.mean(dim=0), dim=-1)  # (D,)
        prof_std  = z.std(dim=0)                                         # (D,)
        self.bank = torch.cat([prof_mean, prof_std])                    # (2D,)
        self.znorm_mu = None
        self.znorm_sigma = None

    def load_bank(self, path: Path = BANK_PATH) -> bool:
        """Load raw windows from npz, re-encode, set as active bank. False if missing.

        Also restores znorm parameters (mu/sigma) if present in the npz.
        """
        if not path.exists():
            return False
        data = np.load(path)
        self.set_bank_from_windows(data["windows"])
        self.znorm_mu = float(data["cal_znorm_mu"]) if "cal_znorm_mu" in data.files else None
        self.znorm_sigma = float(data["cal_znorm_sigma"]) if "cal_znorm_sigma" in data.files else None
        return True

    def to_z(self, p_t: np.ndarray) -> np.ndarray:
        """Convert raw p_t scores to z-scores using the bank's calibrated μ, σ.

        Returns raw p_t unchanged if znorm calibration is not set.
        """
        if self.znorm_mu is None or self.znorm_sigma is None or self.znorm_sigma <= 0:
            return p_t
        return (p_t - self.znorm_mu) / self.znorm_sigma

    @property
    def has_bank(self) -> bool:
        return self.bank is not None

    # ── scoring ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score(self, features: np.ndarray) -> np.ndarray:
        """Score a feature stream (n, 3). Returns p_t (n_windows,) in [0, 1].

        Empty if fewer than seq_len keystrokes, or if no bank is enrolled.
        """
        if self.bank is None or len(features) < self.seq_len:
            return np.zeros(0, dtype=np.float32)
        n = len(features) - self.seq_len + 1
        windows = np.stack([features[i: i + self.seq_len] for i in range(n)])
        z_seq = self._encode_windows(windows)                  # (n, D)
        logits = self.detector(z_seq, self.bank)               # (n,)
        return torch.sigmoid(logits).cpu().numpy()

    def score_latest(self, features: np.ndarray) -> float | None:
        """Return the impostor probability for the most recent window.

        Delegates to score() so the CUSUM accumulator integrates the full stream
        history rather than starting from zero state for each call.
        Returns None if not enough keystrokes or no bank enrolled.
        """
        scores = self.score(features)
        return float(scores[-1]) if len(scores) > 0 else None
