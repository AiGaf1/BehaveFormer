"""Authentication engine — model loading + per-window scoring.

Wraps the trained end-to-end model (Stage-1 encoder + ProfileConditionedAccumulator
detector) for real-time scoring of a keystroke stream against an enrolled user's
bank. Bank construction and enrollment-session I/O live in `app.bank_io`.

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

from data.AaltoDB.features import compute_features, norm_hold_flight  # noqa: E402
from experiments.ca.model import ProfileConditionedAccumulator, build_profile  # noqa: E402
from experiments.loaders import latest_ckpt  # noqa: E402
from experiments.verification.model import Encoder  # noqa: E402

# Model hyper-parameters for the shipped checkpoint, frozen here so the app does
# not import experiments.ca.train (which pulls in pytorch_lightning/wandb). Source
# of truth: experiments/ca/train.py — the film1_std1_lff1_cusum variant the bundled
# checkpoint was trained with.
LFF_FEATURES = 16
HEADS        = 2
DROPOUT      = 0.1
USE_FILM     = True
USE_STD      = True
USE_LFF      = True
ACCUMULATOR  = "cusum"

# DESKTOP CA model (FiLM+std+cusum detector trained on a frozen desktop encoder).
# The app targets desktop keyboards, so it uses the desktop encoder + desktop
# feature normalization. The mobile model collapsed distinct desktop users
# (inter-bank cosine ~0.98, AUC ~0.48); the desktop model separates them.
_CKPT_DIR = _PROJECT_ROOT / "experiments" / "ca" / "best_models" / "film1_std1_lff1_cusum_desktop_frozen"
BANK_DIR  = Path(__file__).resolve().parent / "banks"

# Frozen Dhakal-desktop train-split stats (asinh space), the normalization the
# desktop encoder/detector were trained with. Live features MUST use these, not
# the mobile DEFAULT_STATS, or the encoder sees an out-of-distribution scale.
DESKTOP_STATS = (0.114648, 0.087033, 0.115861, 0.219354)


def keystrokes_to_features(press_ms: list[float], release_ms: list[float],
                           key_ids: list[int]) -> np.ndarray:
    """Convert raw press/release timestamps (ms) + key codes to model features.

    Returns (n, 3) float32: [hold_time_s, flight_time_s, key_id+1]. Normalized
    with DESKTOP_STATS so the desktop encoder sees its native feature scale.
    """
    raw = np.stack([np.asarray(press_ms, dtype=np.float64),
                    np.asarray(release_ms, dtype=np.float64),
                    np.asarray(key_ids, dtype=np.float64)], axis=1)
    return compute_features(raw, stats=DESKTOP_STATS)


def holdflight_to_features(hold_ms, flight_ms, key_ids) -> np.ndarray:
    """Per-keystroke hold/flight (ms) + key codes → (n, 3) model features.

    Thin ms→s adapter over `data.AaltoDB.features.norm_hold_flight` (the same
    normalization the data preprocessing uses), for the hold/flight persisted in
    the enrollment CSV (see app.bank_io.load_session_csv).
    """
    return norm_hold_flight(np.asarray(hold_ms, dtype=np.float64) / 1000.0,
                            np.asarray(flight_ms, dtype=np.float64) / 1000.0,
                            key_ids, stats=DESKTOP_STATS)


class AuthEngine:
    """Loads the trained encoder + detector and scores keystroke streams."""

    def __init__(self, ckpt_path: Path | None = None, seq_len: int = 25,
                 device: str | None = None):
        self.seq_len = seq_len
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt_path = ckpt_path or latest_ckpt(_CKPT_DIR)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        enc_sd = {k.removeprefix("encoder.").removeprefix("backbone."): v
                  for k, v in sd.items() if k.startswith("encoder.")}

        # Infer encoder shape from the checkpoint so mobile/desktop variants (which
        # differ in vocab_size / layers / seq_len) all load cleanly. The LFF freq
        # buffer is restored from the checkpoint, so feature_ranges only needs the
        # right STRUCTURE (number of timing features) — frozen Aalto values below so
        # the app needs neither the dataset nor data.AaltoDB.prepare/stats at runtime.
        ranges     = {
            "timing_0": {"min": 0.004004073329269886, "max": 2.891751766204834},
            "timing_1": {"min": 0.013044270686805248, "max": 3.8859922885894775},
        }
        vocab_size = enc_sd["key_embedding.weight"].shape[0]
        key_emb    = enc_sd["key_embedding.weight"].shape[1]
        ck_seq     = enc_sd["pos_encoding"].shape[0]
        n_layers   = max(int(k.split("layers.")[1].split(".")[0])
                         for k in enc_sd if "layers." in k) + 1

        encoder = Encoder(
            seq_len=ck_seq, vocab_size=vocab_size, key_emb=key_emb,
            feature_ranges=ranges, lff_features=LFF_FEATURES,
            num_layers=n_layers, heads=HEADS, dropout=DROPOUT, use_lff=USE_LFF,
        )
        detector = ProfileConditionedAccumulator(
            embed_dim=encoder.out_dim, dropout=DROPOUT,
            use_film=USE_FILM, use_std=USE_STD, accumulator=ACCUMULATOR,
        )

        encoder.load_state_dict(enc_sd)
        detector.load_state_dict({k[len("detector."):]: v for k, v in sd.items()
                                  if k.startswith("detector.")})

        self.encoder  = encoder.to(self.device).eval()
        self.detector = detector.to(self.device).eval()
        self.bank: torch.Tensor | None = None  # (D,) mean-pooled, L2-normed profile
        # Optional genuine stream prefix (the enrolled user's events) used to seed
        # the live stream so scoring starts from keystroke #1 with a warm accumulator
        # (see set_stream_prefix / score). _prefix_z caches the prefix-only windows.
        self._prefix_feats: np.ndarray | None = None
        self._prefix_z: torch.Tensor | None = None

    # ── bank (enrollment) ────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_windows(self, windows: np.ndarray) -> torch.Tensor:
        """(N, seq_len, 3) feature windows → (N, D) L2-normed embeddings on device.

        Windows are already in the desktop feature space (asinh + z-score with
        DESKTOP_STATS, applied in keystrokes_to_features / holdflight_to_features).
        """
        batch = torch.from_numpy(windows.astype(np.float32)).to(self.device)
        return self.encoder(batch)

    @torch.no_grad()
    def set_bank_from_windows(self, raw_windows: np.ndarray) -> None:
        """raw_windows: (K, seq_len, 3) → encode K windows and store the active
        profile vector (see experiments.ca.model.build_profile).

        Clears any stream prefix — it belongs to a specific enrolled user/bank.
        """
        z = self._encode_windows(raw_windows)                          # (K, D)
        self.bank = build_profile(z, USE_STD)                          # (P,)
        self._prefix_feats = None
        self._prefix_z = None

    @torch.no_grad()
    def set_stream_prefix(self, features: np.ndarray) -> None:
        """Seed live scoring with the enrolled user's own events (a genuine prefix).

        The detector was trained on streams = genuine prefix + suffix with the
        accumulator running across the boundary, so feeding the enrollment session
        as the prefix lets live scoring start from keystroke #1 with a warm,
        genuine-baseline accumulator. Pre-encodes the prefix-only windows once.
        """
        features = np.asarray(features, dtype=np.float32)
        if len(features) < self.seq_len:
            self._prefix_feats = None
            self._prefix_z = None
            return
        self._prefix_feats = features
        n = len(features) - self.seq_len + 1
        windows = np.stack([features[i: i + self.seq_len] for i in range(n)])
        self._prefix_z = self._encode_windows(windows)                 # (P, D)

    @property
    def has_bank(self) -> bool:
        return self.bank is not None

    # ── scoring ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score(self, features: np.ndarray) -> np.ndarray:
        """Score a feature stream (n, 3). Returns p_t in [0, 1].

        With a stream prefix set (see set_stream_prefix), the enrolled user's events
        seed the stream: returns ONE score per live keystroke (length == len(features)),
        scored from keystroke #1 with the accumulator warmed over the prefix. Without
        a prefix, returns (n - seq_len + 1) scores and is empty below seq_len events.
        Empty if no bank is enrolled.
        """
        if self.bank is None:
            return np.zeros(0, dtype=np.float32)

        if self._prefix_z is not None and self._prefix_feats is not None:
            if len(features) == 0:
                return np.zeros(0, dtype=np.float32)
            full = np.concatenate([self._prefix_feats, features], axis=0)
            p_len = len(self._prefix_feats)
            # One window per live keystroke (ends at it; spans prefix tail + live).
            starts = [p_len + j - self.seq_len + 1 for j in range(len(features))]
            live_w = np.stack([full[s: s + self.seq_len] for s in starts])
            live_z = self._encode_windows(live_w)              # (n_live, D)
            z_seq = torch.cat([self._prefix_z, live_z], dim=0)
            logits = self.detector(z_seq, self.bank)
            return torch.sigmoid(logits)[len(self._prefix_z):].cpu().numpy()

        if len(features) < self.seq_len:
            return np.zeros(0, dtype=np.float32)
        n = len(features) - self.seq_len + 1
        windows = np.stack([features[i: i + self.seq_len] for i in range(n)])
        z_seq = self._encode_windows(windows)                  # (n, D)
        logits = self.detector(z_seq, self.bank)               # (n,)
        return torch.sigmoid(logits).cpu().numpy()
