"""Live keystroke authentication inference.

Wraps the trained end-to-end model (Stage-1 encoder + ProfileConditionedAccumulator
detector) for real-time scoring of a keystroke stream against an enrolled user's bank.

Pipeline (mirrors training/eval exactly):
  raw keystrokes  →  features [hold_s, flight_s, key_id+1]  (data.AaltoDB.features)
  features         →  sliding windows of (seq_len, 3)
  windows          →  encoder  →  L2-normed embeddings z_t  (D,)
  z_seq + bank     →  detector →  per-window logit → sigmoid → p_t  (impostor prob)

The bank is K raw feature windows of the legitimate user, re-encoded on load so the
saved file stays valid across encoder checkpoints. Enrollment captures live typing,
slices it into windows, and saves them to `user_bank.npz`.
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
from experiments.stage1_encoder.model import Encoder  # noqa: E402
from experiments.stage2_detector.model import ProfileConditionedAccumulator  # noqa: E402

_DEFAULT_CKPT = _PROJECT_ROOT / "experiments" / "stage2_detector" / "best_models_e2e_L25_ln" / "epoch_9_ausc_0.9659.ckpt"
BANK_DIR  = Path(__file__).resolve().parent / "banks"
BANK_PATH = BANK_DIR / "user_bank.npz"

KEY_EMB      = 4
LFF_FEATURES = 16
NUM_LAYERS   = 2
HEADS        = 2


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

    def __init__(self, ckpt_path: Path = _DEFAULT_CKPT, seq_len: int = 25,
                 device: str | None = None):
        self.seq_len = seq_len
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Encoder construction needs vocab_size + feature_ranges; derive from the
        # dataset stats (the same the model was trained with).
        train_data, _, _ = load_aalto()
        vocab_size = aalto_vocab_size(train_data)
        ranges     = aalto_feature_ranges(train_data)

        encoder = Encoder(
            seq_len=seq_len, vocab_size=vocab_size, key_emb=KEY_EMB,
            feature_ranges=ranges, lff_features=LFF_FEATURES,
            num_layers=NUM_LAYERS, heads=HEADS, dropout=0.1,
        )
        detector = ProfileConditionedAccumulator(embed_dim=encoder.out_dim, tau=0.07)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        encoder.load_state_dict({k[len("encoder."):]: v for k, v in sd.items()
                                 if k.startswith("encoder.")})
        detector.load_state_dict({k[len("detector."):]: v for k, v in sd.items()
                                  if k.startswith("detector.")})

        self.encoder  = encoder.to(self.device).eval()
        self.detector = detector.to(self.device).eval()
        self.bank: torch.Tensor | None = None  # (K, D) L2-normed, on device

    # ── bank (enrollment) ────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_windows(self, windows: np.ndarray) -> torch.Tensor:
        """(N, seq_len, 3) raw features → (N, D) L2-normed embeddings on device."""
        batch = torch.from_numpy(windows.astype(np.float32)).to(self.device)
        return self.encoder(batch)

    @torch.no_grad()
    def set_bank_from_windows(self, raw_windows: np.ndarray) -> None:
        """raw_windows: (K, seq_len, 3) → encode and store as the active bank."""
        self.bank = self._encode_windows(raw_windows)

    def load_bank(self, path: Path = BANK_PATH) -> bool:
        """Load raw windows from npz, re-encode, set as active bank. False if missing."""
        if not path.exists():
            return False
        data = np.load(path)
        self.set_bank_from_windows(data["windows"])
        return True

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

    @torch.no_grad()
    def score_latest(self, features: np.ndarray) -> float | None:
        """Score only the most recent window (cheap, for live updates).

        Returns p_t for the last seq_len keystrokes, or None if not enough yet.
        """
        if self.bank is None or len(features) < self.seq_len:
            return None
        window = features[-self.seq_len:][None]                # (1, seq_len, 3)
        z = self._encode_windows(window)                       # (1, D)
        logit = self.detector(z, self.bank)                    # (1,)
        return float(torch.sigmoid(logit)[-1])


def _nonoverlap_starts(n: int, n_take: int, seq_len: int) -> list[int]:
    """Evenly-spaced window start positions — non-overlapping when n >= n_take*seq_len,
    maximally spread otherwise."""
    if n_take == 1:
        return [0]
    if n >= n_take * seq_len:
        stride = n // n_take          # stride >= seq_len → no overlap
    else:
        stride = max(1, (n - seq_len) // (n_take - 1))
    return [min(i * stride, n - seq_len) for i in range(n_take)]


def build_bank_windows(sessions: list[np.ndarray], seq_len: int, k: int) -> np.ndarray:
    """Build k enrollment windows distributed across sessions.

    Windows within each session are evenly spaced and non-overlapping when
    the session is long enough (n >= n_take * seq_len), maximally spread otherwise.
    sessions: list of (n_i, 3) feature arrays, one per enrollment session.
    Returns (k, seq_len, 3).
    """
    valid = [s for s in sessions if len(s) >= seq_len]
    if not valid:
        raise ValueError(f"Need at least {seq_len} keystrokes per session; got none")
    base, extra = divmod(k, len(valid))
    sample: list[np.ndarray] = []
    for i, sess in enumerate(valid):
        n_take = base + (1 if i < extra else 0)
        if n_take == 0:
            continue
        for s in _nonoverlap_starts(len(sess), n_take, seq_len):
            sample.append(sess[s: s + seq_len])
    return np.stack(sample).astype(np.float32)


def save_bank(raw_windows: np.ndarray, path: Path = BANK_PATH) -> None:
    """Persist raw bank windows (encoder-independent) next to the app."""
    np.savez(path, windows=raw_windows.astype(np.float32))


def score_distributions(
    engine: AuthEngine,
    bank_raw_windows: np.ndarray,
    aalto_users: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Score genuine and impostor distributions for a bank.

    Returns (genuine, impostor) score arrays. Restores bank_raw_windows on exit.
    Genuine: each Aalto user scored against their own bank (model baseline).
    Impostor: each Aalto user scored against the given bank.
    """
    seq_len = engine.seq_len
    engine.set_bank_from_windows(bank_raw_windows)
    impostor = [p for u in aalto_users
                for p in engine.score([s for s in u if len(s) >= seq_len][-1]).tolist()]
    genuine = []
    for u in aalto_users:
        sessions = [s for s in u if len(s) >= seq_len]
        engine.set_bank_from_windows(build_bank_windows(sessions[:5], seq_len, 5))
        genuine.extend(engine.score(sessions[-1]).tolist())
    engine.set_bank_from_windows(bank_raw_windows)
    return np.asarray(genuine), np.asarray(impostor)


def calibrate_threshold(engine: AuthEngine, bank_raw_windows: np.ndarray,
                        n_users: int = 20,
                        real_genuine_scores: np.ndarray | None = None) -> dict:
    """Calibrate threshold and estimate AUSC/PTCR/Usability for the given bank.

    Genuine distribution:
      - real_genuine_scores given (preferred): the actual user's verification stream
        scored against their bank (honest self-calibration).
      - real_genuine_scores None: Aalto users vs their own banks — a proxy that
        assumes the deployed user types like the dataset population.

    Impostor distribution always comes from Aalto users scored against this bank.

    Returns: {'tau', 'eer', 'frr', 'far', 'ausc', 'ptcr', 'usability',
              'genuine_p90', 'impostor_p10', 'n_genuine', 'n_impostor'}
    """
    import random as _random

    import torch
    from evaluation.metrics import Metric

    _, val, _ = load_aalto()
    users = [u for u in val if sum(1 for s in u if len(s) >= engine.seq_len) >= 6][:n_users]

    if real_genuine_scores is not None:
        # Impostor-only Aalto pass — skip the expensive self-vs-own-bank loop.
        seq_len = engine.seq_len
        engine.set_bank_from_windows(bank_raw_windows)
        i = np.asarray([p for u in users
                        for p in engine.score([s for s in u if len(s) >= seq_len][-1]).tolist()])
        g = np.asarray(real_genuine_scores, dtype=np.float32)
    else:
        g, i = score_distributions(engine, bank_raw_windows, users)
    # engine bank is restored to bank_raw_windows above
    thresholds = np.linspace(0.001, 0.999, 999)
    frr = np.array([(g  >  t).mean() for t in thresholds])
    far = np.array([(i <= t).mean() for t in thresholds])
    eer_i = int(np.argmin(np.abs(frr - far)))

    # Build simulated attack streams: Aalto prefix + different Aalto user's suffix,
    # scored against the enrolled user's bank. Mirrors stage_2.build_streams_for_split.
    rng = _random.Random(0)
    test_streams, single_streams = [], []
    for idx, u in enumerate(users):
        sessions = [s for s in u if len(s) >= engine.seq_len]
        prefix = sessions[0]
        t_star = rng.randint(engine.seq_len, max(engine.seq_len + 1, len(prefix)))

        # attack: prefix up to t_star + impostor suffix from a different user
        imp_u = users[(idx + 1) % len(users)]
        imp_sessions = [s for s in imp_u if len(s) >= engine.seq_len]
        imp_suffix = imp_sessions[-1]
        attack_events = np.concatenate([prefix[:t_star], imp_suffix], axis=0)
        test_streams.append({
            "events": attack_events, "t_star": t_star, "stream_type": "attack",
        })

        # same_user: prefix + another session of the same user (all genuine)
        same_suffix = sessions[1] if len(sessions) > 1 else sessions[0]
        same_events = np.concatenate([prefix[:t_star], same_suffix], axis=0)
        test_streams.append({
            "events": same_events, "t_star": t_star, "stream_type": "same_user",
        })

        # single: one full genuine session (for op_fpr baseline)
        single_streams.append({"events": sessions[-1], "t_star": 0, "stream_type": "single"})

    def score_fn(stream: dict) -> torch.Tensor:
        feats = stream["events"]
        scores = engine.score(feats)
        return torch.from_numpy(scores)

    stream_metrics = Metric.evaluate_streams(score_fn, test_streams, single_streams)

    return {
        "tau":          float(thresholds[eer_i]),
        "eer":          float((frr[eer_i] + far[eer_i]) / 2),
        "frr":          float(frr[eer_i]),
        "far":          float(far[eer_i]),
        "ausc":         stream_metrics["ausc"],
        "ptcr":         stream_metrics["ptcr"],
        "usability":    stream_metrics["usability"],
        "genuine_p90":  float(np.percentile(g, 90)),
        "impostor_p10": float(np.percentile(i, 10)),
        "n_genuine":    int(len(g)),
        "n_impostor":   int(len(i)),
    }
