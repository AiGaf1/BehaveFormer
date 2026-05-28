"""Bank construction and threshold calibration.

`build_bank_windows` and `save_bank` are pure helpers; `calibrate_threshold` and
`score_distributions` run the engine against the Aalto val split to estimate
per-bank operating points (τ_op + z-norm μ/σ).
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import numpy as np
import torch

from app.engine import BANK_PATH, AuthEngine
from data.AaltoDB.prepare import load as load_aalto
from evaluation.ca_banks import _nonoverlap_starts
from evaluation.metrics import Metric


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
    if len(sample) < k:
        warnings.warn(
            f"build_bank_windows: requested {k} windows but only {len(sample)} "
            f"could be built from the provided sessions. More enrollment data will "
            f"improve authentication accuracy.",
            stacklevel=2,
        )
    return np.stack(sample).astype(np.float32)


def save_bank(raw_windows: np.ndarray, path: Path = BANK_PATH,
              calibration: dict | None = None) -> None:
    """Persist raw bank windows (encoder-independent) next to the app.

    If `calibration` is given (output of calibrate_threshold), each value is stored
    under a `cal_<key>` array so Start auth can skip the live recalibration pass.
    """
    data: dict = {"windows": raw_windows.astype(np.float32)}
    if calibration is not None:
        for k, v in calibration.items():
            data[f"cal_{k}"] = np.asarray(v)
    np.savez(path, **data)


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
                        real_genuine_events: np.ndarray | None = None) -> dict:
    """Calibrate threshold and estimate AUSC/PTCR/Usability for the given bank.

    τ is selected as `tau_op` from `Metric.evaluate_streams` — the same objective
    (Usability ≈ PTCR on per-stream metrics) used by training/test. This matches
    paper metrics and is far more usability-friendly than the symmetric-error
    point (EER), which was the previous choice.

    Genuine distribution for the per-window EER/p90 stats:
      - real_genuine_events given (preferred): user's own verification stream.
        Also injected as an additional `same_user` stream so `tau_op` is
        calibrated against the user's actual typing distribution, not Aalto's.
      - real_genuine_events None: falls back to Aalto users vs their own banks.

    Impostor distribution always comes from Aalto users scored against this bank.

    Also computes z-norm calibration (μ, σ of genuine p_t) — the decision
    threshold becomes a z-score (e.g. 3.0 = "3σ above this user's baseline"),
    which transfers across users where raw τ doesn't.

    Returns: {'tau', 'eer', 'frr', 'far', 'ausc', 'ptcr', 'usability',
              'genuine_p90', 'impostor_p10', 'n_genuine', 'n_impostor',
              'znorm_mu', 'znorm_sigma', 'z_threshold'}
    """
    _, val, _ = load_aalto()
    users = [u for u in val if sum(1 for s in u if len(s) >= engine.seq_len) >= 6][:n_users]

    engine.set_bank_from_windows(bank_raw_windows)

    if real_genuine_events is not None:
        seq_len = engine.seq_len
        i = np.asarray([p for u in users
                        for p in engine.score([s for s in u if len(s) >= seq_len][-1]).tolist()])
        g = engine.score(real_genuine_events)
    else:
        g, i = score_distributions(engine, bank_raw_windows, users)
    # engine bank is restored to bank_raw_windows above
    thresholds = np.linspace(0.001, 0.999, 999)
    frr = np.array([(g  >  t).mean() for t in thresholds])
    far = np.array([(i <= t).mean() for t in thresholds])
    eer_i = int(np.argmin(np.abs(frr - far)))

    # Build attack streams from Aalto users (their genuine prefix + a different
    # user's suffix), scored against the enrolled user's bank. The user's own
    # verification typing is the ONLY same_user stream — Usability then measures
    # how quiet THIS user's typing is against THEIR bank, not how a random Aalto
    # user happens to look against it.
    rng = random.Random(0)
    test_streams, single_streams = [], []
    for idx, u in enumerate(users):
        sessions = [s for s in u if len(s) >= engine.seq_len]
        prefix = sessions[0]
        t_star = rng.randint(engine.seq_len, max(engine.seq_len + 1, len(prefix)))

        imp_u = users[(idx + 1) % len(users)]
        imp_sessions = [s for s in imp_u if len(s) >= engine.seq_len]
        imp_suffix = imp_sessions[-1]
        attack_events = np.concatenate([prefix[:t_star], imp_suffix], axis=0)
        test_streams.append({
            "events": attack_events, "t_star": t_star, "stream_type": "attack",
        })

        single_streams.append({"events": sessions[-1], "t_star": 0, "stream_type": "single"})

    if real_genuine_events is not None and len(real_genuine_events) >= engine.seq_len:
        evs = np.asarray(real_genuine_events, dtype=np.float32)
        test_streams.append({
            "events": evs, "t_star": len(evs), "stream_type": "same_user",
        })

    def score_fn(stream: dict) -> torch.Tensor:
        return torch.from_numpy(engine.score(stream["events"]))

    stream_metrics = Metric.evaluate_streams(score_fn, test_streams, single_streams)

    mu, sigma = float(np.mean(g)), float(np.std(g))
    engine.znorm_mu = mu
    engine.znorm_sigma = sigma

    return {
        "tau":          float(stream_metrics["tau_op"]),
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
        "znorm_mu":     mu,
        "znorm_sigma":  sigma,
        "z_threshold":  3.0,  # default: alert at 3σ above baseline
    }
