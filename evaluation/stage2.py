"""Stage 2 evaluation — stream-level continuous authentication.

Stage 2 trains a bank-aware detector D_phi. Evaluation produces a per-window
attack probability p_t for each stream, which `Metric.evaluate_streams`
consumes to compute AUSC, PTCR, EDD, Usability, Op.FPR.

Also provides a Stage-1-only baseline (cosine distance to a per-user enrollment
template, no detector, no bank) for the paper's ablation comparison.
"""

import numpy as np
import torch
import torch.nn.functional as F

from data.AaltoDB.stage_2 import ENROLL_END


# ── Stage 2 — bank-aware detector ─────────────────────────────────────────────

@torch.no_grad()
def build_banks(
    encoder,
    aalto_split: list,
    seq_len: int,
    device: torch.device,
    k: int,
    seed: int = 42,
) -> dict[int, torch.Tensor]:
    """For each user, sample k DISTINCT windows from enrollment sessions and
    return their L2-normalised embeddings as (k, D) cpu tensors keyed by user_idx.

    Users with fewer than k enrollment windows are SKIPPED (would otherwise
    require duplicate bank entries, which biases ψ_t stats and the attention
    readout). Callers filter streams against the returned dict.
    """
    rng = np.random.default_rng(seed)
    encoder.eval()
    banks: dict[int, torch.Tensor] = {}
    skipped_no_windows = skipped_short = 0

    for user_idx, user_sessions in enumerate(aalto_split):
        windows = []
        for sess in user_sessions[:ENROLL_END]:
            arr = sess.astype(np.float32)
            if len(arr) < seq_len:
                continue
            for i in range(len(arr) - seq_len + 1):
                windows.append(arr[i: i + seq_len])

        if not windows:
            skipped_no_windows += 1
            continue
        if len(windows) < k:
            skipped_short += 1
            continue
        chosen_idx = rng.choice(len(windows), size=k, replace=False)
        batch = torch.from_numpy(np.stack([windows[i] for i in chosen_idx])).to(device)
        banks[user_idx] = encoder(batch).cpu()

    if skipped_no_windows or skipped_short:
        from utils.logger import get_logger
        get_logger(__name__).info(
            "build_banks: kept %d / %d users  (skipped no-windows=%d, <%d-windows=%d)",
            len(banks), len(aalto_split), skipped_no_windows, k, skipped_short,
        )
    return banks


@torch.no_grad()
def score_stream(
    stream: dict,
    encoder,
    detector,
    banks: dict[int, torch.Tensor],
    seq_len: int,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Score one stream window-by-window with the bank-aware detector.

    Returns p_t (n_windows,) in [0, 1] — higher = more likely impostor.
    """
    events = stream["events"].astype(np.float32)
    n = len(events)
    if n < seq_len:
        return torch.zeros(0)
    windows = np.stack([events[i: i + seq_len] for i in range(n - seq_len + 1)])
    bank = banks[stream["user_idx"]].to(device)
    scores = []
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[start: start + batch_size]).to(device)
        z = encoder(batch)
        p = detector(z, bank.unsqueeze(0).expand(len(batch), -1, -1))
        scores.append(p.cpu())
    return torch.cat(scores)


# ── Stage 1 baseline — cosine to enrollment template, no detector ────────────

@torch.no_grad()
def build_user_template(
    encoder,
    enrollment_sessions: list,
    seq_len: int,
    device: torch.device,
    batch_size: int = 2048,
    max_windows: int | None = 64,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Build one L2-normalised template embedding per user from their enrollment sessions.

    If `max_windows` is set, randomly samples that many windows across the
    enrollment sessions (much faster than embedding every sliding window).
    """
    rng = rng or np.random.default_rng(0)
    pool: list[np.ndarray] = []
    for sess in enrollment_sessions:
        arr = sess.astype(np.float32)
        max_start = len(arr) - seq_len
        if max_start < 0:
            continue
        if max_windows is None:
            for i in range(max_start + 1):
                pool.append(arr[i: i + seq_len])
        else:
            n = min(max_windows, max_start + 1)
            for s in rng.choice(max_start + 1, size=n, replace=False):
                pool.append(arr[s: s + seq_len])

    if not pool:
        raise ValueError("No valid windows in enrollment sessions")

    if max_windows is not None and len(pool) > max_windows:
        idx = rng.choice(len(pool), size=max_windows, replace=False)
        pool = [pool[i] for i in idx]

    batch = torch.from_numpy(np.stack(pool)).to(device)
    embs = []
    for start in range(0, len(batch), batch_size):
        embs.append(encoder(batch[start: start + batch_size]))
    z = torch.cat(embs, dim=0)
    return F.normalize(z.mean(dim=0), dim=-1).cpu()


def score_stream_baseline(
    stream: dict,
    encoder,
    templates: dict[int, torch.Tensor],
    seq_len: int,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Stage 1 baseline: cosine *distance* per window vs the user's enrollment template.

    Higher score = more likely impostor (same convention as score_stream).
    Used by `Metric.evaluate_streams` to compare against the Stage 2 detector.
    """
    events = stream["events"].astype(np.float32)
    n = len(events)
    if n < seq_len:
        return torch.zeros(0)
    windows = np.stack([events[i: i + seq_len] for i in range(n - seq_len + 1)])
    template = templates[stream["user_idx"]].to(device).unsqueeze(0)

    scores = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start: start + batch_size]).to(device)
            z = encoder(batch)
            scores.append((1.0 - F.cosine_similarity(z, template, dim=-1)).cpu())
    return torch.cat(scores)
