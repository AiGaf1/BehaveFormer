"""CA baseline — cosine distance to a per-user enrollment template.

Stream-level baseline used as the Stage-1-only ablation row in the paper.
The detector / accumulator pipeline lives in `experiments.ca.model`;
bank construction lives in `ca_banks.py`.
"""

import numpy as np
import torch
import torch.nn.functional as F


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
    """One L2-normalised template embedding per user from their enrollment sessions.

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
    embs = [encoder(batch[s: s + batch_size]) for s in range(0, len(batch), batch_size)]
    z = torch.cat(embs, dim=0)
    return F.normalize(z.mean(dim=0), dim=-1).cpu()


@torch.no_grad()
def score_stream_baseline(
    stream: dict,
    encoder,
    templates: dict[int, torch.Tensor],
    seq_len: int,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Per-window cosine *distance* vs the user's enrollment template.

    Higher score = more likely impostor (matches the detector convention, so the
    same `Metric.evaluate_streams` works for both pipelines).
    """
    events = stream["events"].astype(np.float32)
    n = len(events)
    if n < seq_len:
        return torch.zeros(0)
    windows = np.stack([events[i: i + seq_len] for i in range(n - seq_len + 1)])
    template = templates[stream["user_idx"]].to(device).unsqueeze(0)
    scores = []
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[start: start + batch_size]).to(device)
        z = encoder(batch)
        scores.append((1.0 - F.cosine_similarity(z, template, dim=-1)).cpu())
    return torch.cat(scores)
