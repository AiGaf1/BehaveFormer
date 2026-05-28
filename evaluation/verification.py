"""Window-level verification (non-CA).

Pairwise EER/AUC/TAR@FAR on encoder window embeddings, plus the BehaveFormer
Aalto-DB protocol (IJCB 2023): per-user enrollment set vs verification probes.
Nothing here knows about streams, banks, or templates — that's `ca_baseline.py`
and `ca_banks.py`.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.metrics import Metric


@torch.no_grad()
def encode_windows(
    encoder,
    dataset,
    device: torch.device,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Encode every window in a dataset. Returns (N, D) on CPU.

    Fast path: if `dataset.windows` is a pre-materialised tensor (e.g.
    `PureWindowDataset`), encode it directly without a DataLoader.
    """
    if hasattr(dataset, "windows"):
        embs = []
        for start in range(0, len(dataset.windows), batch_size):
            batch = dataset.windows[start: start + batch_size].to(device)
            embs.append(encoder(batch).cpu())
        return torch.cat(embs, dim=0)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embs = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        embs.append(encoder(x.to(device)).cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def behaveformer_eer(
    encoder,
    test_data: list,
    seq_len: int,
    device: torch.device,
    n_enroll: int = 5,
    max_users: int | None = None,
    seed: int = 0,
) -> dict:
    """BehaveFormer Aalto-DB verification protocol (IJCB 2023, Table 2).

    For each test user u, the first `n_enroll` sessions are enrollment samples
    and the remaining sessions are verification probes. One sample per session
    (random window of length `seq_len`). The score between a probe `f_{v,a}`
    of user v and the enrollment set of user u is

        s_{u,v}^{a} = (1/E) * sum_e || f_{u,e} - f_{v,a} ||_2

    Genuine pairs: u == v. Impostor pairs: u != v over all probes × all users.
    Returns EER (%), threshold (Euclidean distance), AUC, TAR@FAR table.
    """
    rng = np.random.default_rng(seed)
    encoder.eval()
    enroll_list: list[torch.Tensor] = []
    probe_embs:  list[torch.Tensor] = []
    probe_uids:  list[int]          = []
    kept_uids:   list[int]          = []

    users = list(enumerate(test_data))
    if max_users is not None and len(users) > max_users:
        users = users[:max_users]

    for uid, sessions in users:
        valid = [s for s in sessions if len(s) >= seq_len]
        if len(valid) < n_enroll + 1:
            continue
        enroll_w, probe_w = [], []
        for sess in valid[:n_enroll]:
            arr = sess.astype(np.float32)
            start = int(rng.integers(0, len(arr) - seq_len + 1))
            enroll_w.append(arr[start: start + seq_len])
        for sess in valid[n_enroll:]:
            arr = sess.astype(np.float32)
            start = int(rng.integers(0, len(arr) - seq_len + 1))
            probe_w.append(arr[start: start + seq_len])

        e_batch = torch.from_numpy(np.stack(enroll_w)).to(device)
        p_batch = torch.from_numpy(np.stack(probe_w)).to(device)
        e_emb = encoder(e_batch).cpu()
        p_emb = encoder(p_batch).cpu()
        enroll_list.append(e_emb)
        for pe in p_emb:
            probe_embs.append(pe)
            probe_uids.append(uid)
        kept_uids.append(uid)

    if not kept_uids:
        raise RuntimeError("behaveformer_eer: no users had ≥1 probe after enrollment cut")

    enroll = torch.stack(enroll_list)
    probes = torch.stack(probe_embs)
    probe_uid_t = torch.tensor(probe_uids, dtype=torch.long)
    N_users, E, D = enroll.shape

    enroll_flat = enroll.reshape(N_users * E, D)
    # Chunk along probes so cdist stays under ~200 MB
    chunk = max(1, 200_000_000 // (N_users * E * 4))
    scores_rows: list[torch.Tensor] = []
    for s in range(0, len(probes), chunk):
        d = torch.cdist(probes[s: s + chunk], enroll_flat)
        d = d.reshape(-1, N_users, E).mean(dim=-1)
        scores_rows.append(d)
    scores = torch.cat(scores_rows, dim=0)

    col_of_uid = {uid: i for i, uid in enumerate(kept_uids)}
    probe_cols = torch.tensor([col_of_uid[int(u)] for u in probe_uid_t], dtype=torch.long)
    row_idx = torch.arange(len(probes))
    genuine = scores[row_idx, probe_cols]

    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[row_idx, probe_cols] = False
    impostor = scores[mask]

    # Lower distance ⇒ more genuine; Metric.eer_compute treats higher = genuine
    g_sim, i_sim = -genuine, -impostor
    eer, thr_sim = Metric.eer_compute(g_sim, i_sim)
    auc = Metric.roc_auc(g_sim, i_sim)
    tar_far = Metric.tar_at_far(g_sim, i_sim)
    return {
        "eer": eer,
        "threshold_dist": float(-thr_sim),
        "auc": auc,
        "tar_at_far": tar_far,
        "n_users": N_users,
        "n_genuine": int(len(genuine)),
        "n_impostor": int(len(impostor)),
        "n_enroll": n_enroll,
    }
