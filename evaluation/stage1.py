"""Stage 1 evaluation — window-level encoder verification.

Stage 1 trains the encoder f_theta with SupConLoss on individual windows.
Evaluation matches: extract window embeddings, then compute pairwise verification
metrics (EER, AUC, TAR@FAR) via `Metric.scores_from_embeddings`.

Nothing here knows about sessions, streams, banks, or templates — that's Stage 2.
"""

import torch
from torch.utils.data import DataLoader


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
