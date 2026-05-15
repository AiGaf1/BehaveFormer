"""Stage 2 losses."""

import torch
import torch.nn.functional as F


def time_aware_bce(
    preds:   torch.Tensor,   # (B,)
    labels:  torch.Tensor,   # (B,)
    t_star:  torch.Tensor,   # (B,)
    w_start: torch.Tensor,   # (B,)
    alpha:   float = 0.1,
) -> torch.Tensor:
    """BCE with an *additive* earliness bonus on post-attack windows:

        w_pos(delay) = 1 + exp(-alpha * delay)   ∈ [1, 2]
        w_neg        = 1

    Positives are always weighted at least as much as negatives (no class
    collapse), with up to a 2× bonus at delay=0 to reward early detection.
    Reduction is a weight-normalised mean so gradient norms stay bounded.
    """
    bce     = F.binary_cross_entropy(preds, labels, reduction="none")
    delay   = (w_start - t_star).clamp(min=0).float()
    weights = torch.where(
        labels > 0.5,
        1.0 + torch.exp(-alpha * delay),
        torch.ones_like(delay),
    )
    return (bce * weights).sum() / weights.sum().clamp(min=1e-8)
