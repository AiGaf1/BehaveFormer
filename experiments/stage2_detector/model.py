"""
Stage 2: bank-aware decision module  D_phi.

Given the current window embedding z_t (from the frozen encoder) and the
user embedding bank B_u = {z_1^u, ..., z_K^u}, it outputs an impostor
probability p_t in [0, 1].

Bank interaction produces four feature streams (paper §3c):
  z_t          – current embedding                        (B, D)
  ψ_t          – cosine similarity stats vs bank          (B, 4)
                 [max, top-3 mean, mean, std]
  v_t          – attention readout from bank              (B, D)
  z_t - v_t    – residual (mismatch with user profile)    (B, D)

Concatenated input to MLP head: (B, 3*D + 4)

Forward signature:
  z_t  : (B, D)       – L2-normalised window embeddings from the frozen encoder
  bank : (B, K, D)    – per-sample user banks (K enrollment embeddings, L2-normalised)

The bank tensor is (B, K, D) so different users can be in the same batch during
training.  At inference time, broadcast a single user's bank with .expand().
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class BankAwareDetector(nn.Module):
    """
    Args:
        embed_dim:  embedding dimension D (must match the frozen encoder)
        top_k:      how many top cosine similarities to average in ψ_t
        hidden_dim: hidden size of the MLP head
        dropout:    dropout probability in the MLP
    """

    def __init__(
        self,
        embed_dim: int,
        top_k: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.top_k = top_k
        self._scale = math.sqrt(embed_dim)

        # 4 similarity stats + z_t + v_t + (z_t - v_t)
        in_dim = 4 + 3 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        bank: torch.Tensor,
        ablation: str | None = None,
    ) -> torch.Tensor:
        """
        z_t     : (B, D)    L2-normalised
        bank    : (B, K, D) L2-normalised
        ablation: one of None | 'no_bank' | 'no_psi' | 'no_attn' | 'psi_only'
          'no_bank'  – skip all bank interaction; feed [z_t, 0, 0, 0]
          'no_psi'   – zero out ψ_t stats
          'no_attn'  – zero out v_t and z_t-v_t
          'psi_only' – keep ψ_t; zero out z_t, v_t, and z_t-v_t. Use to test
                       whether the MLP relies on the high-dim embeddings or
                       the 4 calibrated similarity stats.
        returns : (B,) impostor probabilities in [0, 1]
        """
        B, D = z_t.shape

        if ablation == "no_bank":
            psi_t = torch.zeros(B, 4, device=z_t.device)
            v_t   = torch.zeros(B, D, device=z_t.device)
            feat  = torch.cat([z_t, psi_t, v_t, z_t - v_t], dim=1)
            return torch.sigmoid(self.mlp(feat).squeeze(-1))

        # ── cosine similarities ───────────────────────────────────────────────
        sims = F.cosine_similarity(z_t.unsqueeze(1), bank, dim=-1)    # (B, K)

        # ── similarity stats ψ_t ─────────────────────────────────────────────
        k        = min(self.top_k, sims.shape[1])
        sim_max  = sims.max(dim=1).values
        sim_topk = sims.topk(k, dim=1).values.mean(dim=1)
        sim_mean = sims.mean(dim=1)
        sim_std  = sims.std(dim=1, unbiased=False)   # biased std: stable at K=5
        psi_t    = torch.stack([sim_max, sim_topk, sim_mean, sim_std], dim=1)  # (B, 4)

        if ablation == "no_psi":
            psi_t = torch.zeros_like(psi_t)

        # ── attention readout v_t ─────────────────────────────────────────────
        attn = torch.bmm(z_t.unsqueeze(1), bank.transpose(1, 2)) / self._scale  # (B,1,K)
        attn = torch.softmax(attn, dim=-1)
        v_t  = torch.bmm(attn, bank).squeeze(1)                       # (B, D)

        if ablation == "no_attn":
            v_t = torch.zeros_like(v_t)

        if ablation == "psi_only":
            z_t = torch.zeros_like(z_t)
            v_t = torch.zeros_like(v_t)

        # ── concatenate and classify ──────────────────────────────────────────
        feat = torch.cat([z_t, psi_t, v_t, z_t - v_t], dim=1)        # (B, 3D+4)
        return torch.sigmoid(self.mlp(feat).squeeze(-1))               # (B,)
