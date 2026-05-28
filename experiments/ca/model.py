"""Stage 2 detector — `ProfileConditionedAccumulator`.

Personal VAD 2.0–style decision head: a single mean-pooled, L2-normed speaker
(bank) embedding conditions the per-window MLP via FiLM (Feature-wise Linear
Modulation). A learnable leaky-CUSUM recurrence integrates per-window evidence
over the stream.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ProfileConditionedAccumulator(nn.Module):
    """PVAD-2.0 decision head + leaky-CUSUM accumulator.

    Per window t (one stream):
      h_t    = ReLU( FiLM( Linear(z_t),  γ, β ) )            (hidden,)
      γ, β   = chunked Linear(profile)                       (2·hidden,)
      s_t    = w · h_t + b  -  drift                         (1,)
    Leaky CUSUM:
      S_t    = sigmoid(g) · S_{t-1} + s_t                    causal accumulator
      logit_t = a · S_t + c                                  learned scale + bias

    Forward signature:
      z_seq   : (T, D)   L2-normalised window embeddings of ONE stream
      profile : (2D,)    [mean, std] of the K enrollment-bank embeddings
    Returns:
      logits  : (T,)    apply sigmoid externally for p_t in [0, 1]

    FiLM head is initialised so γ=1, β=0 (identity at start): an untrained
    model behaves as a plain MLP, and training learns the modulation deviations.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden: int = 64,
        dropout: float = 0.1,
        use_film: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        self.use_film = use_film
        # Profile is [mean, std] per-dim over the K bank windows → 2*embed_dim.
        if use_film:
            # FiLM: profile predicts (γ, β) that modulate the hidden projection of z_t.
            self.proj = nn.Linear(embed_dim, hidden)
            self.film = nn.Linear(2 * embed_dim, 2 * hidden)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)          # identity at init
        else:
            # Concat: [z_t, profile] → hidden (pure PVAD-2.0 concatenation, no FiLM).
            self.proj = nn.Linear(embed_dim + 2 * embed_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.score   = nn.Linear(hidden, 1)
        # Leaky-CUSUM params.
        self.drift  = nn.Parameter(torch.zeros(1))
        self.forget = nn.Parameter(torch.zeros(1))
        self.scale  = nn.Parameter(torch.ones(1))
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, z_seq: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        T = z_seq.shape[0]
        device, dtype = z_seq.device, z_seq.dtype

        if self.use_film:
            h = self.proj(z_seq)                            # (T, hidden)
            gb = self.film(profile)                         # (2*hidden,)
            gamma = 1.0 + gb[:self.hidden]                  # (hidden,)
            beta  = gb[self.hidden:]                        # (hidden,)
            h = self.dropout(F.relu(gamma * h + beta))      # (T, hidden)
        else:
            feat = torch.cat([z_seq, profile.unsqueeze(0).expand(T, -1)], dim=-1)
            h = self.dropout(F.relu(self.proj(feat)))       # (T, hidden)

        s = self.score(h).squeeze(-1) - self.drift          # (T,) centred evidence
        gate = torch.sigmoid(self.forget)                   # leak factor in (0, 1)

        # Causal leaky-CUSUM recurrence: S_t = gate·S_{t-1} + s_t
        S = torch.empty(T, device=device, dtype=dtype)
        acc = torch.zeros((), device=device, dtype=dtype)
        for t in range(T):
            acc = gate * acc + s[t]
            S[t] = acc
        return self.scale * S + self.bias                   # (T,) logits
