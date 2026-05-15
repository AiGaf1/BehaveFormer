"""
Stage 1 encoder  f_theta: window -> embedding z in R^d.

Architecture:
  KeystrokeModel (transformer)  →  mean-pool over sequence  →  L2-normalised projection head

Mean-pooling collapses (B, T, d_model) to (B, d_model).
The projection head maps that to the embedding dimension `embed_dim` and L2-normalises.

During stage 2 the encoder is frozen; only the decision module D_phi is trained.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class KeystrokeModel(nn.Module):
    """Keystroke transformer encoder (AaltoDB).

    Input shape is (batch, sequence, features). The last feature must be the key
    code; every feature before it is treated as a timing value.
    """

    def __init__(
        self,
        seq_len,
        vocab_size,
        key_emb,
        feature_ranges,
        lff_features=16,
        num_layers=1,
        heads=1,
        dropout=0.1,
    ):
        super().__init__()
        self.timing_encoder = LearnableFourierFeatures(feature_ranges, lff_features)
        self.key_embedding = nn.Embedding(vocab_size, key_emb)
        d_model = self.timing_encoder.d_out + key_emb

        self.pos_encoding: torch.Tensor
        self.register_buffer("pos_encoding", _sinusoidal_encoding(seq_len, d_model))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x):
        timing_values = x[:, :, :-1]
        key_ids = x[:, :, -1].long()

        x = torch.cat(
            [self.timing_encoder(timing_values), self.key_embedding(key_ids)],
            dim=-1,
        )
        x = x + self.pos_encoding[:x.size(1)]
        return self.encoder(x)  # (B, T, d_model)


class LearnableFourierFeatures(nn.Module):
    """Encode raw timing values with learned sin/cos frequency scales."""

    def __init__(self, feature_dict: dict, num_features: int):
        super().__init__()
        periods = [
            torch.logspace(math.log10(bounds["min"]), math.log10(bounds["max"]), steps=num_features)
            for bounds in feature_dict.values()
        ]
        freq = 2 * torch.pi / torch.stack(periods)
        self.freq: torch.Tensor
        self.register_buffer("freq", freq)
        self.scales_raw = nn.Parameter(torch.randn_like(freq) * 0.1)
        self.d_out = 2 * len(feature_dict) * num_features

    def forward(self, x):
        proj = x.unsqueeze(-1) * self.freq * torch.sigmoid(self.scales_raw)
        fourier = torch.stack([proj.sin(), proj.cos()], dim=-1)
        return fourier.flatten(start_dim=-3)


def _sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    positions = torch.arange(seq_len).unsqueeze(1)
    dims = torch.arange(0, d_model, 2)
    div = torch.exp(dims * (-math.log(1000.0) / d_model))
    enc = torch.zeros(seq_len, d_model)
    enc[:, 0::2] = torch.sin(positions * div)
    enc[:, 1::2] = torch.cos(positions * div[:d_model // 2])
    return enc


class Encoder(nn.Module):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        key_emb: int,
        feature_ranges: dict,
        lff_features: int = 16,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = KeystrokeModel(
            seq_len=seq_len,
            vocab_size=vocab_size,
            key_emb=key_emb,
            feature_ranges=feature_ranges,
            lff_features=lff_features,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
        self.out_dim = self.backbone.timing_encoder.d_out + key_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(x)       # (B, T, d_model)
        pooled = tokens.mean(dim=1)     # (B, d_model)
        return F.normalize(pooled, dim=1)
