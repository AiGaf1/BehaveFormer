import math

import torch
from torch import nn


class KeystrokeModel(nn.Module):
    """Keystroke-only transformer used by AaltoDB and HMOGDB.

    Input shape is (batch, sequence, features). The last feature must be the key
    code; every feature before it is treated as a timing value.
    """

    def __init__(
        self,
        seq_len,
        target_len,
        vocab_size,
        embed_dim,
        feature_ranges,
        lff_features=8,
        num_layers=6,
        heads=5,
        dropout=0.1,
    ):
        super().__init__()
        self.timing_encoder = LearnableFourierFeatures(feature_ranges, lff_features)
        self.key_embedding = nn.Embedding(vocab_size, embed_dim)
        d_model = self.timing_encoder.d_out + embed_dim

        self.pos_encoding = PositionalEncoding(20, d_model, seq_len)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dropout=dropout, batch_first=True),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.projection = build_projection_head(d_model * seq_len, (d_model * seq_len) // 2, target_len)

    def forward(self, x):
        timing_values = x[:, :, :-1]
        key_ids = x[:, :, -1].long()

        x = torch.cat(
            [self.timing_encoder(timing_values), self.key_embedding(key_ids)],
            dim=-1,
        )
        x = self.pos_encoding(x)
        return self.projection(torch.flatten(self.encoder(x), 1, 2))


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


class PositionalEncoding(nn.Module):
    """Learned Gaussian positional encoding from the original BehaveFormer model."""

    def __init__(self, k, d_model, seq_len):
        super().__init__()

        self.embedding = nn.Parameter(torch.zeros([k, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.positions: torch.Tensor
        self.register_buffer("positions", torch.arange(seq_len).unsqueeze(1).repeat(1, k))
        interval = seq_len / k
        self.mu = nn.Parameter((torch.arange(k, dtype=torch.float) * interval).unsqueeze(0))
        self.sigma = nn.Parameter(torch.full((1, k), 50.0))

    def forward(self, inputs):
        distance = self.positions - self.mu
        log_p = -distance * distance / (2 * self.sigma ** 2) - torch.log(self.sigma)
        pdfs = torch.nn.functional.softmax(log_p, dim=1)
        return inputs + torch.matmul(pdfs, self.embedding).unsqueeze(0)



def build_projection_head(input_dim, hidden_dim, output_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


# Legacy keystroke_imu_combined building blocks.

class CombinedTransformerEncoderLayer(nn.Module):
    """Dual-attention (feature + sequence axes) + CNN layer used in keystroke_imu_combined."""

    def __init__(self, d_model, heads, seq_heads, dropout, seq_len):
        super().__init__()
        self.attention  = nn.MultiheadAttention(d_model,  heads,     batch_first=True)
        self._attention = nn.MultiheadAttention(seq_len,  seq_heads, batch_first=True)
        self.attn_norm  = nn.LayerNorm(d_model)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 1, (1, 1)), nn.BatchNorm2d(1), nn.Dropout(dropout), nn.ReLU(),
            nn.Conv2d(1, 1, (3, 3), padding=1), nn.BatchNorm2d(1), nn.Dropout(dropout), nn.ReLU(),
            nn.Conv2d(1, 1, (5, 5), padding=2), nn.BatchNorm2d(1), nn.Dropout(dropout), nn.ReLU(),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, src):
        t = src.transpose(-1, -2)
        src = self.attn_norm(src + self.attention(src, src, src)[0] + self._attention(t, t, t)[0].transpose(-1, -2))
        return self.final_norm(src + self.cnn(src.unsqueeze(1)).squeeze(1))


class CombinedTransformerEncoder(nn.Module):
    def __init__(self, d_model, heads, seq_heads, seq_len, num_layers=5, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            CombinedTransformerEncoderLayer(d_model, heads, seq_heads, dropout, seq_len)
            for _ in range(num_layers)
        )

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src
