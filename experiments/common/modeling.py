import torch
from torch import nn


class KeystrokeModel(nn.Module):
    """Single-stream keystroke transformer with key embedding. Used by AaltoDB and HMOGDB."""

    def __init__(self, seq_len, target_len, vocab_size, embed_dim, num_layers=6, heads=5, dropout=0.1):
        super().__init__()
        timing_features = 9
        d_model = timing_features + embed_dim
        self.key_embedding = KeyEmbedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(20, d_model, seq_len)
        self.encoder = TransformerEncoder(d_model, heads, num_layers, dropout)
        self.ff = build_projection_head(d_model * seq_len, (d_model * seq_len) // 2, target_len)

    def forward(self, x):
        x = self.pos_encoding(self.key_embedding(x))
        return self.ff(torch.flatten(self.encoder(x), 1, 2))


class KeyEmbedding(nn.Module):
    """Replaces the raw integer key-code column with a learned embedding.

    Expects input of shape (batch, seq_len, feature_count) where the last
    column contains integer key codes. Returns a tensor of shape
    (batch, seq_len, feature_count - 1 + embed_dim).
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        timing = x[:, :, :-1]
        keys = x[:, :, -1].long()
        return torch.cat([timing, self.embedding(keys)], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, k, d_model, seq_len):
        super().__init__()

        self.embedding = nn.Parameter(torch.zeros([k, d_model], dtype=torch.float), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        self.register_buffer("positions", torch.arange(seq_len).unsqueeze(1).repeat(1, k))
        interval = seq_len / k
        self.mu = nn.Parameter((torch.arange(k, dtype=torch.float) * interval).unsqueeze(0))
        self.sigma = nn.Parameter(torch.full((1, k), 50.0))

    def normal_pdf(self, pos, mu, sigma):
        distance = pos - mu
        log_p = -1 * torch.mul(distance, distance) / (2 * (sigma**2)) - torch.log(sigma)
        return torch.nn.functional.softmax(log_p, dim=1)

    def forward(self, inputs):
        pdfs = self.normal_pdf(self.positions, self.mu, self.sigma)
        pos_enc = torch.matmul(pdfs, self.embedding)
        return inputs + pos_enc.unsqueeze(0)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, heads=2, num_layer=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layer,
            enable_nested_tensor=False,
        )

    def forward(self, src):
        return self.encoder(src)


def build_projection_head(input_dim, hidden_dim, output_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


# ── keystroke_imu_combined building blocks ───────────────────────────────────

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
