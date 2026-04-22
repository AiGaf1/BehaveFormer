import torch
from torch import nn


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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, _heads, dropout, seq_len, cnn_units=1):
        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self._attention = nn.MultiheadAttention(seq_len, _heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_units, (1, 1)),
            nn.BatchNorm2d(cnn_units),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(cnn_units, cnn_units, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_units),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv2d(cnn_units, 1, (5, 5), padding=2),
            nn.BatchNorm2d(1),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        del src_mask
        feature_attention = self.attention(src, src, src)[0]
        sequence_attention = self._attention(
            src.transpose(-1, -2),
            src.transpose(-1, -2),
            src.transpose(-1, -2),
        )[0].transpose(-1, -2)
        src = self.attn_norm(src + feature_attention + sequence_attention)
        src = self.final_norm(src + self.cnn(src.unsqueeze(dim=1)).squeeze(dim=1))
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, heads, _heads, seq_len, num_layer=2, dropout=0.1, cnn_units=1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, heads, _heads, dropout, seq_len, cnn_units=cnn_units)
                for _ in range(num_layer)
            ]
        )

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


def build_projection_head(input_dim, hidden_dim, output_dim, dropout=0.1):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )
