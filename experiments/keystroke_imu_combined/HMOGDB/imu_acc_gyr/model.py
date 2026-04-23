import torch
from torch import nn

from experiments.common.modeling import CombinedTransformerEncoder, PositionalEncoding, build_projection_head


class _Stream(nn.Module):
    def __init__(self, feature_count, seq_len, heads, seq_heads):
        super().__init__()
        self.pos_encoding = PositionalEncoding(20, feature_count, seq_len)
        self.encoder = CombinedTransformerEncoder(feature_count, heads, seq_heads, seq_len)

    def forward(self, x):
        return torch.flatten(self.encoder(self.pos_encoding(x)), 1, 2)


class Model(nn.Module):
    def __init__(self, ks_features, imu_features, ks_len, imu_len, trg_len):
        super().__init__()
        self.ks_stream  = _Stream(ks_features,  ks_len,  heads=5, seq_heads=10)
        self.imu_stream = _Stream(imu_features, imu_len, heads=6, seq_heads=10)
        self.ks_head    = build_projection_head(ks_features  * ks_len,  (ks_features  * ks_len)  // 2, trg_len)
        self.imu_head   = build_projection_head(imu_features * imu_len, (imu_features * imu_len) // 2, trg_len)
        self.final      = nn.Linear(trg_len * 2, trg_len)

    def forward(self, inputs):
        ks, imu = inputs
        return self.final(torch.cat([self.ks_head(self.ks_stream(ks)), self.imu_head(self.imu_stream(imu))], dim=-1))
