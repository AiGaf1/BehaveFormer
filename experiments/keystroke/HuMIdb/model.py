import torch
from torch import nn

from experiments.keystroke.common.modeling import (
    PositionalEncoding,
    TransformerEncoder,
    TransformerEncoderLayer,
    build_projection_head,
)


class Transformer(nn.Module):
    def __init__(self, num_layer, d_model, k, heads, _heads, seq_len, trg_len, dropout):
        super().__init__()
        del trg_len
        self.pos_encoding = PositionalEncoding(k, d_model, seq_len)
        self.encoder = TransformerEncoder(d_model, heads, _heads, seq_len, num_layer, dropout)

    def forward(self, inputs):
        encoded_inputs = self.pos_encoding(inputs)
        return self.encoder(encoded_inputs)


class Model(nn.Module):
    def __init__(self, feature_count, l, trg_len):
        super().__init__()
        self.keystroke_transformer = Transformer(5, feature_count, 20, 3, 10, l, trg_len, 0.1)
        self.linear = build_projection_head(feature_count * l, (feature_count * l) // 2, trg_len)

    def forward(self, inputs):
        return self.linear(torch.flatten(self.keystroke_transformer(inputs), start_dim=1, end_dim=2))
