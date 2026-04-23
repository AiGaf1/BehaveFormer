import torch
from torch import nn

from experiments.common.modeling import (
    PositionalEncoding,
    TransformerEncoder,
    build_projection_head,
)


class KeystrokeTransformer(nn.Module):
    def __init__(self, num_layer, d_model, k, heads, _heads, seq_len, trg_len, inner_dropout):
        super().__init__()
        self.pos_encoding = PositionalEncoding(k, d_model, seq_len)
        self.encoder = TransformerEncoder(
            d_model,
            heads,
            _heads,
            seq_len,
            num_layer,
            inner_dropout,
            cnn_units=seq_len,
        )
        self.ff = build_projection_head(d_model * seq_len, (d_model * seq_len) // 2, trg_len)

    def forward(self, inputs):
        encoded_inputs = self.pos_encoding(inputs)
        enc_out = self.encoder(encoded_inputs)
        return self.ff(torch.flatten(enc_out, start_dim=1, end_dim=2))
