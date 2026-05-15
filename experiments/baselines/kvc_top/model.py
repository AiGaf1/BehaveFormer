import torch
from torch import nn

from experiments.stage1_encoder.model import KeystrokeModel  # noqa: F401


class KeystrokeModelBCE(nn.Module):
    """Wraps KeystrokeModel with a per-token binary classification head.

    Output: (B, T) sigmoid probabilities, one per keystroke.
    """

    def __init__(self, base: KeystrokeModel):
        super().__init__()
        self.base = base
        d_model = base.timing_encoder.d_out + base.key_embedding.embedding_dim
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        tokens = self.base(x)
        return torch.sigmoid(self.head(tokens).squeeze(-1))  # (B, T)
