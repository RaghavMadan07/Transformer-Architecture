import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TransformerConfig


class SwiGLUFeedForward(nn.Module):
    """SwiGLU FFN (Shazeer, 2020): w2( silu(w1 x) * w3 x )."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        h = cfg.ffn_hidden
        self.w1 = nn.Linear(cfg.d_model, h, bias=False)  # gate
        self.w3 = nn.Linear(cfg.d_model, h, bias=False)  # up
        self.w2 = nn.Linear(h, cfg.d_model, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
