import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig


class ReLUFeedForward(nn.Module):
    """Original position-wise FFN: W2( dropout( relu(W1 x) ) )."""

    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden)
        self.w2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.relu(self.w1(x))))


class SwiGLUFeedForward(nn.Module):
    """SwiGLU FFN (Shazeer, 2020): W2( silu(W1 x) * W3 x )."""

    def __init__(self, d_model: int, hidden: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)  # gate projection
        self.w3 = nn.Linear(d_model, hidden, bias=False)  # value projection
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


def build_ffn(cfg: TransformerConfig) -> nn.Module:
    cls = SwiGLUFeedForward if cfg.ffn_type == "swiglu" else ReLUFeedForward
    return cls(cfg.d_model, cfg.ffn_hidden, cfg.dropout)
