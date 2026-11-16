from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .config import TransformerConfig
from .feedforward import build_ffn


class EncoderLayer(nn.Module):
    """Self-attention sub-layer + feed-forward sub-layer, each wrapped in a residual
    connection and layer norm (post-LN by default, as in the paper)."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.norm_first = cfg.norm_first
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ffn = build_ffn(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.norm_first:
            h = self.norm1(x)
            x = x + self.drop(self.self_attn(h, h, h, src_mask))
            x = x + self.drop(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop(self.self_attn(x, x, x, src_mask)))
            x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_enc_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model) if cfg.norm_first else nn.Identity()

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.final_norm(x)
