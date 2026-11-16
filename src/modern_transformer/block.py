from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import TransformerConfig
from .layers import GroupedQueryAttention, MoEFeedForward, RMSNorm, SwiGLUFeedForward
from .layers.attention import KVCache


class TransformerBlock(nn.Module):
    """Pre-norm block:  x + Attn(norm(x));  x + FFN(norm(x))."""

    def __init__(self, cfg: TransformerConfig, layer_idx: int):
        super().__init__()
        self.is_moe = cfg.is_moe_layer(layer_idx)
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = MoEFeedForward(cfg) if self.is_moe else SwiGLUFeedForward(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], torch.Tensor]:
        h, present = self.attn(self.attn_norm(x), cos, sin, past_kv, use_cache, key_padding_mask)
        x = x + self.drop(h)

        if self.is_moe:
            f, aux = self.ffn(self.ffn_norm(x))
        else:
            f = self.ffn(self.ffn_norm(x))
            aux = x.new_zeros((), dtype=torch.float32)
        x = x + self.drop(f)
        return x, present, aux
