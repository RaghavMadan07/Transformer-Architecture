"""Grouped-Query Attention with RoPE, KV-cache, sliding-window and padding-mask support."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TransformerConfig
from .rope import apply_rotary_emb

KVCache = Tuple[torch.Tensor, torch.Tensor]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, Hkv, S, D) -> (B, Hkv * n_rep, S, D)."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def build_attention_mask(
    t: int,
    s: int,
    device: torch.device,
    window: Optional[int] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Boolean mask (True = may attend) of shape (1|B, 1, T, S).

    The T queries are the *last* T positions of the S keys (this is what makes
    incremental decoding with a KV-cache work).
    """
    q_pos = torch.arange(s - t, s, device=device)[:, None]
    k_pos = torch.arange(s, device=device)[None, :]
    mask = k_pos <= q_pos
    if window is not None:
        mask = mask & ((q_pos - k_pos) < window)
    mask = mask[None, None]
    if key_padding_mask is not None:
        mask = mask & key_padding_mask[:, None, None, :].bool()
        # Rows that are fully masked (queries sitting on padding) would give NaNs
        # in softmax; let them attend everywhere — their outputs are ignored anyway.
        mask = mask | ~mask.any(dim=-1, keepdim=True)
    return mask


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.window = cfg.sliding_window
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None
        s = k.size(2)

        # cache stores the compact (n_kv_heads) tensors; expand only for compute
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        drop = self.dropout if self.training else 0.0
        if t == s and self.window is None and key_padding_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        else:
            mask = build_attention_mask(t, s, x.device, self.window, key_padding_mask)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)

        out = out.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim)
        return self.wo(out), present
