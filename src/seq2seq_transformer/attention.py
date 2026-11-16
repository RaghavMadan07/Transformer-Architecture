"""Scaled dot-product attention and multi-head attention (Sec. 3.2 of the paper)."""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    q: (..., Tq, d_k)   k, v: (..., Tk, d_k)
    mask: bool, broadcastable to (..., Tq, Tk); True = keep, False = block.
    Returns (output, attention_weights).
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = F.softmax(scores, dim=-1)
    attended = dropout(weights) if dropout is not None else weights
    return torch.matmul(attended, v), weights


class MultiHeadAttention(nn.Module):
    """MultiHead(Q,K,V) = Concat(head_1..head_h) W_O, head_i = Attention(Q W_Qi, K W_Ki, V W_Vi).

    Used for encoder self-attention, masked decoder self-attention and
    encoder-decoder cross-attention (query from decoder, key/value from encoder).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.store_attn = False  # set True to keep the last attention map for visualisation
        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, tq, _ = query.shape
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        out, weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        if self.store_attn:
            self.attn_weights = weights.detach()

        out = out.transpose(1, 2).contiguous().view(b, tq, self.n_heads * self.d_k)
        return self.w_o(out)
