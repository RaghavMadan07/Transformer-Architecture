"""Rotary position embeddings (Su et al., 2021), 'rotate-half' formulation."""
from typing import Optional, Tuple

import torch


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin), each of shape (max_seq_len, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (L, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (L, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q: (B, Hq, T, D), k: (B, Hkv, T, D), cos/sin: (T, D)."""
    cos = cos[None, None].to(q.dtype)
    sin = sin[None, None].to(q.dtype)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k
