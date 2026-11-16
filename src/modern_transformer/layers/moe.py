"""Sparse top-k Mixture-of-Experts feed-forward (Shazeer 2017, Switch/Mixtral style)."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TransformerConfig
from .feedforward import SwiGLUFeedForward


class MoEFeedForward(nn.Module):
    """Each token is routed to its top-k experts; outputs are combined with the
    renormalised router probabilities. Returns (output, load_balancing_loss).

    Load-balancing loss (Switch Transformer):  E * sum_e f_e * P_e
      f_e = fraction of routing slots assigned to expert e
      P_e = mean router probability for expert e
    It equals 1.0 under perfectly uniform routing.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.moe_top_k
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUFeedForward(cfg) for _ in range(cfg.n_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        flat = x.reshape(-1, d)

        probs = F.softmax(self.gate(flat).float(), dim=-1)  # (N, E)
        top_p, top_i = probs.topk(self.top_k, dim=-1)  # (N, k)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token_idx, slot = (top_i == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            y = expert(flat[token_idx])
            w = top_p[token_idx, slot].unsqueeze(-1).to(y.dtype)
            out.index_add_(0, token_idx, (y * w).to(out.dtype))

        dispatch = F.one_hot(top_i, self.n_experts).sum(dim=1).float()  # (N, E)
        f = dispatch.mean(dim=0) / self.top_k
        p = probs.mean(dim=0)
        aux_loss = self.n_experts * (f * p).sum()

        return out.view(b, t, d), aux_loss
