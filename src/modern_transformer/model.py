"""Decoder-only Transformer language model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .block import TransformerBlock
from .config import TransformerConfig
from .layers import RMSNorm, build_rope_cache
from .layers.attention import KVCache


@dataclass
class TransformerOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None  # lm_loss + coef * aux_loss
    lm_loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None  # mean MoE load-balancing loss (0 if dense)
    past_key_values: Optional[List[KVCache]] = None


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style: scale residual-branch output projections by 1/sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[KVCache]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        last_token_only: bool = False,
    ) -> TransformerOutput:
        """
        input_ids:       (B, T) token ids.
        targets:         (B, T) next-token labels already aligned with input_ids
                         (i.e. targets[:, i] is the token that follows input_ids[:, i]);
                         use -100 to ignore positions.
        past_key_values: list of per-layer (k, v) from a previous call.
        attention_mask:  (B, past_len + T), 1 = real token, 0 = padding (optional).
        last_token_only: only compute logits for the final position (fast decoding).
        """
        b, t = input_ids.shape
        start = past_key_values[0][0].size(2) if past_key_values else 0
        if start + t > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {start + t} exceeds max_seq_len={self.cfg.max_seq_len}"
            )
        cos = self.rope_cos[start : start + t]
        sin = self.rope_sin[start : start + t]

        x = self.drop(self.tok_emb(input_ids))

        presents: List[KVCache] = []
        aux_losses = []
        use_ckpt = self.cfg.gradient_checkpointing and self.training and not use_cache
        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values else None
            if use_ckpt:
                x, present, aux = checkpoint(
                    block, x, cos, sin, None, False, attention_mask, use_reentrant=False
                )
            else:
                x, present, aux = block(x, cos, sin, past, use_cache, attention_mask)
            if use_cache:
                presents.append(present)
            if block.is_moe:
                aux_losses.append(aux)

        x = self.norm(x)
        if last_token_only:
            x = x[:, -1:, :]
        logits = self.lm_head(x)

        loss = lm_loss = aux_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
            aux_loss = torch.stack(aux_losses).mean() if aux_losses else lm_loss.new_zeros(())
            loss = lm_loss + self.cfg.moe_aux_loss_coef * aux_loss

        return TransformerOutput(
            logits=logits,
            loss=loss,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            past_key_values=presents if use_cache else None,
        )

    # ------------------------------------------------------------------
    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def configure_optimizer(
        self,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        betas=(0.9, 0.95),
    ) -> torch.optim.AdamW:
        """AdamW with weight decay only on >=2-D tensors (matrices / embeddings)."""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if p.requires_grad:
                (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        from .generation import generate

        return generate(self, input_ids, **kwargs)
