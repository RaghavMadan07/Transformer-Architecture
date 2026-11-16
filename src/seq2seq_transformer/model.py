"""Encoder-decoder Transformer ("Attention Is All You Need")."""
import math

import torch
import torch.nn as nn

from .config import TransformerConfig
from .decoder import Decoder
from .embeddings import SinusoidalPositionalEncoding, TokenEmbedding
from .encoder import Encoder
from .masks import make_src_mask, make_tgt_mask


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tgt_tok = TokenEmbedding(cfg.tgt_vocab_size, cfg.d_model, cfg.pad_id)
        self.src_tok = (
            self.tgt_tok
            if cfg.share_embeddings
            else TokenEmbedding(cfg.src_vocab_size, cfg.d_model, cfg.pad_id)
        )
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.generator = nn.Linear(cfg.d_model, cfg.tgt_vocab_size, bias=False)
        if cfg.tie_output:
            self.generator.weight = self.tgt_tok.embedding.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # embeddings: N(0, d_model^-0.5) so that after the sqrt(d_model) scale they have unit variance
        for emb in {id(m): m for m in (self.src_tok, self.tgt_tok)}.values():
            nn.init.normal_(emb.embedding.weight, mean=0.0, std=self.cfg.d_model ** -0.5)
            with torch.no_grad():
                emb.embedding.weight[self.cfg.pad_id].zero_()

    # ------------------------------------------------------------------
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.pos(self.src_tok(src)), src_mask)

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(self.pos(self.tgt_tok(tgt_in)), memory, src_mask, tgt_mask)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward pass.

        src:    (B, S)  source ids (ending in EOS, padded with pad_id)
        tgt_in: (B, T)  decoder input = [BOS] + target tokens (shifted right)
        returns logits (B, T, tgt_vocab_size) predicting tgt_in shifted left.
        """
        src_mask = make_src_mask(src, self.cfg.pad_id)
        tgt_mask = make_tgt_mask(tgt_in, self.cfg.pad_id)
        memory = self.encode(src, src_mask)
        return self.generator(self.decode(tgt_in, memory, src_mask, tgt_mask))

    # ------------------------------------------------------------------
    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights are counted once
        if non_embedding:
            seen = {id(self.tgt_tok.embedding.weight): self.tgt_tok.embedding.weight}
            seen[id(self.src_tok.embedding.weight)] = self.src_tok.embedding.weight
            n -= sum(w.numel() for w in seen.values())
        return n

    def configure_optimizer(
        self, lr: float = 1e-3, weight_decay: float = 0.01, betas=(0.9, 0.98), eps: float = 1e-9
    ) -> torch.optim.AdamW:
        """AdamW (paper's beta/eps) with weight decay only on matrices/embeddings."""
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if p.requires_grad:
                (decay if p.dim() >= 2 else no_decay).append(p)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas,
            eps=eps,
        )
