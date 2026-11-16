import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Embedding lookup scaled by sqrt(d_model) (Sec. 3.4)."""

    def __init__(self, vocab_size: int, d_model: int, pad_id: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * self.scale


class SinusoidalPositionalEncoding(nn.Module):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        if t > self.max_len:
            raise ValueError(f"Sequence length {t} exceeds max_len={self.max_len}")
        return self.dropout(x + self.pe[:, :t].to(x.dtype))
