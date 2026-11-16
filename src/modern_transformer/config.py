"""Model configuration."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional, Union


@dataclass
class TransformerConfig:
    # --- vocabulary / context ---
    vocab_size: int = 259  # 256 bytes + PAD/BOS/EOS (see ByteTokenizer)
    max_seq_len: int = 1024

    # --- core dimensions ---
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4  # < n_heads => Grouped-Query Attention; ==1 => MQA; ==n_heads => MHA

    # --- feed-forward (SwiGLU) ---
    ffn_mult: float = 8 / 3  # hidden = ffn_mult * d_model, rounded up to `multiple_of`
    multiple_of: int = 64

    # --- norm / positions ---
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # --- regularisation ---
    dropout: float = 0.0

    # --- attention variants ---
    sliding_window: Optional[int] = None  # local attention window (None = full causal)

    # --- Mixture of Experts ---
    use_moe: bool = False
    moe_every: int = 2  # every n-th layer (the last in each group) uses an MoE FFN
    n_experts: int = 8
    moe_top_k: int = 2
    moe_aux_loss_coef: float = 0.01

    # --- misc ---
    tie_embeddings: bool = True
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        h = int(self.ffn_mult * self.d_model)
        return self.multiple_of * ((h + self.multiple_of - 1) // self.multiple_of)

    def is_moe_layer(self, layer_idx: int) -> bool:
        return self.use_moe and (layer_idx % self.moe_every == self.moe_every - 1)

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be >= 1")
        if self.use_moe:
            if self.moe_every < 1:
                raise ValueError("moe_every must be >= 1")
            if not 1 <= self.moe_top_k <= self.n_experts:
                raise ValueError("moe_top_k must be in [1, n_experts]")

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TransformerConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
