"""Model configuration. Defaults reproduce the 'base' model of Vaswani et al. (2017),
except for the feed-forward block which uses SwiGLU (set ffn_type='relu' for the paper's FFN)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Union


@dataclass
class TransformerConfig:
    src_vocab_size: int = 32
    tgt_vocab_size: int = 32
    d_model: int = 512
    n_heads: int = 8
    n_enc_layers: int = 6
    n_dec_layers: int = 6
    d_ff: int = 2048
    ffn_type: str = "swiglu"  # "swiglu" | "relu"
    dropout: float = 0.1
    max_len: int = 512
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    norm_first: bool = False  # False = post-LN (original paper), True = pre-LN
    share_embeddings: bool = False  # share src/tgt embeddings (needs equal vocab sizes)
    tie_output: bool = True  # tie output projection to target embedding

    def __post_init__(self) -> None:
        self.validate()

    @property
    def d_k(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn_hidden(self) -> int:
        """SwiGLU has 3 matrices instead of 2, so the hidden size is scaled by 2/3
        to keep the parameter count comparable to a d_ff-wide ReLU FFN."""
        if self.ffn_type == "relu":
            return self.d_ff
        h = int(2 * self.d_ff / 3)
        return 64 * ((h + 63) // 64)

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_model % 2 != 0:
            raise ValueError("d_model must be even (sinusoidal positional encoding)")
        if self.ffn_type not in ("swiglu", "relu"):
            raise ValueError("ffn_type must be 'swiglu' or 'relu'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.share_embeddings and self.src_vocab_size != self.tgt_vocab_size:
            raise ValueError("share_embeddings requires src_vocab_size == tgt_vocab_size")
        if len({self.pad_id, self.bos_id, self.eos_id}) != 3:
            raise ValueError("pad_id, bos_id and eos_id must be distinct")

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
