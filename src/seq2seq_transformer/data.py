"""Datasets for sequence-to-sequence training.

* SyntheticSeq2Seq  — copy / reverse / sort tasks on random integer sequences (no downloads).
* ParallelTextDataset — tab-separated parallel text (e.g. Tatoeba / Anki en-fr pairs).
"""
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import BOS_ID, EOS_ID, NUM_SPECIAL, PAD_ID, CharTokenizer

TASKS = ("copy", "reverse", "sort")


class SyntheticSeq2Seq(Dataset):
    def __init__(
        self, task: str, n: int, vocab_size: int, min_len: int, max_len: int, seed: int = 0
    ):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if vocab_size <= NUM_SPECIAL:
            raise ValueError("vocab_size must be larger than the number of special tokens")
        rng = random.Random(seed)
        self.items: List[Tuple[List[int], List[int]]] = []
        for _ in range(n):
            length = rng.randint(min_len, max_len)
            src = [rng.randrange(NUM_SPECIAL, vocab_size) for _ in range(length)]
            if task == "copy":
                tgt = list(src)
            elif task == "reverse":
                tgt = src[::-1]
            else:
                tgt = sorted(src)
            self.items.append((src, tgt))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


class ParallelTextDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[str, str]], tok: CharTokenizer, max_len: int):
        self.items = [
            (tok.encode(s)[: max_len - 1], tok.encode(t)[: max_len - 1]) for s, t in pairs
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


def read_tsv_pairs(path: str, limit: int = 0) -> List[Tuple[str, str]]:
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) >= 2 and cols[0].strip() and cols[1].strip():
            pairs.append((cols[0].strip(), cols[1].strip()))
        if limit and len(pairs) >= limit:
            break
    return pairs


class Seq2SeqCollator:
    """Pads a batch and builds teacher-forcing tensors:
        src     = tokens + [EOS]
        tgt_in  = [BOS] + tokens        (decoder input, shifted right)
        tgt_out = tokens + [EOS]        (labels, shifted left)
    """

    def __init__(self, pad_id: int = PAD_ID, bos_id: int = BOS_ID, eos_id: int = EOS_ID):
        self.pad, self.bos, self.eos = pad_id, bos_id, eos_id

    def _pad(self, seqs: List[List[int]]) -> torch.Tensor:
        width = max(len(s) for s in seqs)
        return torch.tensor([s + [self.pad] * (width - len(s)) for s in seqs], dtype=torch.long)

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        src = [s + [self.eos] for s, _ in batch]
        tgt_in = [[self.bos] + t for _, t in batch]
        tgt_out = [t + [self.eos] for _, t in batch]
        return {"src": self._pad(src), "tgt_in": self._pad(tgt_in), "tgt_out": self._pad(tgt_out)}
