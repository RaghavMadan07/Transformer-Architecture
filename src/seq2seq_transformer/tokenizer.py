from typing import Iterable, List

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
NUM_SPECIAL = 4


class CharTokenizer:
    """Character-level tokenizer with PAD/BOS/EOS/UNK specials. Swap for BPE/SentencePiece
    on real translation data (just update vocab sizes in the config)."""

    def __init__(self, chars: List[str]):
        self.chars = list(chars)
        self.stoi = {c: i + NUM_SPECIAL for i, c in enumerate(self.chars)}

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CharTokenizer":
        return cls(sorted(set("".join(texts))))

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + NUM_SPECIAL

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, UNK_ID) for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == EOS_ID:
                break
            if i >= NUM_SPECIAL:
                out.append(self.chars[i - NUM_SPECIAL])
        return "".join(out)
