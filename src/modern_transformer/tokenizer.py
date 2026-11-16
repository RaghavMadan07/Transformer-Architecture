from typing import List


class ByteTokenizer:
    """Dependency-free byte-level tokenizer: ids 0-255 are raw UTF-8 bytes,
    followed by three special tokens. Swap in BPE/SentencePiece for real use
    (just set `vocab_size` in the config accordingly)."""

    PAD = 256
    BOS = 257
    EOS = 258
    vocab_size = 259

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids = [self.BOS] + ids
        if add_eos:
            ids = ids + [self.EOS]
        return ids

    def decode(self, ids) -> str:
        data = bytes(int(i) for i in ids if 0 <= int(i) < 256)
        return data.decode("utf-8", errors="replace")
