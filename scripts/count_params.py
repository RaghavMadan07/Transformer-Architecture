#!/usr/bin/env python
"""Print parameter counts for a config (default: paper base model, 32k vocab)."""
import sys

from seq2seq_transformer import Transformer, TransformerConfig

cfg = TransformerConfig.load(sys.argv[1]) if len(sys.argv) > 1 else TransformerConfig()
cfg.src_vocab_size = cfg.tgt_vocab_size = 32000
m = Transformer(cfg)
print(f"total:         {m.num_parameters() / 1e6:.2f}M")
print(f"non-embedding: {m.num_parameters(non_embedding=True) / 1e6:.2f}M")
