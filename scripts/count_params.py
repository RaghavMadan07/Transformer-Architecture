#!/usr/bin/env python
"""Print parameter counts for a config file."""
import sys

from modern_transformer import Transformer, TransformerConfig

cfg = TransformerConfig.load(sys.argv[1]) if len(sys.argv) > 1 else TransformerConfig()
m = Transformer(cfg)
print(f"total params:         {m.num_parameters() / 1e6:.2f}M")
print(f"non-embedding params: {m.num_parameters(non_embedding=True) / 1e6:.2f}M")
