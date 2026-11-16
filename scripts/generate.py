#!/usr/bin/env python
"""Sample text from a trained checkpoint.

Example:
    python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "To be, or not" --top_p 0.9
"""
import argparse

import torch

from modern_transformer import ByteTokenizer, Transformer, TransformerConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    if a.seed is not None:
        torch.manual_seed(a.seed)
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(a.checkpoint, map_location="cpu")
    model = Transformer(TransformerConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    tok = ByteTokenizer()
    ids = torch.tensor([tok.encode(a.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=a.max_new_tokens,
        temperature=a.temperature,
        top_k=a.top_k,
        top_p=a.top_p,
        repetition_penalty=a.repetition_penalty,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
