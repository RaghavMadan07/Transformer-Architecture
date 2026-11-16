#!/usr/bin/env python
"""Run a trained checkpoint.

    python scripts/infer.py --checkpoint checkpoints/best.pt --input "5 9 3 12 7"     # synthetic tasks
    python scripts/infer.py --checkpoint checkpoints/best.pt --input "Go home."       # --task text
"""
import argparse

import torch

from seq2seq_transformer import CharTokenizer, Transformer, TransformerConfig, greedy_decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location="cpu")
    cfg = TransformerConfig.from_dict(ckpt["config"])
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    if ckpt["task"] == "text":
        tok = CharTokenizer(ckpt["tokenizer"])
        ids = tok.encode(a.input)
    else:
        ids = [int(x) for x in a.input.replace(",", " ").split()]
    src = torch.tensor([ids + [cfg.eos_id]], device=device)
    out = greedy_decode(model, src, ckpt["max_decode"])[0].tolist()

    if ckpt["task"] == "text":
        print(tok.decode(out))
    else:
        res = []
        for t in out:
            if t == cfg.eos_id:
                break
            res.append(t)
        print(" ".join(map(str, res)))


if __name__ == "__main__":
    main()
