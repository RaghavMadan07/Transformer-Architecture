#!/usr/bin/env python
"""Train a Transformer on a plain-text file with a byte-level tokenizer.

Example:
    python scripts/train.py --data data/input.txt --config configs/tiny.json --max_steps 2000
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from modern_transformer import Transformer, TransformerConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to a UTF-8 text file")
    p.add_argument("--config", default=None, help="path to a JSON config (default: built-in defaults)")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=None, help="defaults to config.max_seq_len")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def lr_at(step, a):
    if step < a.warmup_steps:
        return a.lr * (step + 1) / a.warmup_steps
    if step >= a.max_steps:
        return a.min_lr
    ratio = (step - a.warmup_steps) / max(1, a.max_steps - a.warmup_steps)
    return a.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (a.lr - a.min_lr)


def get_batch(data: torch.Tensor, bs: int, seq_len: int, device):
    ix = torch.randint(len(data) - seq_len - 1, (bs,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).long()
    y = torch.stack([data[i + 1 : i + 1 + seq_len] for i in ix]).long()
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, a, seq_len, device, ctx):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = []
        for _ in range(a.eval_iters):
            x, y = get_batch(data, a.batch_size, seq_len, device)
            with ctx:
                losses.append(model(x, targets=y).lm_loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    a = parse_args()
    torch.manual_seed(a.seed)

    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if "cuda" in device else "cpu"
    if a.dtype == "auto":
        dtype = (
            torch.bfloat16
            if device_type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
    else:
        dtype = getattr(torch, a.dtype)
    ctx = (
        torch.autocast(device_type=device_type, dtype=dtype)
        if dtype != torch.float32
        else torch.autocast(device_type=device_type, enabled=False)
    )
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == torch.float16))

    # ---- model ----
    ckpt = torch.load(a.resume, map_location="cpu") if a.resume else None
    if ckpt:
        cfg = TransformerConfig.from_dict(ckpt["config"])
    elif a.config:
        cfg = TransformerConfig.load(a.config)
    else:
        cfg = TransformerConfig()
    seq_len = a.seq_len or cfg.max_seq_len
    assert seq_len <= cfg.max_seq_len, "seq_len cannot exceed config.max_seq_len"

    model = Transformer(cfg).to(device)
    optimizer = model.configure_optimizer(a.lr, a.weight_decay)
    step, best_val = 0, float("inf")
    if ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, best_val = ckpt["step"], ckpt.get("best_val", best_val)
        print(f"Resumed from {a.resume} at step {step}")
    print(f"Parameters: {model.num_parameters() / 1e6:.2f}M  | device={device} dtype={dtype}")
    train_model = torch.compile(model) if a.compile else model

    # ---- data (raw bytes -> ids 0..255) ----
    raw = np.frombuffer(Path(a.data).read_bytes(), dtype=np.uint8).copy()
    n = int(0.9 * len(raw))
    splits = {"train": torch.from_numpy(raw[:n]), "val": torch.from_numpy(raw[n:])}
    assert len(splits["val"]) > seq_len + 1, "Dataset too small for this seq_len"
    print(f"Tokens: train={n:,} val={len(raw) - n:,}")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(name):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.to_dict(),
                "step": step,
                "best_val": best_val,
            },
            out_dir / name,
        )

    model.train()
    t0 = time.time()
    while step < a.max_steps:
        lr = lr_at(step, a)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % a.eval_interval == 0 and step > 0:
            losses = estimate_loss(train_model, splits, a, seq_len, device, ctx)
            print(f"[eval] step {step}: train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save("best.pt")

        optimizer.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):
            x, y = get_batch(splits["train"], a.batch_size, seq_len, device)
            with ctx:
                out = train_model(x, targets=y)
                loss = out.loss / a.grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1

        if step % a.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            toks = a.batch_size * a.grad_accum * seq_len * a.log_interval
            print(
                f"step {step:>6} | loss {out.lm_loss.item():.4f} | aux {out.aux_loss.item():.3f} "
                f"| lr {lr:.2e} | gnorm {gnorm:.2f} | {toks / dt:,.0f} tok/s"
            )

    save("last.pt")
    print(f"Done. Checkpoints in {out_dir}/")


if __name__ == "__main__":
    main()
