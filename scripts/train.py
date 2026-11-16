#!/usr/bin/env python
"""Train the encoder-decoder Transformer with teacher forcing.

Synthetic tasks (no data needed):
    python scripts/train.py --task reverse --config configs/tiny.json --epochs 15

Real parallel text (tab-separated "source<TAB>target" per line):
    python scripts/train.py --task text --data data/fra.txt --config configs/tiny.json
"""
import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from seq2seq_transformer import CharTokenizer, Transformer, TransformerConfig, greedy_decode
from seq2seq_transformer.data import (
    ParallelTextDataset,
    Seq2SeqCollator,
    SyntheticSeq2Seq,
    read_tsv_pairs,
)
from seq2seq_transformer.scheduler import build_scheduler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="reverse", choices=["copy", "reverse", "sort", "text"])
    p.add_argument("--data", default=None, help="TSV file for --task text")
    p.add_argument("--max_pairs", type=int, default=0, help="limit #pairs read from the TSV (0 = all)")
    p.add_argument("--config", default=None, help="JSON config (default: paper base model)")
    p.add_argument("--out_dir", default="checkpoints")
    # synthetic data
    p.add_argument("--vocab_size", type=int, default=32)
    p.add_argument("--min_len", type=int, default=3)
    p.add_argument("--max_seq", type=int, default=20)
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_val", type=int, default=1000)
    # optimisation
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "noam"])
    p.add_argument("--lr", type=float, default=5e-4, help="peak LR (cosine only)")
    p.add_argument("--noam_factor", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def cut_at_eos(seq, eos_id):
    out = []
    for t in seq:
        if t == eos_id:
            break
        out.append(t)
    return out


def run_epoch(model, loader, device, ctx, label_smoothing, optimizer=None, scheduler=None, clip=1.0):
    train = optimizer is not None
    model.train(train)
    cfg = model.cfg
    loss_sum, n_tok = 0.0, 0
    for batch in loader:
        src, tin, tout = (batch[k].to(device) for k in ("src", "tgt_in", "tgt_out"))
        with torch.set_grad_enabled(train):
            with ctx:
                logits = model(src, tin)
            n = (tout != cfg.pad_id).sum()
            loss_total = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                tout.reshape(-1),
                ignore_index=cfg.pad_id,
                label_smoothing=label_smoothing if train else 0.0,
                reduction="sum",
            )
        if train:
            optimizer.zero_grad(set_to_none=True)
            (loss_total / n).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
        loss_sum += loss_total.item()
        n_tok += n.item()
    return loss_sum / max(1, n_tok)


@torch.no_grad()
def sequence_accuracy(model, loader, device, max_decode_len, max_batches=10):
    eos = model.cfg.eos_id
    correct = total = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        pred = greedy_decode(model, batch["src"].to(device), max_decode_len).tolist()
        for p, t in zip(pred, batch["tgt_out"].tolist()):
            correct += int(cut_at_eos(p, eos) == cut_at_eos(t, eos))
            total += 1
    return correct / max(1, total)


def main():
    a = parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if "cuda" in device else "cpu"
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if a.amp and device_type == "cuda"
        else torch.autocast(device_type=device_type, enabled=False)
    )

    # ---------------- data ----------------
    tok = None
    if a.task == "text":
        assert a.data, "--data is required for --task text"
        pairs = read_tsv_pairs(a.data, a.max_pairs)
        random.Random(a.seed).shuffle(pairs)
        tok = CharTokenizer.from_texts(s + t for s, t in pairs)
        n_val = max(1, int(0.05 * len(pairs)))
        max_len_cfg = 256
        train_ds = ParallelTextDataset(pairs[n_val:], tok, max_len_cfg)
        val_ds = ParallelTextDataset(pairs[:n_val], tok, max_len_cfg)
        vocab = tok.vocab_size
        max_decode = max_len_cfg - 2
        print(f"Loaded {len(pairs):,} pairs | char vocab = {vocab}")
    else:
        train_ds = SyntheticSeq2Seq(a.task, a.n_train, a.vocab_size, a.min_len, a.max_seq, seed=a.seed)
        val_ds = SyntheticSeq2Seq(a.task, a.n_val, a.vocab_size, a.min_len, a.max_seq, seed=a.seed + 1)
        vocab = a.vocab_size
        max_decode = a.max_seq + 2

    collate = Seq2SeqCollator()
    train_loader = DataLoader(train_ds, a.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, a.batch_size, shuffle=False, collate_fn=collate)

    # ---------------- model ----------------
    cfg = TransformerConfig.load(a.config) if a.config else TransformerConfig()
    cfg.src_vocab_size = cfg.tgt_vocab_size = vocab
    cfg.max_len = max(cfg.max_len, max_decode + 2)
    cfg.validate()
    model = Transformer(cfg).to(device)
    print(
        f"Parameters: {model.num_parameters() / 1e6:.2f}M "
        f"({model.num_parameters(non_embedding=True) / 1e6:.2f}M non-embedding) | device={device}"
    )

    steps_total = a.epochs * len(train_loader)
    base_lr = 1.0 if a.scheduler == "noam" else a.lr  # noam lambda returns the absolute LR
    optimizer = model.configure_optimizer(lr=base_lr, weight_decay=a.weight_decay)
    scheduler = build_scheduler(
        a.scheduler, optimizer, cfg.d_model, a.warmup_steps, steps_total, a.noam_factor
    )

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, ctx, a.label_smoothing, optimizer, scheduler, a.grad_clip)
        va = run_epoch(model, val_loader, device, ctx, 0.0)
        msg = (
            f"epoch {epoch:>3}/{a.epochs} | train {tr:.4f} | val {va:.4f} "
            f"(ppl {math.exp(min(va, 20)):.2f}) | lr {optimizer.param_groups[0]['lr']:.2e}"
        )
        if a.task != "text":
            msg += f" | exact-match {sequence_accuracy(model, val_loader, device, max_decode):.1%}"
        print(msg + f" | {time.time() - t0:.1f}s")

        if va < best:
            best = va
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "task": a.task,
                    "tokenizer": tok.chars if tok else None,
                    "max_decode": max_decode,
                    "epoch": epoch,
                    "val_loss": va,
                },
                out_dir / "best.pt",
            )

    if tok is not None:  # show a few translations
        for i in range(min(5, len(val_ds))):
            s, t = val_ds[i]
            src = torch.tensor([s + [cfg.eos_id]], device=device)
            hyp = greedy_decode(model, src, max_decode)[0].tolist()
            print(f"\nSRC: {tok.decode(s)}\nREF: {tok.decode(t)}\nHYP: {tok.decode(hyp)}")
    print(f"\nBest val loss {best:.4f}. Checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
