"""Autoregressive sampling with KV-cache."""
from typing import Optional

import torch
import torch.nn.functional as F


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    remove = (cum - probs) > p  # always keeps the top token
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Returns (B, T + new_tokens). temperature=0 => greedy."""
    was_training = model.training
    model.eval()

    ids = input_ids
    b, t0 = ids.shape
    budget = min(max_new_tokens, model.cfg.max_seq_len - t0)
    if budget <= 0:
        raise ValueError("Prompt already fills the context window (max_seq_len).")

    past = None
    cur = ids
    finished = torch.zeros(b, dtype=torch.bool, device=ids.device)

    for _ in range(budget):
        out = model(
            cur,
            past_key_values=past,
            use_cache=True,
            attention_mask=attention_mask,
            last_token_only=True,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            score = logits.gather(1, ids)
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits = logits.scatter(1, ids, score)

        if temperature == 0.0:
            nxt = logits.argmax(dim=-1)
        else:
            logits = logits / temperature
            if top_k is not None:
                logits = _top_k_filter(logits, top_k)
            if top_p is not None:
                logits = _top_p_filter(logits, top_p)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)

        if eos_token_id is not None:
            nxt = torch.where(finished, torch.full_like(nxt, eos_token_id), nxt)
            finished = finished | (nxt == eos_token_id)

        ids = torch.cat([ids, nxt[:, None]], dim=1)
        cur = nxt[:, None]
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, 1)], dim=1)
        if eos_token_id is not None and bool(finished.all()):
            break

    model.train(was_training)
    return ids
