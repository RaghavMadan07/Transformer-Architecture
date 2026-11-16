import torch

from .masks import make_src_mask, make_tgt_mask
from .model import Transformer


@torch.no_grad()
def greedy_decode(model: Transformer, src: torch.Tensor, max_len: int) -> torch.Tensor:
    """Autoregressive greedy decoding. The encoder runs once; the decoder is re-run on the
    growing prefix each step (no KV-cache — simple and exact).

    Returns (B, <=max_len) generated ids (BOS stripped; finished rows are padded after EOS).
    """
    cfg = model.cfg
    was_training = model.training
    model.eval()

    max_len = min(max_len, cfg.max_len - 1)
    b = src.size(0)
    src_mask = make_src_mask(src, cfg.pad_id)
    memory = model.encode(src, src_mask)

    ys = torch.full((b, 1), cfg.bos_id, dtype=torch.long, device=src.device)
    finished = torch.zeros(b, dtype=torch.bool, device=src.device)
    for _ in range(max_len):
        out = model.decode(ys, memory, src_mask, make_tgt_mask(ys, cfg.pad_id))
        nxt = model.generator(out[:, -1]).argmax(dim=-1)
        nxt = torch.where(finished, torch.full_like(nxt, cfg.pad_id), nxt)
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        finished = finished | (nxt == cfg.eos_id)
        if bool(finished.all()):
            break

    model.train(was_training)
    return ys[:, 1:]
