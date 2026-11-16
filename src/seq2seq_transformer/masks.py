"""Boolean attention masks. Convention everywhere: True = position MAY be attended to."""
import torch


def make_pad_mask(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(B, T) token ids -> (B, 1, 1, T) mask hiding padding keys."""
    return (x != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(t: int, device=None) -> torch.Tensor:
    """(1, 1, T, T) lower-triangular mask: position i sees positions <= i."""
    return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))[None, None]


def make_src_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    return make_pad_mask(src, pad_id)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Padding mask AND causal mask -> (B, 1, T, T)."""
    return make_pad_mask(tgt, pad_id) & make_causal_mask(tgt.size(1), tgt.device)
