"""Learning-rate schedules, returned as LambdaLR multipliers."""
import math
from typing import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def noam_lambda(d_model: int, warmup_steps: int, factor: float = 1.0) -> Callable[[int], float]:
    """Paper schedule (Eq. 3):  lr = factor * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5).

    NOTE: this returns the *absolute* learning rate, so build the optimizer with lr=1.0.
    """

    def fn(step: int) -> float:
        s = step + 1
        return factor * d_model ** -0.5 * min(s ** -0.5, s * warmup_steps ** -1.5)

    return fn


def cosine_lambda(warmup_steps: int, total_steps: int, min_ratio: float = 0.1) -> Callable[[int], float]:
    """Linear warmup then cosine decay to min_ratio * peak. Multiplier of the optimizer's lr."""

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        ratio = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * ratio))

    return fn


def build_scheduler(
    name: str,
    optimizer: Optimizer,
    d_model: int,
    warmup_steps: int,
    total_steps: int,
    noam_factor: float = 1.0,
    min_ratio: float = 0.1,
) -> LambdaLR:
    if name == "noam":
        return LambdaLR(optimizer, noam_lambda(d_model, warmup_steps, noam_factor))
    if name == "cosine":
        return LambdaLR(optimizer, cosine_lambda(warmup_steps, total_steps, min_ratio))
    raise ValueError(f"Unknown scheduler '{name}'")
