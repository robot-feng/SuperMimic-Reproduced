"""Cosine learning-rate schedule (paper Table A.5): decay to `final_ratio` over `decay_fraction` of training, then constant."""

from __future__ import annotations

import math


def cosine_lr(step: int, total_steps: int, base_lr: float, final_ratio: float = 0.03, decay_fraction: float = 250_000 / 320_000) -> float:
    decay_steps = max(1, int(total_steps * decay_fraction))
    if step >= decay_steps:
        return base_lr * final_ratio
    cos = 0.5 * (1.0 + math.cos(math.pi * step / decay_steps))
    return base_lr * (final_ratio + (1.0 - final_ratio) * cos)


def training_learning_rates(
    step: int,
    total_steps: int,
    wm_lr: float,
    actor_lr: float,
    critic_lr: float,
    final_ratio: float = 0.03,
    decay_fraction: float = 250_000 / 320_000,
) -> tuple[float, float, float]:
    """Apply the same schedule multiplier to each independently configured rate."""
    factor = cosine_lr(step, total_steps, 1.0, final_ratio, decay_fraction)
    return factor * wm_lr, factor * actor_lr, factor * critic_lr
