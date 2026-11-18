import math


def lr_at(step: int, *, lr: float, warmup: int, max_steps: int, min_ratio: float, kind: str) -> float:
    """Warmup linear seguido de cosine, constante ou WSD (warmup-stable-decay, decai nos últimos 20%)."""
    if step < warmup:
        return lr * (step + 1) / warmup
    min_lr = lr * min_ratio
    if kind == "constant":
        return lr
    if kind == "cosine":
        progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))
    if kind == "wsd":
        decay_start = int(max_steps * 0.8)
        if step < decay_start:
            return lr
        progress = min(1.0, (step - decay_start) / max(1, max_steps - decay_start))
        return lr - (lr - min_lr) * progress
    raise ValueError(f"schedule desconhecido: {kind}")
