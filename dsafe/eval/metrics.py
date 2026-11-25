"""Transfer metrics and confidence intervals."""
from __future__ import annotations

import random


def transfer_ratio(s_toxic: float, s_clean: float, t_toxic: float, t_clean: float, eps: float = 1e-6) -> float | None:
    """TR = (M(S_toxicT) - M(S_cleanT)) / (M(T_toxic) - M(T_clean)).

    1 = full inheritance, 0 = none. Returns None when the teacher gap is ~0 (undefined)."""
    denom = t_toxic - t_clean
    if abs(denom) < eps:
        return None
    return (s_toxic - s_clean) / denom


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI of a list of per-sample scores (e.g. 0/1 hits)."""
    if not values:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return sum(values) / n, lo, hi
