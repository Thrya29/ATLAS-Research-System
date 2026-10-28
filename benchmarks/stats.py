"""Seed-level statistics: bootstrap CIs, paired tests, effect sizes, Holm correction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def bootstrap_ci(x: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Compare per-seed values a (method) vs b (baseline) on the same seeds."""
    d = np.asarray(a, float) - np.asarray(b, float)
    mean, lo, hi = bootstrap_ci(d)
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else float("inf") * np.sign(d.mean())
    return {"diff": mean, "lo": lo, "hi": hi, "p": p, "dz": dz}


def holm(pvals: Sequence[float]) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
