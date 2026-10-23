"""Significance tests."""

from math import exp, lgamma, log

import numpy as np


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired correctness."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n, k = n01 + n10, min(n01, n10)
    if not n:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    logp = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * log(2) for i in range(k + 1)]
    return {"a_only": n01, "b_only": n10, "p": round(min(1.0, 2 * sum(exp(x) for x in logp)), 6)}


def bootstrap(a, b, n=2000, seed=0):
    """Mean of a - b and its 95% percentile interval, resampling items."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return round(100 * d.mean(), 2), round(100 * np.percentile(m, 2.5), 2), round(100 * np.percentile(m, 97.5), 2)
