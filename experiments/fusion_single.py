"""Learned fusion of the views on single-answer questions.

Each view is mapped to the answers that contradict the fewest of its judgements. Per model we average the
pignistic share (BetP), the certain share (Bel) and the possible share (Pl) of every answer, and a conditional
logit over the answers of a question learns how much each model is worth.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from common import load_split
from views_io import call_index, load_views

FREE = ["Qwen3.5-4B", "Qwen3-8B", "GLM-4-9B-0414"]
CHEAP = "Qwen3.5-122B-A10B"
FRONT = ["DeepSeek-V4-Pro", "Kimi-K2.6"]
STRATEGIES = ("direct", "judge", "eliminate", "statements")


@dataclass
class Item:
    uid: str
    cands: list
    gold: int
    compat: dict = field(default_factory=dict)
    calls: dict = field(default_factory=dict)


def candidates(q):
    opts = sorted(q.options)
    if q.qtype == "multiple":
        return [frozenset(c) for r in range(1, len(opts) + 1) for c in combinations(opts, r)]
    return [frozenset([o]) for o in opts]


def compatible(q, cands, strategy, vec):
    acc = {u for u, x in vec.items() if x == "+"}
    rej = {u for u, x in vec.items() if x == "-"}
    if not acc and not rej:
        return None
    if strategy == "statements":
        if not q.is_statement_combo:
            return None
        units = [set(q.option_statements[next(iter(a))]) for a in cands]
    else:
        units = [set(a) for a in cands]
    d = [len(acc - u) + len(rej & u) for u in units]
    m = min(d)
    return tuple(i for i, x in enumerate(d) if x == m)


def load_items(splits):
    items = []
    for sp in splits:
        views = load_views(sp)
        calls = call_index(sp)
        for q in load_split(sp):
            cands = candidates(q)
            gold = cands.index(frozenset(q.answer)) if frozenset(q.answer) in cands else -1
            it = Item(q.uid, cands, gold, calls=calls.get(q.uid, {}))
            for k, x in views.items():
                vec = x.get(q.uid)
                if vec:
                    c = compatible(q, cands, k.split("__")[1], vec)
                    if c is not None:
                        it.compat[k] = c
            items.append(it)
    return items


def evidence(it, keys):
    n = len(it.cands)
    out = {}
    for k in keys:
        c = it.compat.get(k)
        if c is None:
            continue
        g = out.setdefault(k.split("__")[0], [np.zeros(n), np.zeros(n), np.zeros(n), 0])
        g[0][list(c)] += 1 / len(c)
        g[2][list(c)] += 1
        if len(c) == 1:
            g[1][c[0]] += 1
        g[3] += 1
    return {g: (b / m, l / m, p / m) for g, (b, l, p, m) in out.items()}


def plurality(it, keys, w):
    s = np.zeros(len(it.cands))
    for k in keys:
        c = it.compat.get(k)
        if c is not None:
            s[list(c)] += w.get(k, 1.0) / len(c)
    return s


class Stacker:
    def __init__(self, keys, lam=1.0):
        self.keys, self.lam = keys, lam
        self.models = sorted({k.split("__")[0] for k in keys})

    def phi(self, it):
        n = len(it.cands)
        ev = evidence(it, self.keys)
        cols = []
        for g in self.models:
            cols += list(ev.get(g, (np.zeros(n), np.zeros(n), np.zeros(n))))
        size = np.array([len(a) for a in it.cands], float)
        multi = float(len(it.cands) > 5 or any(len(a) > 1 for a in it.cands))
        for s in (1, 2, 3):
            cols.append(multi * (size == s))
        return np.stack(cols, 1)

    def fit(self, items):
        data = [(self.phi(it), it.gold) for it in items if it.gold >= 0]
        F = data[0][0].shape[1]
        th = np.zeros(F)
        for _ in range(30):
            g, H = self.lam * th, self.lam * np.eye(F)
            for X, y in data:
                z = X @ th
                p = np.exp(z - z.max())
                p /= p.sum()
                mu = p @ X
                g += mu - X[y]
                H += (X * p[:, None]).T @ X - np.outer(mu, mu)
            step = np.linalg.solve(H, g)
            th -= step
            if np.abs(step).max() < 1e-6:
                break
        self.theta = th
        return self

    def predict(self, it):
        z = self.phi(it) @ self.theta
        p = np.exp(z - z.max())
        return p / p.sum()
