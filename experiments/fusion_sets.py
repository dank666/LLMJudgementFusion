"""Fusion of the views on set-valued answers, and the rules it is compared with.

A logistic model turns the evidence about each label (option) into a probability. The answer is the set with
the smallest expected loss: for exact match, the best candidate of a conditional logit over candidate sets; for
the Jaccard index, the best prefix of the labels ranked by probability.
"""

import math
from collections import defaultdict

import numpy as np

from common import load_split
from llmfusion.consensus import per_option_vote
from llmfusion.fca import FormalContext
from views_io import load_confidence, load_views

STRATEGIES = ("direct", "judge", "eliminate", "confidence")


class Item:
    def __init__(self, q, views, conf):
        self.q, self.uid = q, q.uid
        self.units = sorted(q.options)
        self.gold = frozenset(q.answer)
        self.views = {k: v for k, v in views.items() if v}
        self.conf = conf


def load(split, models):
    views = load_views(split, include_confidence=True)
    conf = {k: load_confidence(split, k.split("__")[0]) for k in views if k.endswith("__confidence")}
    items = []
    for q in load_split(split):
        vs = {k: x.get(q.uid) for k, x in views.items()
              if k.split("__")[0] in models and k.split("__")[1] in STRATEGIES}
        cf = {k: c[q.uid] / 100 for k, c in conf.items() if c.get(q.uid) is not None}
        items.append(Item(q, vs, cf))
    return items


def scores(pred, gold):
    inter, union = len(pred & gold), len(pred | gold)
    return {"em": float(pred == gold), "jac": inter / union if union else 1.0, "tp": inter,
            "fp": len(pred - gold), "fn": len(gold - pred)}


def summary(rows):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    return {"exact": round(100 * np.mean([r["em"] for r in rows]), 2),
            "jaccard": round(100 * np.mean([r["jac"] for r in rows]), 2),
            "micro_f1": round(100 * 2 * tp / max(1, 2 * tp + fp + fn), 2)}


def direct_set(it, model):
    v = it.views.get(f"{model}__direct") or {}
    return frozenset(u for u, x in v.items() if x == "+")


def weights(train, keys):
    """Log-odds of each view's accuracy on the labels it decides."""
    w = {}
    for k in keys:
        ok = tot = 0
        for it in train:
            for u, x in (it.views.get(k) or {}).items():
                if x in "+-":
                    ok += (x == "+") == (u in it.gold)
                    tot += 1
        if tot:
            a = (ok + 1) / (tot + 2)
            w[k] = max(0.05, math.log(a / (1 - a)))
    return w


def set_plurality(it, keys, w):
    votes = defaultdict(float)
    for k in keys:
        v = it.views.get(k)
        if v and all(x in "+-" for x in v.values()):
            S = frozenset(u for u, x in v.items() if x == "+")
            if S:
                votes[S] += w.get(k, 1.0)
    if not votes:
        return frozenset(per_option_vote({k: it.views[k] for k in keys if k in it.views}, it.units, True, w, 1))
    return max(votes, key=lambda S: (votes[S], -len(S)))


def unit_vote(it, keys, w):
    vecs = {k: it.views[k] for k in keys if k in it.views}
    return frozenset(per_option_vote(vecs, it.units, True, w, 1))


def share(it, keys, u):
    a = r = 0
    for k in keys:
        x = (it.views.get(k) or {}).get(u)
        a += x == "+"
        r += x == "-"
    return a / (a + r) if a + r else 0.5


def fit_threshold(train, keys, grid=np.linspace(0.05, 0.95, 19)):
    def pred(it, t):
        s = {u: share(it, keys, u) for u in it.units}
        return frozenset(u for u in s if s[u] > t) or frozenset([max(s, key=s.get)])
    return max(grid, key=lambda t: np.mean([scores(pred(it, t), it.gold)["jac"] for it in train])), pred


def dawid_skene(items, keys, iters=30):
    """One-coin Dawid-Skene: a sensitivity and a specificity per view, fitted without labels."""
    T = {(n, u): share(it, keys, u) for n, it in enumerate(items) for u in it.units}
    for _ in range(iters):
        se, sp = {}, {}
        for k in keys:
            tp = fn = tn = fp = 1e-3
            for n, it in enumerate(items):
                v = it.views.get(k)
                if not v:
                    continue
                for u in it.units:
                    x, t = v.get(u), T[(n, u)]
                    if x == "+":
                        tp, fp = tp + t, fp + 1 - t
                    elif x == "-":
                        fn, tn = fn + t, tn + 1 - t
            se[k], sp[k] = tp / (tp + fn), tn / (tn + fp)
        prior = float(np.mean(list(T.values())))
        for n, it in enumerate(items):
            for u in it.units:
                l1, l0 = math.log(prior), math.log(1 - prior)
                for k in keys:
                    x = (it.views.get(k) or {}).get(u)
                    if x == "+":
                        l1, l0 = l1 + math.log(se[k]), l0 + math.log(1 - sp[k])
                    elif x == "-":
                        l1, l0 = l1 + math.log(1 - se[k]), l0 + math.log(sp[k])
                T[(n, u)] = 1 / (1 + math.exp(max(-50.0, min(50.0, l0 - l1))))
    return [{u: T[(n, u)] for u in it.units} for n, it in enumerate(items)]


def unit_features(it, keys, u):
    f = []
    for g in sorted({k.split("__")[0] for k in keys}):
        vs = [it.views[k] for k in keys if k.split("__")[0] == g and k in it.views and not k.endswith("__confidence")]
        n = max(1, len(vs))
        f += [sum(v.get(u) == "+" for v in vs) / n, sum(v.get(u) == "-" for v in vs) / n,
              sum(v.get(u, "?") == "?" for v in vs) / n]
        ck = f"{g}__confidence"
        if ck in keys:
            cv = it.views.get(ck, {})
            f += [float(cv.get(u) == "+") * it.conf.get(ck, 0.0), float(cv.get(u) == "+")]
    for s in ("direct", "judge", "eliminate"):
        vs = [it.views[k] for k in keys if k.endswith(f"__{s}") and k in it.views]
        n = max(1, len(vs))
        f += [sum(v.get(u) == "+" for v in vs) / n - sum(v.get(u) == "-" for v in vs) / n]
    return f


class Logistic:
    def __init__(self, X, y, lam=1.0, iters=60):
        X = np.asarray(X, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        y = np.asarray(y, float)
        b = np.zeros(Z.shape[1])
        R = lam * np.eye(Z.shape[1])
        R[0, 0] = 1e-6
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30)))
            step = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + R, Z.T @ (p - y) + R @ b)
            b -= step
            if np.abs(step).max() < 1e-8:
                break
        self.b = b

    def __call__(self, X):
        X = np.asarray(X, float)
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        return 1 / (1 + np.exp(-np.clip(Z @ self.b, -30, 30)))


def fit_unit_model(train, keys):
    X = [unit_features(it, keys, u) for it in train for u in it.units]
    y = [float(u in it.gold) for it in train for u in it.units]
    return Logistic(X, y)


def decode(p, metric):
    """Exact match: labels with p > 1/2. Jaccard: the prefix of the ranking with the largest expected Jaccard."""
    order = sorted(p, key=p.get, reverse=True)
    if metric == "em":
        S = [u for u in order if p[u] > 0.5]
        return frozenset(S or order[:1])
    tot = sum(p.values())
    best, bestv, acc = order[:1], -1.0, 0.0
    for k in range(1, len(order) + 1):
        acc += p[order[k - 1]]
        v = acc / (k + tot - acc)
        if v > bestv:
            best, bestv = order[:k], v
    return frozenset(best)


def candidate_family(it, keys, p):
    """The closed sets of the context, the sets proposed by the views and the first prefixes of the ranking."""
    vecs = {k: it.views[k] for k in keys if k in it.views}
    pos = [(k, u) for k, v in vecs.items() for u, x in v.items() if x == "+"]
    neg = [(k, u) for k, v in vecs.items() for u, x in v.items() if x == "-"]
    K = FormalContext(list(vecs), it.units, pos, neg)
    C = {frozenset(K.attrs(i)) for i in K.intents()}
    C |= {frozenset(u for u, x in v.items() if x == "+") for v in vecs.values()}
    order = sorted(p, key=p.get, reverse=True)
    C |= {frozenset(order[:k]) for k in range(1, min(len(order), 6) + 1)}
    return sorted((S for S in C if S), key=lambda S: (len(S), sorted(S)))


def set_features(it, keys, S, p, w):
    vecs = {k: it.views[k] for k in keys if k in it.views}
    W = sum(w.get(k, 1.0) for k in vecs) or 1.0
    rest = [u for u in it.units if u not in S]
    certain = sum(w.get(k, 1.0) for k, v in vecs.items()
                  if all(v.get(u) == "+" for u in S) and all(v.get(u) == "-" for u in rest)) / W
    possible = sum(w.get(k, 1.0) for k, v in vecs.items()
                   if not any(v.get(u) == "-" for u in S) and not any(v.get(u) == "+" for u in rest)) / W
    lp = sum(math.log(max(p[u], 1e-6)) for u in S) + sum(math.log(max(1 - p[u], 1e-6)) for u in rest)
    size = len(S)
    return [certain, possible, lp, min(p[u] for u in S), max([p[u] for u in rest] or [0.0]),
            size == 1, size == 2, size == 3, size >= 4]


class SetModel:
    """Conditional logit over the candidate sets of each item."""

    def __init__(self, data, lam=1.0, iters=40):
        F = len(data[0][0][0])
        th = np.zeros(F)
        for _ in range(iters):
            g, H = lam * th, lam * np.eye(F)
            for X, y in data:
                X = np.asarray(X, float)
                z = X @ th
                q = np.exp(z - z.max())
                q /= q.sum()
                mu = q @ X
                g += mu - X[y]
                H += (X * q[:, None]).T @ X - np.outer(mu, mu)
            step = np.linalg.solve(H, g)
            th -= step
            if np.abs(step).max() < 1e-7:
                break
        self.th = th


class Fusion:
    """The learned fusion of one ensemble, fitted on labelled development items."""

    def __init__(self, train, keys, decoder):
        self.keys, self.decoder = keys, decoder
        self.w = weights(train, keys)
        self.um = fit_unit_model(train, keys)
        self.sm = None
        if decoder == "set":
            data = []
            for it in train:
                p = self.p(it)
                C = candidate_family(it, keys, p)
                if it.gold in C:
                    data.append(([set_features(it, keys, S, p, self.w) for S in C], C.index(it.gold)))
            self.sm = SetModel(data)

    def p(self, it):
        return dict(zip(it.units, self.um([unit_features(it, self.keys, u) for u in it.units])))

    def predict(self, it):
        p = self.p(it)
        if self.decoder == "set":
            C = candidate_family(it, self.keys, p)
            z = np.asarray([set_features(it, self.keys, S, p, self.w) for S in C], float) @ self.sm.th
            q = np.exp(z - z.max())
            return C[int(np.argmax(q / q.sum()))]
        return decode(p, "jac")
