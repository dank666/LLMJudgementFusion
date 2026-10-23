"""Consensus of the views on one question.

The views that agree with a candidate answer S accept every unit of S and reject the others; their weight is
the support of S. On a single answer this is plurality voting over the views that decide every unit.
"""

from dataclasses import dataclass, field
from itertools import combinations

from .fca import FormalContext


@dataclass
class Consensus:
    answer: list[str]
    support: float
    margin: float
    option_net: dict[str, float]
    candidates: list[tuple[list[str], float]] = field(default_factory=list)
    coalition: list[str] = field(default_factory=list)


def build_context(vectors: dict[str, dict[str, str]], options: list[str]) -> FormalContext:
    views = list(vectors)
    pos = [(v, o) for v in views for o in options if vectors[v].get(o) == "+"]
    neg = [(v, o) for v in views for o in options if vectors[v].get(o) == "-"]
    return FormalContext(views, options, pos, neg)


def option_net(vectors, options, weights) -> dict[str, float]:
    W = sum(weights.get(v, 1.0) for v in vectors) or 1.0
    out = {}
    for o in options:
        s = 0.0
        for v, vec in vectors.items():
            s += weights.get(v, 1.0) * (1 if vec.get(o) == "+" else -1 if vec.get(o) == "-" else 0)
        out[o] = s / W
    return out


def candidate_sets(K: FormalContext, options: list[str], multiple: bool, min_size: int) -> list[frozenset]:
    if not multiple:
        return [frozenset([o]) for o in options]
    closed = {K.attrs(i) for i in K.intents()}
    cands = {S for S in closed if len(S) >= min_size}
    if len(options) <= 5:
        cands |= {frozenset(c) for r in range(min_size, len(options) + 1) for c in combinations(options, r)}
    return sorted(cands, key=lambda S: (len(S), sorted(S)))


def consensus(
    vectors: dict[str, dict[str, str]],
    options: list[str],
    multiple: bool,
    weights: dict[str, float] | None = None,
    min_size: int = 1,
    candidates: list[frozenset] | None = None,
) -> Consensus:
    """`candidates` restricts the answers, e.g. to the statement sets of a statement-combination item."""
    weights = weights or {}
    K = build_context(vectors, options)
    W = sum(weights.get(v, 1.0) for v in K.objects) or 1.0
    net = option_net(vectors, options, weights)
    scored = []
    for S in candidates or candidate_sets(K, options, multiple, min_size):
        rest = [o for o in options if o not in S]
        X = K.down(K.attr_mask(S)) & K.neg_down(K.attr_mask(rest))
        sup = sum(weights.get(v, 1.0) for v in K.objs(X)) / W
        tie = sum(net[o] for o in S) - sum(net[o] for o in rest)
        scored.append((sup, tie, S, K.objs(X)))
    scored.sort(key=lambda t: (t[0], t[1], -len(t[2])), reverse=True)
    best = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    return Consensus(
        answer=sorted(best[2]),
        support=round(best[0], 4),
        margin=round(best[0] - runner, 4),
        option_net={o: round(x, 4) for o, x in net.items()},
        candidates=[(sorted(S), round(s, 4)) for s, _, S, _ in scored[:5]],
        coalition=sorted(best[3]),
    )


def per_option_vote(vectors, options, multiple, weights=None, min_size=1) -> list[str]:
    net = option_net(vectors, options, weights or {})
    ranked = sorted(options, key=lambda o: net[o], reverse=True)
    if not multiple:
        return [ranked[0]]
    ans = [o for o in options if net[o] > 0]
    for o in ranked:
        if len(ans) >= min_size:
            break
        if o not in ans:
            ans.append(o)
    return sorted(ans)


def solve_item(q, views: dict[str, dict[str, str]], weights=None, min_multi: int = 2) -> Consensus:
    from .views import unit_vector

    if q.is_statement_combo:
        units = list(q.statements)
        vecs = {}
        for name, vec in views.items():
            uv = unit_vector(q, name.split("__")[-1], vec)
            if uv:
                vecs[name] = uv
        cands = [frozenset(q.option_statements[o]) for o in sorted(q.options)]
        c = consensus(vecs, units, True, weights, 1, candidates=cands)
        S = frozenset(c.answer)
        best = min(sorted(q.options), key=lambda o: len(S ^ frozenset(q.option_statements[o])))
        c.answer = [best]
        return c
    opts = sorted(q.options)
    vecs = {name: vec for name, vec in views.items() if vec}
    multi = q.qtype == "multiple"
    return consensus(vecs, opts, multi, weights, min_multi if multi else 1)
