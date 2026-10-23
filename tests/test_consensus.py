import itertools
import random
from collections import Counter

from llmfusion.consensus import consensus, per_option_vote
from llmfusion.views import parse_letters

OPTS = ["A", "B", "C"]


def test_single_answer():
    vecs = {
        "m1-direct": {"A": "+", "B": "-", "C": "-"},
        "m2-direct": {"A": "+", "B": "-", "C": "-"},
        "m3-direct": {"A": "-", "B": "+", "C": "-"},
        "m1-elim": {"A": "?", "B": "?", "C": "-"},
    }
    c = consensus(vecs, OPTS, multiple=False)
    assert c.answer == ["A"]
    assert c.support == 0.5 and c.coalition == ["m1-direct", "m2-direct"]


def test_sets_need_agreement_on_every_unit():
    vecs = {
        "v1": {"A": "+", "B": "-", "C": "+"},
        "v2": {"A": "+", "B": "-", "C": "+"},
        "v3": {"A": "+", "B": "+", "C": "-"},
        "v4": {"A": "-", "B": "+", "C": "-"},
        "v5": {"A": "?", "B": "+", "C": "-"},
    }
    assert per_option_vote(vecs, OPTS, multiple=True, min_size=2) == ["A", "B"]
    c = consensus(vecs, OPTS, multiple=True, min_size=2)
    assert c.answer == ["A", "C"] and c.support == 0.4


def test_open_views_do_not_count():
    vecs = {"v1": {"A": "?", "B": "?", "C": "?"}, "v2": {"A": "+", "B": "-", "C": "-"}}
    c = consensus(vecs, OPTS, multiple=False)
    assert c.answer == ["A"] and c.support == 0.5 and c.coalition == ["v2"]


def test_consensus_is_plurality_of_decisive_views():
    for seed in range(300):
        rng = random.Random(seed)
        opts = list("ABCD")[: rng.randint(2, 4)]
        multiple = rng.random() < 0.5
        vecs = {}
        for i in range(rng.randint(1, 7)):
            acc = ({o for o in opts if rng.random() < 0.5} or {rng.choice(opts)}) if multiple else {rng.choice(opts)}
            vecs[f"v{i}"] = {o: ("+" if o in acc else "-") for o in opts}
        for i in range(rng.randint(0, 3)):
            vecs[f"open{i}"] = {o: rng.choice("-?") for o in opts} | {rng.choice(opts): "?"}
        decisive = [v for v in vecs.values() if "?" not in v.values()]
        c = consensus(vecs, opts, multiple)
        counts = Counter(tuple(sorted(o for o in opts if v[o] == "+")) for v in decisive)
        top = max(counts.values())
        assert abs(c.support - top / len(vecs)) < 1e-4
        assert tuple(c.answer) in {k for k, n in counts.items() if n == top}


def test_parse_letters():
    L = list("ABCD")
    assert parse_letters("B  \nC", L) == ["B", "C"]
    assert parse_letters("A, C", L) == ["A", "C"]
    assert parse_letters("Answer: B", L) == ["B"]
    assert parse_letters("The answer is C.", L) == ["C"]
    assert parse_letters("ABC", L) == ["A", "B", "C"]
    assert parse_letters("ACK", list("ABCDEFGHIJK")) == ["A", "C", "K"]


def test_jaccard_decision_is_a_prefix():
    from fusion_sets import decode

    def J(S, p):
        a = sum(p[u] for u in S)
        return a / (len(S) + sum(p.values()) - a)

    for seed in range(200):
        rng = random.Random(seed)
        p = {u: rng.random() for u in "ABCDEFG"[: rng.randint(1, 7)]}
        best = max(J(S, p) for r in range(1, len(p) + 1) for S in itertools.combinations(p, r))
        S = decode(p, "jac")
        assert abs(J(S, p) - best) < 1e-12
        order = sorted(p, key=p.get, reverse=True)
        assert S == frozenset(order[: len(S)])
