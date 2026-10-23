import random

import pytest

from llmfusion.fca import FormalContext


def test_operators():
    K = FormalContext(["g1", "g2", "g3"], ["a", "b", "c"],
                      [("g1", "a"), ("g1", "b"), ("g2", "a"), ("g3", "c")],
                      [("g1", "c"), ("g2", "c"), ("g3", "a")])
    assert K.objs(K.down(K.attr_mask(["a"]))) == {"g1", "g2"}
    assert K.objs(K.neg_down(K.attr_mask(["c"]))) == {"g1", "g2"}
    assert K.objs(K.neg_down(K.attr_mask(["b"]))) == set()
    assert K.attrs(K.up(K.obj_mask(["g1", "g2"]))) == {"a"}


def test_intents_are_intersections_of_rows():
    for seed in range(50):
        rng = random.Random(seed)
        objs, attrs = [f"g{i}" for i in range(6)], list("abcde")
        K = FormalContext(objs, attrs, [(g, a) for g in objs for a in attrs if rng.random() < 0.4])
        for mask in K.intents():
            assert K.up(K.down(mask)) == mask


def test_contradictory_pairs_are_rejected():
    with pytest.raises(ValueError):
        FormalContext(["g"], ["a"], [("g", "a")], [("g", "a")])
