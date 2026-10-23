"""Test evaluation on the examination questions (Table 2 and the numbers in Section 4.2)."""

import json

import numpy as np

from common import ROOT, load_split
from fusion_single import CHEAP, FREE, FRONT, STRATEGIES, Stacker, load_items, plurality
from llmfusion.consensus import solve_item
from stats import mcnemar
from views_io import load_views, log_odds_weights

DEV = ["intl_dev", "multi_dev"]
TEST = ["intl_test", "gaokao", "multi_test"]
LARGE = FRONT + [CHEAP]


def cost(items, keys):
    return np.array([sum(it.calls.get(k, [0.0])[0] for k in keys) for it in items])


def correct(it, key):
    c = it.compat.get(key) or ()
    return len(c) == 1 and c[0] == it.gold


def consensus_accuracy(strategies, w):
    """Weighted consensus of the three large models using the given ways of asking."""
    ok = n = 0
    for sp in TEST:
        views = load_views(sp)
        for q in load_split(sp):
            vecs = {k: v[q.uid] for k, v in views.items()
                    if k.split("__")[0] in LARGE and k.split("__")[1] in strategies and v.get(q.uid)}
            ok += solve_item(q, vecs, w, 1).answer == sorted(q.answer)
            n += 1
    return round(100 * ok / n, 2)


def main():
    dev, test = load_items(DEV), load_items(TEST)
    n = len(test)
    gold = np.array([it.gold for it in test])
    direct = [f"{m}__direct" for m in LARGE]
    context = [f"{m}__{s}" for m in FREE for s in STRATEGIES] + [f"{CHEAP}__direct"] + \
              [f"{m}__{s}" for m in FRONT for s in STRATEGIES]

    vote = np.array([int(np.argmax(plurality(it, direct, {}))) for it in test]) == gold
    stacker = Stacker(context).fit(dev)
    fused = np.array([int(np.argmax(stacker.predict(it))) for it in test]) == gold
    best = np.array([correct(it, f"{CHEAP}__direct") for it in test])

    report = {"n": n, "single": {}}
    for m in FREE + [CHEAP] + FRONT:
        report["single"][m] = {"accuracy": round(100 * np.mean([correct(it, f"{m}__direct") for it in test]), 2),
                               "cost_per_1000": round(1000 * cost(test, [f"{m}__direct"]).mean(), 3)}
    report["vote"] = {"accuracy": round(100 * vote.mean(), 2), "cost_per_1000": round(1000 * cost(test, direct).mean(), 3)}
    paid = [f"{CHEAP}__direct"] + [f"{m}__{s}" for m in FRONT for s in STRATEGIES]
    report["fusion"] = {"accuracy": round(100 * fused.mean(), 2), "cost_per_1000": round(1000 * cost(test, paid).mean(), 3)}
    report["vote_vs_qwen122b"] = mcnemar(list(vote), list(best))
    report["fusion_vs_vote"] = mcnemar(list(fused), list(vote))
    w = log_odds_weights(DEV)
    report["consensus_direct"] = consensus_accuracy(("direct",), w)
    report["consensus_direct_judge_eliminate"] = consensus_accuracy(("direct", "judge", "eliminate"), w)

    out = ROOT / "results" / "singlechoice" / "test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
