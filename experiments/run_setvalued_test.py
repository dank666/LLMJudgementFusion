"""Test evaluation on the set-valued tasks (Tables 3 and 4, Figure 2).

Fusion models are fitted on the development split of each task and applied to its test split. Ensembles with
the frontier models use the items listed in data/processed/frontier_uids/.
"""

import json

import numpy as np

from common import ROOT
from fusion_sets import (Fusion, decode, direct_set, dawid_skene, fit_threshold, load, scores, set_plurality,
                         summary, unit_vote)
from stats import bootstrap
from views_io import call_index

CFG = json.loads((ROOT / "experiments" / "setvalued_config.json").read_text(encoding="utf-8"))
UIDS = ROOT / "data" / "processed" / "frontier_uids"
MODELS = [m for part in CFG["views"].values() for m in part["models"]]
FRONTIER = CFG["views"]["frontier"]["models"]


def ensemble_keys(items, ens):
    allowed = {(m, s) for part in CFG["ensembles"][ens] for m in CFG["views"][part]["models"]
               for s in CFG["views"][part]["strategies"]}
    return sorted({k for it in items for k in it.views if tuple(k.split("__")) in allowed})


def value(S, it, metric):
    return scores(S, it.gold)["em" if metric == "exact" else "jac"]


def cost_per_1000(items, keys, calls):
    return round(1000 * float(np.mean([sum(c[0] for k, c in calls.get(it.uid, {}).items() if k in keys)
                                       for it in items])), 3)


def evaluate(train, test, ens, decoder, metric):
    keys = ensemble_keys(train + test, ens)
    models = sorted({k.split("__")[0] for k in keys})
    rows, per_item = {}, {}

    def add(name, sets):
        rows[name] = summary([scores(S, it.gold) for S, it in zip(sets, test)])
        per_item[name] = [value(S, it, metric) for S, it in zip(sets, test)]

    fusion = Fusion(train, keys, decoder)
    for m in models:
        if f"{m}__direct" in keys:
            add(f"{m} direct", [direct_set(it, m) for it in test])
    add("set plurality", [set_plurality(it, keys, fusion.w) for it in test])
    add("unit vote", [unit_vote(it, keys, fusion.w) for it in test])
    t, rule = fit_threshold(train, keys)
    add("unit vote, tuned threshold", [rule(it, t) for it in test])
    add("Dawid-Skene", [decode(p, "em" if metric == "exact" else "jac") for p in dawid_skene(test, keys)])
    if len(models) > 1:
        for m in models:
            own = Fusion(train, [k for k in keys if k.startswith(f"{m}__")], decoder)
            add(f"{m} calibrated", [own.predict(it) for it in test])
    add("fusion", [fusion.predict(it) for it in test])

    rules = ["set plurality", "unit vote", "unit vote, tuned threshold", "Dawid-Skene"]
    best_rule = max(rules, key=lambda r: np.mean(per_item[r]))
    singles = [r for r in rows if r.endswith(" calibrated")] or [r for r in rows if r.endswith(" direct")]
    best_single = max(singles, key=lambda r: np.mean(per_item[r]))
    return keys, {"views": len(keys), "rows": rows,
                  "vs_best_rule": [best_rule, bootstrap(per_item["fusion"], per_item[best_rule])],
                  "vs_best_single": [best_single, bootstrap(per_item["fusion"], per_item[best_single])]}


def selection(items, extra):
    """Labels selected per item, precision, recall and Jaccard index of each model's direct answers and of the
    sets in `extra` (name -> one set per item)."""
    out = {"gold_labels_per_item": round(float(np.mean([len(it.gold) for it in items])), 2)}
    rows = {m: [(it, direct_set(it, m)) for it in items if f"{m}__direct" in it.views] for m in MODELS}
    rows.update({name: list(zip(items, sets)) for name, sets in extra.items()})
    for name, pairs in rows.items():
        r = [scores(S, it.gold) for it, S in pairs]
        tp, fp, fn = (sum(x[k] for x in r) for k in ("tp", "fp", "fn"))
        out[name] = {"n": len(pairs), "labels_per_item": round(float(np.mean([len(S) for _, S in pairs])), 2),
                     "precision": round(100 * tp / (tp + fp), 1), "recall": round(100 * tp / (tp + fn), 1),
                     **summary(r)}
    return out


def main():
    report = {"tasks": {}, "selection": {}, "frontier": {}}
    for task, spec in CFG["tasks"].items():
        metric, decoder = spec["metric"], spec["decoder"]
        dev, test = load(f"{task}_dev", MODELS), load(f"{task}_test", MODELS)
        fdev = set((UIDS / f"{task}_dev.txt").read_text().split())
        ftest = set((UIDS / f"{task}_test.txt").read_text().split())
        calls = call_index(f"{task}_test")
        report["tasks"][task] = {}
        for ens in CFG["ensembles"]:
            frontier = "frontier" in CFG["ensembles"][ens]
            tr = [it for it in dev if it.uid in fdev] if frontier else dev
            te = [it for it in test if it.uid in ftest] if frontier else test
            keys, res = evaluate(tr, te, ens, decoder, metric)
            report["tasks"][task][ens] = {"n": len(te), "cost_per_1000": cost_per_1000(te, keys, calls), **res}

        # small and inexpensive models against the frontier models on the frontier items
        sub = [it for it in test if it.uid in ftest]
        key = "exact" if metric == "exact" else "jaccard"
        row = {}
        for m in FRONTIER:
            row[m] = {key: summary([scores(direct_set(it, m), it.gold) for it in sub])[key],
                      "cost_per_1000": cost_per_1000(sub, [f"{m}__direct"], calls)}
        for ens in ("free", "free + inexpensive"):
            keys = ensemble_keys(dev + test, ens)
            fusion = Fusion(dev, keys, decoder)
            row[ens] = {key: summary([scores(fusion.predict(it), it.gold) for it in sub])[key],
                        "cost_per_1000": cost_per_1000(sub, keys, calls)}
        report["frontier"][task] = {"n": len(sub), **row}
        report["selection"][task] = selection(test, {"fusion": [fusion.predict(it) for it in test]})

    out = ROOT / "results" / "setvalued" / "test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for task, res in report["tasks"].items():
        for ens, r in res.items():
            print(task, ens, r["n"], r["rows"]["fusion"], r["vs_best_rule"], r["vs_best_single"])


if __name__ == "__main__":
    main()
