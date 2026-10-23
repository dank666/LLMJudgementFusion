"""Multi-subject exam benchmark (all subjects except geography, which has its own sets).

Stratified samples per source, disjoint dev/test, fixed seed:
  MMLU          test 10 per subject; dev 5 per subject from the validation split
  EXAMS         test 600 / dev 300, stratified by (language, subject)
  ArabicMMLU    test 400 / dev 150, stratified by subject
  IndoMMLU      test 400 / dev 150, stratified by subject
  TurkishMMLU   test 60 per subject; dev = the dev files plus 20 per subject from the remaining pool
  GAOKAO-Bench  politics, history, biology, chemistry: 80% test / 20% dev; physics: all test
Outputs data/processed/multi_{dev,test}.jsonl and multi_report.json.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmfusion.data import (  # noqa: E402
    load_arabicmmlu_all,
    load_exams_all,
    load_gaokao_subject,
    load_indommlu_all,
    load_mmlu_all,
    load_turkishmmlu_subject,
    write_jsonl,
)

RAW = ROOT / "data" / "raw" / "multi"
OUT = ROOT / "data" / "processed"
SEED = 2026
GEO = ("geograph", "geografi")


def is_geo(q) -> bool:
    return any(g in str(q.meta.get("subject", "")).lower() for g in GEO)


def stratified(qs, key, n_total=None, per_stratum=None, rng=None):
    groups = defaultdict(list)
    for q in qs:
        groups[key(q)].append(q)
    for g in groups.values():
        rng.shuffle(g)
    if per_stratum is not None:
        return [q for k in sorted(groups) for q in groups[k][:per_stratum]]
    # proportional allocation with at least one item per stratum
    total = len(qs)
    picked = []
    for k in sorted(groups):
        m = max(1, round(n_total * len(groups[k]) / total))
        picked += groups[k][:m]
    rng.shuffle(picked)
    return picked[:n_total]


def split_pool(qs, key, n_test, n_dev, rng):
    test = stratified(qs, key, n_total=n_test, rng=rng)
    tid = {q.uid for q in test}
    rest = [q for q in qs if q.uid not in tid]
    dev = stratified(rest, key, n_total=n_dev, rng=rng)
    return dev, test


def main() -> None:
    rng = random.Random(SEED)
    dev, test = [], []

    mmlu_t = [q for q in load_mmlu_all(RAW / "mmlu_all_test.parquet") if not is_geo(q)]
    mmlu_v = [q for q in load_mmlu_all(RAW / "mmlu_all_validation.parquet") if not is_geo(q)]
    test += stratified(mmlu_t, lambda q: q.meta["subject"], per_stratum=10, rng=rng)
    dev += stratified(mmlu_v, lambda q: q.meta["subject"], per_stratum=5, rng=rng)

    exams_t = [q for q in load_exams_all(ROOT / "data" / "raw" / "intl" / "exams_multilingual_test.parquet") if not is_geo(q)]
    exams_v = [q for q in load_exams_all(ROOT / "data" / "raw" / "intl" / "exams_multilingual_validation.parquet") if not is_geo(q)]
    key = lambda q: (q.meta["language"], q.meta["subject"])  # noqa: E731
    test += stratified(exams_t, key, n_total=600, rng=rng)
    dev += stratified(exams_v, key, n_total=300, rng=rng)

    for loader, path, n_test, n_dev in (
        (load_arabicmmlu_all, RAW / "arabic" / "All_test.csv", 400, 150),
        (load_indommlu_all, ROOT / "data" / "raw" / "intl" / "indommlu.csv", 400, 150),
    ):
        pool = [q for q in loader(path) if not is_geo(q)]
        d, t = split_pool(pool, lambda q: q.meta["subject"], n_test, n_dev, rng)
        dev += d
        test += t

    for f in sorted((RAW / "tr").glob("test_TurkishMMLU_*.json")):
        subject = f.stem.replace("test_TurkishMMLU_", "")
        if subject.lower().startswith("geograph"):
            continue
        pool = load_turkishmmlu_subject(f, subject, "test")
        rng.shuffle(pool)
        test += pool[:60]
        dev += pool[60:80]
        devf = RAW / "tr" / f"dev_TurkishMMLU_{subject}.json"
        if devf.exists():
            dev += load_turkishmmlu_subject(devf, subject, "dev")

    for subject, fname in (("Politics", "Political_Science"), ("History", "History"), ("Biology", "Biology"),
                           ("Chemistry", "Chemistry"), ("Physics", "Physics")):
        qs, _ = load_gaokao_subject(RAW / f"gaokao_2010-2022_{fname}_MCQs.json", subject)
        rng.shuffle(qs)
        if subject == "Physics":
            test += qs
            continue
        k = round(0.2 * len(qs))
        dev += qs[:k]
        test += qs[k:]

    assert not {q.uid for q in dev} & {q.uid for q in test}
    write_jsonl(dev, OUT / "multi_dev.jsonl")
    write_jsonl(test, OUT / "multi_test.jsonl")

    def summary(qs):
        return {
            "n": len(qs),
            "by_dataset": dict(Counter(q.dataset for q in qs)),
            "languages": len({q.meta.get("language") for q in qs}),
            "subjects": len({(q.dataset, q.meta.get("subject")) for q in qs}),
            "multiple_select": sum(q.qtype == "multiple" for q in qs),
            "statement_combo": sum(q.is_statement_combo for q in qs),
            "statement_combo_by_dataset": dict(Counter(q.dataset for q in qs if q.is_statement_combo)),
        }

    report = {"dev": summary(dev), "test": summary(test)}
    (OUT / "multi_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
