"""Build the international geography exam benchmark.

dev  (tuning only): MMLU-HSG validation+dev, EXAMS-Geography validation,
                    ArabicMMLU Geography (middle school), IndoMMLU Geografi (90 sampled),
                    TurkishMMLU Geography dev
test (final only):  MMLU-HSG test, EXAMS-Geography test, ArabicMMLU Geography (high school),
                    IndoMMLU Geografi (remaining), TurkishMMLU Geography test

Public benchmarks keep their original option order so results stay comparable
with published numbers. Outputs: data/processed/intl_{dev,test}.jsonl, intl_report.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmfusion.data import (  # noqa: E402
    load_arabicmmlu_geography,
    load_exams_geography,
    load_indommlu_geography,
    load_mmlu_geography,
    load_turkishmmlu_geography,
    write_jsonl,
)

RAW = ROOT / "data" / "raw" / "intl"
OUT = ROOT / "data" / "processed"
SEED = 2026
INDO_DEV = 90


def main() -> None:
    indo, skipped = load_indommlu_geography(RAW / "indommlu.csv")
    rng = random.Random(SEED)
    uids = sorted(q.uid for q in indo)
    rng.shuffle(uids)
    indo_dev = set(uids[:INDO_DEV])

    dev = (
        load_mmlu_geography(RAW / "mmlu_hsg_validation.parquet")
        + load_mmlu_geography(RAW / "mmlu_hsg_dev.parquet")
        + load_exams_geography(RAW / "exams_multilingual_validation.parquet")
        + load_arabicmmlu_geography(RAW / "arabicmmlu_geo_Middle_School_test.csv")
        + [q for q in indo if q.uid in indo_dev]
        + load_turkishmmlu_geography(RAW / "turkishmmlu_geo_dev.json")
    )
    test = (
        load_mmlu_geography(RAW / "mmlu_hsg_test.parquet")
        + load_exams_geography(RAW / "exams_multilingual_test.parquet")
        + load_arabicmmlu_geography(RAW / "arabicmmlu_geo_High_School_test.csv")
        + [q for q in indo if q.uid not in indo_dev]
        + load_turkishmmlu_geography(RAW / "turkishmmlu_geo_test.json")
    )
    assert not {q.uid for q in dev} & {q.uid for q in test}

    write_jsonl(dev, OUT / "intl_dev.jsonl")
    write_jsonl(test, OUT / "intl_test.jsonl")

    def summary(qs):
        return {
            "n": len(qs),
            "by_dataset": dict(Counter(q.dataset for q in qs)),
            "by_language": dict(Counter(q.meta["language"] for q in qs)),
            "n_options": dict(Counter(len(q.options) for q in qs)),
            "statement_combo": sum(q.is_statement_combo for q in qs),
            "majority_label_acc": {
                d: round(100 * max(Counter(q.answer[0] for q in qs if q.dataset == d).values())
                         / sum(1 for q in qs if q.dataset == d), 2)
                for d in sorted({q.dataset for q in qs})
            },
        }

    report = {"dev": summary(dev), "test": summary(test), "indommlu_skipped": skipped}
    (OUT / "intl_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
