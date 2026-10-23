"""Build the Chinese college entrance examination geography questions (gaokao split).

Output: data/processed/gaokao_geo.jsonl, the AGIEval gaokao-geography questions and the GAOKAO-Bench geography
questions that are not already in AGIEval.
Usage: python experiments/prepare_gaokao.py
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmfusion.data import load_agieval, load_gaokao_bench, normalize, write_jsonl  # noqa: E402


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _options_key(q) -> str:
    return normalize("".join(q.options[k] for k in sorted(q.options)))


def find_duplicates(gaokao, agieval) -> dict[str, str]:
    """Map GAOKAO-Bench uid -> AGIEval uid for items present in both."""
    dup = {}
    for q in gaokao:
        best, best_score = None, 0.0
        for p in agieval:
            so = _sim(_options_key(q), _options_key(p))
            if so < 0.85:
                continue
            stem_tail = _sim(normalize(q.stem)[-30:], normalize(p.stem)[-30:])
            head = _sim(normalize(q.full_text)[:60], normalize(p.full_text)[:60])
            score = min(so, max(stem_tail, head))
            if score > best_score:
                best, best_score = p, score
        if best is not None and best_score > 0.6:
            if best.answer != q.answer:
                raise ValueError(f"answer mismatch between duplicates {q.uid} / {best.uid}")
            dup[q.uid] = best.uid
    return dup


def main() -> None:
    raw = ROOT / "data" / "raw"
    agieval = load_agieval(raw / "public" / "agieval_gaokao_geography.jsonl")
    gaokao, failures = load_gaokao_bench(raw / "public" / "gaokao_bench_2010_2022_geography_mcqs.json")
    if failures:
        raise SystemExit(f"unparsed GAOKAO-Bench groups: {failures}")
    dup = find_duplicates(gaokao, agieval)
    public = agieval + [q for q in gaokao if q.uid not in dup]
    write_jsonl(public, ROOT / "data" / "processed" / "gaokao_geo.jsonl")
    print(f"{len(public)} questions ({len(agieval)} AGIEval, {len(public) - len(agieval)} GAOKAO-Bench)")


if __name__ == "__main__":
    main()
