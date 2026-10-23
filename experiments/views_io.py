"""Reading the recorded views, their costs, and log-odds view weights."""

import json
import math
from collections import Counter, defaultdict

from common import ROOT, load_split

# CNY per million input / output tokens (SiliconFlow list prices, October 2026)
PRICES = {"Qwen3.5-122B-A10B": (0.8, 6.4), "DeepSeek-V4-Pro": (12.0, 24.0), "Kimi-K2.6": (6.5, 27.0)}


def view_files(split):
    for path in sorted((ROOT / "results" / "views" / split).glob("*.jsonl")):
        if "." not in path.stem.split("__")[-1]:
            yield path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_views(split, include_confidence=False):
    """{"model__strategy": {uid: {unit: "+", "-" or "?"}}}; failed calls count as no view."""
    views = {}
    for path in view_files(split):
        if path.stem.endswith("__confidence") and not include_confidence:
            continue
        views[path.stem] = {r["uid"]: ({} if "error" in r else r["vector"]) for r in read_rows(path)}
    return views


def load_confidence(split, model):
    path = ROOT / "results" / "views" / split / f"{model}__confidence.jsonl"
    return {r["uid"]: r.get("confidence") for r in read_rows(path)}


def call_index(split):
    """{uid: {"model__strategy": [cny, calls, tokens]}} over the recorded calls."""
    idx = defaultdict(dict)
    for path in view_files(split):
        model = path.stem.split("__")[0]
        pi, po = PRICES.get(model, (0.0, 0.0))
        for r in read_rows(path):
            pt, ct = r.get("prompt_tokens", 0), r.get("completion_tokens", 0)
            if not pt and not ct:
                continue
            c = idx[r["uid"]].setdefault(path.stem, [0.0, 0, 0])
            c[0] += (pt * pi + ct * po) / 1e6
            c[1] += 1
            c[2] += pt + ct
    return idx


def answer_of(vec, options, multiple):
    acc = sorted(o for o in options if vec.get(o) == "+")
    if not multiple:
        return acc[:1] if len(acc) == 1 else []
    return acc


def view_correct(q, name, vec):
    opts = sorted(q.options)
    strategy = name.split("__")[-1]
    if strategy == "eliminate":
        return [o for o in opts if vec.get(o) != "-"] == sorted(q.answer)
    if strategy == "statements":
        return None
    return answer_of(vec, opts, q.qtype == "multiple") == sorted(q.answer)


def log_odds_weights(splits):
    """Weight of a view: log-odds of its accuracy minus that of guessing among four options."""
    ok, tot = Counter(), Counter()
    for sp in splits:
        views = load_views(sp)
        for q in load_split(sp):
            for k, v in views.items():
                vec = v.get(q.uid)
                if not vec:
                    continue
                c = view_correct(q, k, vec)
                if c is None:
                    continue
                ok[k] += c
                tot[k] += 1
    w = {}
    for k in tot:
        a = (ok[k] + 1) / (tot[k] + 2)
        w[k] = max(0.05, math.log(a / (1 - a)) - math.log(1 / 3))
    return w
