"""Ask the models for their views on a split and write results/views/<split>/<model>__<strategy>.jsonl.

Every call is cached in results/cache, so an interrupted run can be restarted without paying twice.
Example: python experiments/collect_views.py --split jecqa_dev --models Qwen/Qwen3.5-122B-A10B --strategies direct judge
"""

import argparse
import fcntl
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import FREE_MODELS, ROOT, SPLITS, load_split, make_client
from llmfusion.views import STRATEGIES, run_view

# CNY per million input / output tokens
PRICES = {"Qwen/Qwen3.5-122B-A10B": (0.8, 6.4), "deepseek-ai/DeepSeek-V4-Pro": (12.0, 24.0),
          "Pro/moonshotai/Kimi-K2.6": (6.5, 27.0)}
LEDGER = ROOT / "results" / "spend_ledger.json"


def spent(clients, model):
    pi, po = PRICES.get(model, (0.0, 0.0))
    tot = 0.0
    for c in clients:
        u = c.usage
        frac = (u.calls - u.cached) / max(u.calls, 1)
        tot += frac * (u.prompt_tokens * pi + u.completion_tokens * po) / 1e6
    return tot


def ledger(model, add=0.0):
    """Total spent on a model so far; with add > 0 the amount is added first."""
    with open(LEDGER.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
        if add > 0:
            data[model] = round(data.get(model, 0.0) + add, 4)
            LEDGER.write_text(json.dumps(data, indent=1))
        return data.get(model, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--models", nargs="+", default=FREE_MODELS)
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    ap.add_argument("--uid-file", help="only the questions listed in this file")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-cny", type=float, default=0.0, help="stop once a model's total spend exceeds this")
    args = ap.parse_args()

    questions = load_split(args.split)
    if args.uid_file:
        keep = set(Path(args.uid_file).read_text().split())
        questions = [q for q in questions if q.uid in keep]
    out_dir = ROOT / "results" / "views" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        jc = make_client(model, json_mode=True, max_tokens=256)
        tc = make_client(model, json_mode=False, max_tokens=16)
        recorded = 0.0
        for strategy in args.strategies:
            if args.budget_cny and ledger(model) + spent((jc, tc), model) - recorded > args.budget_cny:
                print(f"budget for {model} reached, stopping")
                break
            t0 = time.time()

            def one(q):
                try:
                    return {"uid": q.uid, **run_view(q, strategy, jc, tc)}
                except Exception as exc:
                    return {"uid": q.uid, "vector": {o: "?" for o in q.options}, "error": repr(exc)[:200]}

            with ThreadPoolExecutor(args.workers) as pool:
                rows = list(pool.map(one, questions))
            path = out_dir / f"{model.split('/')[-1]}__{strategy}.jsonl"
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            now = spent((jc, tc), model)
            ledger(model, now - recorded)
            recorded = now
            errors = sum("error" in r for r in rows)
            print(f"{args.split} {model} {strategy}: {len(rows)} rows, {errors} errors, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
