"""Paths, splits and model clients shared by the experiment scripts."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llmfusion.data import read_jsonl  # noqa: E402
from llmfusion.llm import LLMClient, LLMConfig  # noqa: E402

FREE_MODELS = ["Qwen/Qwen3.5-4B", "Qwen/Qwen3-8B", "THUDM/GLM-4-9B-0414"]
SETVALUED = ["jecqa", "tweet_emotion", "tweet_topic"]
SPLITS = ["intl_dev", "intl_test", "gaokao", "multi_dev", "multi_test"]
SPLITS += [f"{t}_{p}" for t in SETVALUED for p in ("dev", "test")]
EXCLUDED_PATH = ROOT / "data" / "processed" / "excluded.json"


def load_split(name, exclude=True):
    processed = ROOT / "data" / "processed"
    if name == "gaokao":
        qs = read_jsonl(processed / "gaokao_geo.jsonl")
    elif name in SPLITS:
        qs = read_jsonl(processed / f"{name}.jsonl")
    else:
        raise ValueError(f"unknown split {name}")
    if exclude and EXCLUDED_PATH.exists():
        dropped = json.loads(EXCLUDED_PATH.read_text(encoding="utf-8"))
        qs = [q for q in qs if q.uid not in dropped]
    return qs


def make_client(model, json_mode, max_tokens=1024, cache="views"):
    # thinking is switched off for the hybrid reasoning models
    hybrid = ("Qwen/Qwen3", "deepseek-ai/DeepSeek-V4", "Pro/moonshotai/Kimi")
    extra = {"enable_thinking": False} if model.startswith(hybrid) else None
    cfg = LLMConfig.from_env("BASELINE", model=model, extra_body=extra)
    cfg.json_mode, cfg.temperature, cfg.max_tokens = json_mode, 0.0, max_tokens
    return LLMClient(cfg, ROOT / "results" / "cache" / cache)
