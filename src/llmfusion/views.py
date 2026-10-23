"""Asking a model for its view of a question.

direct     the chosen options are accepted, the others rejected
judge      each option is called correct, incorrect or unsure
eliminate  the options listed as certainly wrong are rejected, the others left open
confidence the answer together with a stated confidence
statements each statement of a statement-combination item is called true, false or unsure
"""

import re

from .data import Question, split_statements
from .llm import LLMClient, parse_json
from .prompts import (
    confidence_messages,
    direct_messages,
    eliminate_messages,
    judge_all_messages,
    judge_statements_messages,
)

STRATEGIES = ("direct", "judge", "eliminate", "statements")
_VERDICT = {"correct": "+", "incorrect": "-", "true": "+", "false": "-", "yes": "+", "no": "-"}


def parse_letters(text: str | None, labels: list[str]) -> list[str]:
    """Option letters in a short answer such as "B", "AC", "A, C", "A\nC" or "Answer: B"."""
    if not text:
        return []
    text = text.strip().upper()
    cls = "A-H" if all(c <= "H" for c in labels) else "".join(sorted(labels))
    if re.fullmatch(rf"[{cls}]+", re.sub(r"[\s,，、;；/&+]", "", text)):
        found = [c for c in text if c in labels]
    else:
        head = re.split(r"\n", text)[0]
        found = [c for c in re.findall(rf"(?<![A-Z])([{cls}])(?![A-Z])", head) if c in labels]
    return sorted(dict.fromkeys(found))


def run_view(q: Question, strategy: str, json_client: LLMClient, text_client: LLMClient) -> dict:
    labels = sorted(q.options)
    multiple = q.qtype == "multiple"
    sysmsg = {"system": q.meta["system"]} if q.meta.get("system") else {}
    header = {"header": q.meta["header"]} if q.meta.get("header") else {}
    if strategy == "direct":
        rec = text_client.chat(direct_messages(q.stem, q.passage, q.options, multiple, **sysmsg, **header), tag=f"V-{strategy}:{q.uid}")
        chosen = parse_letters(rec["content"], labels)
        if not multiple:
            chosen = chosen[:1]
        vec = {o: ("+" if o in chosen else "-") for o in labels} if chosen else {o: "?" for o in labels}
    elif strategy == "judge":
        data, rec = json_client.chat_json(judge_all_messages(q.stem, q.passage, q.options, multiple, **sysmsg), tag=f"V-judge:{q.uid}")
        data = data or {}
        vec = {o: _VERDICT.get(str(data.get(o, "")).strip().lower(), "?") for o in labels}
    elif strategy == "eliminate":
        data, rec = json_client.chat_json(eliminate_messages(q.stem, q.passage, q.options, **sysmsg), tag=f"V-elim:{q.uid}")
        wrong = {str(x).strip().upper()[:1] for x in (data or {}).get("incorrect", [])}
        vec = {o: ("-" if o in wrong else "?") for o in labels}
        if all(v == "-" for v in vec.values()):
            vec = {o: "?" for o in labels}
    elif strategy == "confidence":
        data, rec = json_client.chat_json(confidence_messages(q.stem, q.passage, q.options, multiple, **sysmsg), tag=f"V-conf:{q.uid}")
        data = data or {}
        chosen = parse_letters(str(data.get("answer", "")), labels)
        if not multiple:
            chosen = chosen[:1]
        vec = {o: ("+" if o in chosen else "-") for o in labels} if chosen else {o: "?" for o in labels}
        try:
            conf = float(data.get("confidence"))
        except (TypeError, ValueError):
            conf = None
        u = rec.get("usage") or {}
        return {"vector": vec, "confidence": conf, "raw": rec["content"], "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0), "served_model": rec.get("model")}
    elif strategy == "statements":
        if not q.is_statement_combo:
            return {"vector": {}, "raw": "", "prompt_tokens": 0, "completion_tokens": 0, "served_model": None}
        data, rec = json_client.chat_json(
            judge_statements_messages(split_statements(q.stem), q.passage, q.statements), tag=f"V-stmt:{q.uid}"
        )
        data = data or {}
        vec = {s: _VERDICT.get(str(data.get(s, "")).strip().lower(), "?") for s in q.statements}
    else:
        raise ValueError(strategy)
    u = rec.get("usage") or {}
    return {"vector": vec, "raw": rec["content"], "prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0), "served_model": rec.get("model")}


def unit_vector(q: Question, strategy: str, vec: dict[str, str]) -> dict[str, str]:
    """The view over the decision units: the options, or the statements of a statement-combination item."""
    if not q.is_statement_combo:
        return vec if strategy != "statements" else {}
    if strategy == "statements":
        return vec
    if strategy == "direct":
        chosen = [o for o, v in vec.items() if v == "+"]
        if len(chosen) != 1:
            return {s: "?" for s in q.statements}
        acc = set(q.option_statements[chosen[0]])
        return {s: ("+" if s in acc else "-") for s in q.statements}
    return {s: "?" for s in q.statements}

