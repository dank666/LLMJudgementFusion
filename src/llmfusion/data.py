"""Question schema and loaders of the examination benchmarks."""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

CIRCLED = "①②③④⑤⑥⑦⑧"
FIGURE_CUES = re.compile(
    r"如图|读图|下图|上图|图中|图示|示意图|据图|图例|右图|左图|图[甲乙丙丁]|图\s*\d|"
    r"下表|表中|据表|上表|如表|表[一二三]|表\s*\d|统计图|等高线图|等值线图"
)


@dataclass
class Question:
    uid: str
    dataset: str
    qtype: str  # single | multiple
    stem: str
    options: dict[str, str]
    answer: list[str]
    domain: str | None = None
    passage: str | None = None
    # statement-combination items: the statements and the statements each option stands for
    statements: dict[str, str] = field(default_factory=dict)
    option_statements: dict[str, list[str]] = field(default_factory=dict)
    figure_dependent: bool = False
    source: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_statement_combo(self) -> bool:
        return bool(self.option_statements)

    @property
    def full_text(self) -> str:
        return f"{self.passage}\n{self.stem}" if self.passage else self.stem

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(**d)


def normalize(text: str) -> str:
    text = text.replace("．", ".").replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", "", text)
    return text


def _mentions_figure(text: str) -> bool:
    return bool(FIGURE_CUES.search(text))


def parse_statements(stem: str, options: dict[str, str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Detect ①②③④-style statement-combination items.

    Returns (statements, option_statements); both empty when the item is not a
    statement-combination item.
    """
    option_sets = {}
    for label, text in options.items():
        marks = [c for c in text if c in CIRCLED]
        residue = re.sub(rf"[{CIRCLED}\s、,，和与]", "", text)
        if not marks or residue:
            return {}, {}
        option_sets[label] = marks
    parts = re.split(rf"([{CIRCLED}])", stem)
    statements = {}
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1].strip(" \n\t；;。.,，")
        if body:
            statements[parts[i]] = body
    used = {m for marks in option_sets.values() for m in marks}
    if not used or not used <= set(statements):
        return {}, {}
    return statements, option_sets




def load_agieval(path: Path) -> list[Question]:
    questions = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        item = json.loads(line)
        options = {}
        for opt in item["options"]:
            m = re.match(r"\s*\(([A-D])\)\s*(.*)", opt, re.S) or re.match(r"\s*([A-D])\s*[.．]\s*(.*)", opt, re.S)
            options[m.group(1)] = m.group(2).strip()
        stem = item["question"].strip()
        passage = (item.get("passage") or "").strip() or None
        statements, option_statements = parse_statements(stem, options)
        questions.append(
            Question(
                uid=f"agieval-{i:03d}",
                dataset="agieval",
                qtype="single",
                stem=stem,
                options=options,
                answer=[item["label"].strip()],
                passage=passage,
                statements=statements,
                option_statements=option_statements,
                figure_dependent=_mentions_figure((passage or "") + stem),
                source=item.get("other", {}).get("source"),
            )
        )
    return questions



# Sub-question markers such as "9．" or "1．（4 分）". OCR sometimes glues the
# marker to its neighbours ("提高效 益3. 无人机", "1．2010～2020 年"), so the
# pattern is permissive; consecutive numbering and option parsing reject
# false matches such as decimals in tables.
_SUBQ = re.compile(r"(?<!\d)(\d{1,2})\s*[．.]\s*(?:[（(]\s*\d+\s*分\s*[）)])?")
_CJK_GAP = re.compile(r"(?<=[一-鿿，。、；：（）])\s+(?=[一-鿿，。、；：（）])")

# Documented corrections of OCR errors in the raw GAOKAO-Bench file.
GAOKAO_OCR_FIXES = {
    27: [("C.科研实力 B.配套产业", "C.科研实力 D.配套产业")],
}
_OPT = re.compile(r"([A-D])\s*[．.]")


def _split_options(segment: str) -> tuple[str, dict[str, str]] | None:
    """Split a sub-question segment into stem and A-D options."""
    positions = []
    start = 0
    for label in "ABCD":
        for m in _OPT.finditer(segment, start):
            if m.group(1) == label:
                positions.append((label, m.start(), m.end()))
                start = m.end()
                break
        else:
            return None
    stem = segment[: positions[0][1]].strip()
    options = {}
    for idx, (label, _, end) in enumerate(positions):
        stop = positions[idx + 1][1] if idx + 1 < len(positions) else len(segment)
        options[label] = re.sub(r"\s+", " ", segment[end:stop]).strip(" ．.\n")
    return stem, options


def _parse_gaokao_group(text: str, n_answers: int) -> tuple[str, list[tuple[str, dict[str, str]]]] | None:
    marks = list(_SUBQ.finditer(text))
    # Take the first run of consecutively numbered markers whose length equals
    # the number of answers and whose segments all parse into A-D options.
    for first in range(len(marks)):
        run = [marks[first]]
        for m in marks[first + 1 :]:
            if int(m.group(1)) == int(run[-1].group(1)) + 1:
                run.append(m)
            if len(run) == n_answers:
                break
        if len(run) != n_answers:
            continue
        segments = []
        for j, m in enumerate(run):
            stop = run[j + 1].start() if j + 1 < len(run) else len(text)
            parsed = _split_options(text[m.end() : stop])
            if parsed is None:
                break
            segments.append(parsed)
        else:
            passage = re.sub(r"\s+", " ", text[: run[0].start()]).strip()
            return passage, segments
    return None


def load_gaokao_bench(path: Path) -> tuple[list[Question], list[dict]]:
    items = json.loads(path.read_text(encoding="utf-8"))["example"]
    questions, failures = [], []
    for item in items:
        text = item["question"]
        for old, new in GAOKAO_OCR_FIXES.get(int(item["index"]), []):
            text = text.replace(old, new)
        parsed = _parse_gaokao_group(text, len(item["answer"]))
        if parsed is None:
            failures.append({"index": item["index"], "year": item["year"]})
            continue
        passage, subs = parsed
        passage = _CJK_GAP.sub("", passage)
        for k, ((stem, options), ans) in enumerate(zip(subs, item["answer"])):
            stem = _CJK_GAP.sub("", re.sub(r"\s+", " ", stem).strip())
            options = {label: _CJK_GAP.sub("", text) for label, text in options.items()}
            statements, option_statements = parse_statements(stem, options)
            questions.append(
                Question(
                    uid=f"gaokao-{int(item['index']):02d}-{k + 1}",
                    dataset="gaokao_bench",
                    qtype="single",
                    stem=stem,
                    options=options,
                    answer=[ans.strip()],
                    passage=passage or None,
                    statements=statements,
                    option_statements=option_statements,
                    figure_dependent=_mentions_figure(passage + stem),
                    source=f"{item['year']}{item['category']}",
                )
            )
    return questions, failures




def write_jsonl(questions: list[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(q.to_json() for q in questions) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[Question]:
    return [Question.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]



LETTERS = "ABCDEFGH"
ROMAN = re.compile(r"(?<![A-Za-z])(I{1,3}|IV|V)\s*[.)]")


def _roman_statements(stem: str, options: dict[str, str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Detect I/II/III-style statement-combination items (e.g. options "I ve II", "Yalnız III")."""
    marks = [m for m in ROMAN.finditer(stem)]
    if len(marks) < 2:
        return {}, {}
    statements = {}
    for k, m in enumerate(marks):
        stop = marks[k + 1].start() if k + 1 < len(marks) else len(stem)
        body = stem[m.end() : stop].strip(" ,;\n\t")
        statements[m.group(1)] = body
    option_sets = {}
    for label, text in options.items():
        found = re.findall(r"(?<![A-Za-z])(I{1,3}|IV|V)(?![A-Za-z])", text)
        if not found:
            return {}, {}
        option_sets[label] = list(dict.fromkeys(found))
    if not {s for v in option_sets.values() for s in v} <= set(statements):
        return {}, {}
    # The last statement body usually carries the question sentence; cut it at a line break.
    last = list(statements)[-1]
    statements[last] = re.split(r"\n|\s{3,}", statements[last])[0].strip()
    return statements, option_sets


def _question(uid, dataset, stem, options, answer, language, source=None, passage=None, meta=None) -> Question:
    stem = stem.strip()
    options = {k: str(v).strip() for k, v in options.items()}
    statements, option_statements = parse_statements(stem, options)
    if not option_statements:
        statements, option_statements = _roman_statements(stem, options)
    return Question(
        uid=uid,
        dataset=dataset,
        qtype="single",
        stem=stem,
        options=options,
        answer=[answer],
        passage=passage,
        statements=statements,
        option_statements=option_statements,
        source=source,
        meta={"language": language, **(meta or {})},
    )


def load_mmlu_geography(path) -> list[Question]:
    import pandas as pd

    df = pd.read_parquet(path)
    split = Path(path).stem.split("_")[-1]
    out = []
    for i, r in df.iterrows():
        opts = {LETTERS[k]: c for k, c in enumerate(r["choices"])}
        out.append(_question(f"mmlu-{split}-{i:03d}", "mmlu_hsg", r["question"], opts, LETTERS[int(r["answer"])], "English", "MMLU high_school_geography"))
    return out


def load_exams_geography(path) -> list[Question]:
    import pandas as pd

    df = pd.read_parquet(path)
    split = Path(path).stem.split("_")[-1]
    out = []
    for _, r in df.iterrows():
        info = r["info"]
        if "Geograph" not in str(info.get("subject")):
            continue
        ch = r["question"]["choices"]
        opts = dict(zip(ch["label"], ch["text"]))
        out.append(
            _question(f"exams-{split}-{r['id']}", "exams", r["question"]["stem"], opts, r["answerKey"], info["language"],
                      f"EXAMS grade {info.get('grade')}")
        )
    return out


def load_arabicmmlu_geography(path) -> list[Question]:
    import pandas as pd

    df = pd.read_csv(path)
    level = Path(path).stem.replace("arabicmmlu_geo_", "").replace("_test", "")
    out = []
    for _, r in df.iterrows():
        opts = {LETTERS[k]: r[f"Option {k + 1}"] for k in range(5) if isinstance(r.get(f"Option {k + 1}"), str)}
        ctx = r["Context"] if isinstance(r["Context"], str) else None
        out.append(_question(f"arabicmmlu-{int(r['ID'])}", "arabicmmlu", r["Question"], opts, str(r["Answer Key"]).strip(),
                             "Arabic", f"{r['Country']} {level}", passage=ctx))
    return out


def load_indommlu_geography(path) -> tuple[list[Question], int]:
    import pandas as pd

    df = pd.read_csv(path)
    df = df[df["subject"] == "Geografi"]
    out, skipped = [], 0
    for _, r in df.iterrows():
        opts = {}
        for line in str(r["jawaban"]).splitlines():
            m = re.match(r"\s*([A-E])\s*[.)]\s*(.*)", line)
            if m:
                opts[m.group(1)] = m.group(2)
        if len(opts) < 2 or str(r["kunci"]).strip() not in opts:
            skipped += 1
            continue
        out.append(_question(f"indommlu-{int(r['id'])}", "indommlu", str(r["soal"]), opts, str(r["kunci"]).strip(),
                             "Indonesian", f"{r['level']} {r['keterangan sumber']}"))
    return out, skipped


def load_turkishmmlu_geography(path) -> list[Question]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    split = Path(path).parent.name if Path(path).parent.name in ("dev", "test") else Path(path).stem.split("_")[-1]
    return [
        _question(f"turkishmmlu-{split}-{i:03d}", "turkishmmlu", x["question"], {LETTERS[k]: c for k, c in enumerate(x["choices"])},
                  x["answer"], "Turkish", "TurkishMMLU Geography")
        for i, x in enumerate(items)
    ]




def load_gaokao_subject(path, subject: str) -> tuple[list[Question], list[dict]]:
    """A GAOKAO-Bench objective file of any subject; multi-letter keys become multiple-select items."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))["example"]
    questions, failures = [], []
    for item in items:
        text = item["question"]
        parsed = _parse_gaokao_group(text, len(item["answer"]))
        if parsed is None:
            failures.append({"index": item["index"], "year": item["year"]})
            continue
        passage, subs = parsed
        passage = _CJK_GAP.sub("", passage)
        for k, ((stem, options), ans) in enumerate(zip(subs, item["answer"])):
            stem = _CJK_GAP.sub("", re.sub(r"\s+", " ", stem).strip())
            options = {label: _CJK_GAP.sub("", t) for label, t in options.items()}
            letters = sorted(set(re.sub(r"[^A-D]", "", ans.upper())))
            if not letters:
                failures.append({"index": item["index"], "year": item["year"], "answer": ans})
                continue
            statements, option_statements = parse_statements(stem, options)
            questions.append(
                Question(
                    uid=f"gaokao{subject.lower()}-{int(item['index']):03d}-{k + 1}",
                    dataset=f"gaokao_{subject.lower()}",
                    qtype="multiple" if len(letters) > 1 else "single",
                    stem=stem,
                    options=options,
                    answer=letters,
                    passage=passage or None,
                    statements=statements,
                    option_statements=option_statements,
                    figure_dependent=_mentions_figure((passage or "") + stem),
                    source=f"{item['year']}{item['category']}",
                    meta={"language": "Chinese", "subject": subject},
                )
            )
    return questions, failures


def load_mmlu_all(path) -> list[Question]:
    import pandas as pd

    df = pd.read_parquet(path)
    split = Path(path).stem.split("_")[-1]
    out = []
    for i, r in df.iterrows():
        opts = {LETTERS[k]: c for k, c in enumerate(r["choices"])}
        q = _question(f"mmlu-{split}-{r['subject']}-{i:05d}", "mmlu", r["question"], opts, LETTERS[int(r["answer"])],
                      "English", f"MMLU {r['subject']}")
        q.meta["subject"] = r["subject"]
        out.append(q)
    return out


def load_exams_all(path) -> list[Question]:
    import pandas as pd

    df = pd.read_parquet(path)
    split = Path(path).stem.split("_")[-1]
    out = []
    for _, r in df.iterrows():
        info = r["info"]
        ch = r["question"]["choices"]
        q = _question(f"exams-{split}-{r['id']}", "exams", r["question"]["stem"], dict(zip(ch["label"], ch["text"])),
                      r["answerKey"], info["language"], f"EXAMS grade {info.get('grade')}")
        q.meta["subject"] = str(info.get("subject"))
        out.append(q)
    return out


def load_arabicmmlu_all(path) -> list[Question]:
    import pandas as pd

    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        opts = {LETTERS[k]: r[f"Option {k + 1}"] for k in range(5) if isinstance(r.get(f"Option {k + 1}"), str)}
        if str(r["Answer Key"]).strip() not in opts:
            continue
        ctx = r["Context"] if isinstance(r["Context"], str) else None
        q = _question(f"arabicmmlu-{int(r['ID'])}", "arabicmmlu", r["Question"], opts, str(r["Answer Key"]).strip(),
                      "Arabic", f"{r['Country']} {r['Level']}", passage=ctx)
        q.meta["subject"] = str(r["Subject"])
        out.append(q)
    return out


def load_indommlu_all(path) -> list[Question]:
    import pandas as pd

    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        opts = {}
        for line in str(r["jawaban"]).splitlines():
            m = re.match(r"\s*([A-E])\s*[.)]\s*(.*)", line)
            if m:
                opts[m.group(1)] = m.group(2)
        if len(opts) < 2 or str(r["kunci"]).strip() not in opts:
            continue
        q = _question(f"indommlu-{int(r['id'])}", "indommlu", str(r["soal"]), opts, str(r["kunci"]).strip(),
                      "Indonesian", f"{r['level']} {r['keterangan sumber']}")
        q.meta["subject"] = str(r["subject"])
        out.append(q)
    return out


def load_turkishmmlu_subject(path, subject: str, split: str) -> list[Question]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for i, x in enumerate(items):
        q = _question(f"turkishmmlu-{split}-{subject}-{i:03d}", "turkishmmlu", x["question"],
                      {LETTERS[k]: c for k, c in enumerate(x["choices"])}, x["answer"], "Turkish", f"TurkishMMLU {subject}")
        q.meta["subject"] = subject
        out.append(q)
    return out


def split_statements(stem: str) -> str:
    """Stem text without the circled statements (①②③④) of a statement-combination item."""
    m = re.search(rf"[{CIRCLED}]", stem)
    return stem[: m.start()] if m else stem
