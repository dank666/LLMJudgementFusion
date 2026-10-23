"""Prompt templates of the views."""

DIRECT_SYSTEM = "You are a geography expert."
DIRECT_HEADER = "Please answer the following geography question."

DIRECT_SINGLE = """{header}

{passage}Question:
{stem}

Options:
{options}

Instructions:
1. Select the single best answer.
2. Return only one option letter.
3. Do not provide explanations or additional text.

Answer:"""

DIRECT_MULTIPLE = """{header}

{passage}Question:
{stem}

Options:
{options}

Instructions:
1. One or more options may be correct.
2. Return only the corresponding option letters in alphabetical order.
3. Do not provide explanations or additional text.

Examples:
A
AC
ABC

Answer:"""


def _passage_block(passage: str | None, zh: bool = True) -> str:
    if not passage:
        return ""
    return f"材料：{passage}\n" if zh else f"Material:\n{passage}\n\n"


def direct_messages(stem: str, passage: str | None, options: dict[str, str], multiple: bool,
                    system: str = DIRECT_SYSTEM, header: str = DIRECT_HEADER) -> list[dict]:
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    tmpl = DIRECT_MULTIPLE if multiple else DIRECT_SINGLE
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": tmpl.format(header=header, passage=_passage_block(passage, zh=False), stem=stem,
                                                options=opts)},
    ]


JUDGE_ALL = """{passage}Question:
{stem}

Options:
{options}

Judge EACH option on its own merits: is it a correct answer to the question? {constraint}
Respond only with JSON mapping every option letter to "correct", "incorrect" or "unsure", e.g. {example}"""

ELIMINATE = """{passage}Question:
{stem}

Options:
{options}

Identify every option that is definitely NOT a correct answer. Do not list an option you are unsure about.
Respond only with JSON: {{"incorrect": ["..."]}}"""


def _options_block(options: dict[str, str]) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


def judge_all_messages(stem, passage, options, multiple, system: str = DIRECT_SYSTEM) -> list[dict]:
    example = "{" + ", ".join(f'"{k}": "..."' for k in sorted(options)) + "}"
    constraint = "One or more options may be correct." if multiple else ""
    user = JUDGE_ALL.format(passage=_passage_block(passage, zh=False), stem=stem, options=_options_block(options),
                            constraint=constraint, example=example)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def eliminate_messages(stem, passage, options, system: str = DIRECT_SYSTEM) -> list[dict]:
    user = ELIMINATE.format(passage=_passage_block(passage, zh=False), stem=stem, options=_options_block(options))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


JUDGE_STATEMENTS = """{passage}Question:
{stem}

Statements:
{statements}

Judge EACH statement on its own merits in the context of the question: is it true / does it correctly answer the question?
Respond only with JSON mapping every statement label to "true", "false" or "unsure", e.g. {example}"""


def judge_statements_messages(stem, passage, statements: dict[str, str]) -> list[dict]:
    block = "\n".join(f"{k}. {v}" for k, v in statements.items())
    example = "{" + ", ".join(f'"{k}": "..."' for k in statements) + "}"
    user = JUDGE_STATEMENTS.format(passage=_passage_block(passage, zh=False), stem=stem, statements=block, example=example)
    return [{"role": "system", "content": DIRECT_SYSTEM}, {"role": "user", "content": user}]


CONFIDENCE = """{passage}Question:
{stem}

Options:
{options}

{instruction} Also state how confident you are that your answer is correct, as an integer from 0 to 100.
Respond only with JSON: {{"answer": "...", "confidence": 0}}"""


def confidence_messages(stem, passage, options, multiple, system: str = DIRECT_SYSTEM) -> list[dict]:
    instruction = (
        "One or more options may be correct; give all correct option letters in alphabetical order."
        if multiple
        else "Select the single best answer and give its letter."
    )
    user = CONFIDENCE.format(passage=_passage_block(passage, zh=False), stem=stem, options=_options_block(options),
                             instruction=instruction)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

