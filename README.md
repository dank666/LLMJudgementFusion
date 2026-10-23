# When Is Voting Enough? Fusing the Judgements of Large Language Models

Code and model outputs for the paper "When Is Voting Enough? Fusing the Judgements of Large Language Models on
Single and Set-Valued Answers".

The benchmarks are not included. `data/download.py` downloads them from their sources and checks them against the
copies used in the paper. All model answers are in `results/views/`, so the results can be reproduced without
calling any model.

## Setup

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m pytest

## Reproducing the results

    .venv/bin/python data/download.py
    .venv/bin/python experiments/prepare_intl.py
    .venv/bin/python experiments/prepare_multi.py
    .venv/bin/python experiments/prepare_gaokao.py
    .venv/bin/python experiments/prepare_setvalued.py
    .venv/bin/python experiments/dedupe.py
    .venv/bin/python data/check_processed.py
    .venv/bin/python experiments/run_singlechoice_test.py
    .venv/bin/python experiments/run_setvalued_test.py
    .venv/bin/python experiments/plot_selection.py

`run_singlechoice_test.py` gives Table 2 and the numbers of Section 4.2, `run_setvalued_test.py` Tables 3 and 4
and the data of Figure 2, which `plot_selection.py` draws (it needs a LaTeX installation for the fonts). All three
write to `results/`.

## Files

- `src/llmfusion/`: formal contexts (`fca.py`), consensus (`consensus.py`), prompts and views (`prompts.py`,
  `views.py`), a cached client for OpenAI-compatible APIs (`llm.py`) and the question loaders (`data.py`).
- `experiments/`: data preparation, `collect_views.py` for asking the models, the fusion methods
  (`fusion_single.py`, `fusion_sets.py`), the two evaluation scripts and the plotting script.
- `results/views/<split>/<model>__<strategy>.jsonl`: one line per question with the parsed judgement, the raw
  response and the token counts.

To collect new views, put an OpenAI-compatible endpoint into `.env` (`BASELINE_BASE_URL`, `BASELINE_API_KEY`) and
run `experiments/collect_views.py`. Responses are cached in `results/cache/`.

## License

MIT for the code. The benchmarks keep the licenses of their original sources.
