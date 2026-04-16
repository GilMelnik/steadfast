# Steadfast support ticket triage pipeline

This document describes how to install, configure, and run the repository. The **entry point** is [`src/pipeline.py`](src/pipeline.py): it loads data, runs preprocessing, LLM classification, validation, heuristics, evaluation, and error analysis in one pass.

---

## Python version

This project was run in a local virtual environment at **`.venv`** using **Python 3.10.12**:

```bash
.venv/bin/python --version   # Python 3.10.12
```

Use **Python 3.10** and ensure `pip` installs into the same interpreter you use to execute `src/pipeline.py`.

Create or reuse a venv from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

---

## Install

From the **repository root** (the directory that contains `data/`, `src/`, and `requirements.txt`):

```bash
pip install -r requirements.txt
```

---

## Configuration (`.env`)

1. Copy the example file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set credentials and endpoint for the LLM client (see [`.env.example`](.env.example)).

The pipeline does **not** load `.env` automatically unless your shell or tooling exports those variables. Typical options:

- **Export in the shell** before running (recommended for one-off runs):

  ```bash
  set -a && source .env && set +a
  python src/pipeline.py
  ```

- Or use a tool that loads `.env` into the environment (for example `direnv`, or your IDE’s run configuration).

### Environment variables (LLM)

[`src/agent.py`](src/agent.py) resolves configuration as follows:

| Variable | Role |
|----------|------|
| `LSP_API_KEY` or `API_KEY` | API key for the OpenAI-compatible endpoint |
| `LSP_API_BASE` or `BASE_URL` | Base URL for the API (OpenAI-compatible `/v1` usage) |

If `BASE_URL` / `LSP_API_BASE` is unset, the code falls back to its default OpenAI-compatible base URL. If no model env var is set, a default model id in code is used.

You can override the model per run with `--llm-model` (see below).

---

## Where to run from

Always run commands from the **repository root**, so paths resolve correctly:

- `data/knowledge_base.csv` — knowledge base  
- `data/eval_set.json` — labeled dev / eval tickets  
- `outputs/` — timestamped run directories (LLM logs + JSON outputs)  
- `output/` — copy of the **latest** run’s JSON (and plots when enabled)

Running `python src/pipeline.py` from another working directory will fail or read the wrong files.

---

## Command-line interface

Invoke the entry point (from repo root):

```bash
python src/pipeline.py [options]
```

The built-in help matches the implementation:

```bash
python src/pipeline.py --help
```

### Options

| Option | Meaning |
|--------|---------|
| `--verbose` | Print per-ticket progress and a short metrics summary |
| `--plot` | Write Matplotlib PNG charts (metrics, confusion-style views, error analysis plots). Requires `matplotlib` from `requirements.txt` |
| `--output-name-suffix S` | Append `S` (sanitized) to the timestamped folder name under `outputs/` |
| `--llm-model M` | Use model id `M` for API calls |
| `--llm-rerun` | **Default behavior** if no other LLM mode is given: call the LLM and write `llm_outputs.jsonl` in the new run directory |
| `--llm-use-last` | Skip API calls; reuse rows from the **newest** `outputs/*/llm_outputs.jsonl` |
| `--llm-from PATH` | Skip API calls; reuse rows from a specific `llm_outputs.jsonl` file or a directory that contains it |
| `-h`, `--help` | Show usage and exit |

Use **only one** of `--llm-rerun`, `--llm-use-last`, and `--llm-from`.

The CLI does **not** define `--eval`. A normal run already performs **evaluation** (metrics) and **error analysis**; there is no separate “eval-only” flag.

---

## What a run does

For each ticket in `data/eval_set.json`, the pipeline:

1. Loads and preprocesses the knowledge base (`data/knowledge_base.csv`).  
2. Calls the LLM (unless you used `--llm-use-last` or `--llm-from`).  
3. Validates model output (`src/validate.py`).  
4. Applies heuristics (`src/postprocess.py`).  
5. Aggregates metrics and runs error analysis (`src/evaluate.py`, `src/analyze.py`).

On success, the process prints one summary line to stdout with category exact accuracy, priority accuracy, and response quality score.

---

## Output artifacts

### Per run: `outputs/<timestamp>_<optional_suffix>/`

| File | Contents |
|------|----------|
| `llm_outputs.jsonl` | One JSON object per ticket (raw LLM path); present when the LLM was invoked for this run |
| `eval_results.json` | `validation_failure_rate`, `metrics`, and full `predictions` list |
| `error_analysis.json` | Structured error analysis used for iteration |

The same `eval_results.json` and `error_analysis.json` are **also written** to `output/` (latest run overwrite).

### Optional plots (`--plot`)

When `--plot` is passed, PNGs such as overall/category/priority scores, confusion-style charts, score distributions, and error-analysis figures are written under **both** the timestamped `outputs/.../` folder and `output/`. Filenames are defined in [`src/visualize.py`](src/visualize.py) (for example `overall_metric_scores.png`, `error_root_causes.png`, etc.).


---

## Quick reference

```bash
# First-time: install and configure
pip install -r requirements.txt
cp .env.example .env
# edit .env, then export vars (example)
set -a && source .env && set +a

# Full run with API calls, verbose logs, and charts
python src/pipeline.py --verbose --plot

# Re-run without calling the LLM again
python src/pipeline.py --llm-use-last --verbose
```
