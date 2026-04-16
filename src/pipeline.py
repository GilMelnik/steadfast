"""Steadfast support ticket triage pipeline entry point."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import LLM_OUTPUTS_FILENAME, classify_ticket, load_llm_outputs_jsonl
from src.analyze import analyze_errors
from src.evaluate import evaluate_predictions
from src.postprocess import apply_heuristics
from src.preprocess import load_knowledge_base, preprocess_knowledge_base
from src.validate import validate_prediction
from src.visualize import render_pipeline_visualizations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _sanitize_output_name_suffix(raw: str) -> str:
    s = raw.strip()
    if not s:
        return ""
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.UNICODE)
    return s[:120]


def _find_latest_llm_outputs_file() -> Path:
    if not OUTPUTS_DIR.is_dir():
        raise FileNotFoundError(f"No outputs directory: {OUTPUTS_DIR}")
    candidates: List[Tuple[float, Path]] = []
    for run_dir in OUTPUTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        candidate = run_dir / LLM_OUTPUTS_FILENAME
        if candidate.is_file():
            candidates.append((candidate.stat().st_mtime, candidate))
    if not candidates:
        raise FileNotFoundError(
            f"No {LLM_OUTPUTS_FILENAME} found under {OUTPUTS_DIR}. Run the pipeline without "
            "--llm-use-last first to generate LLM logs."
        )
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


@dataclass(frozen=False)
class PipelineCliOptions:
    verbose: bool
    plot: bool
    output_name_suffix: str
    llm_mode: str  # "rerun" | "last" | "path"
    llm_path: Optional[Path]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def load_eval_set(eval_path: Path) -> List[Dict[str, str]]:
    with eval_path.open("r", encoding="utf-8") as handle:
        data = cast(List[Dict[str, str]], json.load(handle))
    return [dict(item) for item in data]


def _build_run_output_dir(now: datetime | None = None, name_suffix: str = "") -> Path:
    run_time = now or datetime.now()
    base = run_time.strftime("%Y_%m_%d_%H_%M")
    suffix = _sanitize_output_name_suffix(name_suffix)
    if suffix:
        run_dir_name = f"{base}_{suffix}_"
    else:
        run_dir_name = base + "_"
    return OUTPUTS_DIR / run_dir_name


def _write_pipeline_outputs(target_dir: Path, eval_payload: Dict[str, object], error_analysis: Dict[str, object]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with (target_dir / "eval_results.json").open("w", encoding="utf-8") as handle:
        json.dump(eval_payload, handle, indent=2)

    with (target_dir / "error_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(error_analysis, handle, indent=2)


def _run_single_pass(
    tickets: List[Dict[str, str]],
    kb_index: Dict[str, object],
    verbose: bool = False,
    *,
    llm_output_dir: Optional[Path] = None,
    llm_cache: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object], float]:
    predictions: List[Dict[str, object]] = []
    validation_issue_count = 0

    for i, ticket in enumerate(tickets):
        raw = classify_ticket(
            ticket,
            kb_index,
            llm_output_dir=llm_output_dir,
            llm_cache=llm_cache,
        )
        validated, validation_issues = validate_prediction(raw)
        validation_issue_count += len(validation_issues)
        final = apply_heuristics(ticket, validated)
        predictions.append(final)

        if verbose:
            print(
                f"[{i + 1}/{len(tickets)}] ticket_id={ticket.get('ticket_id', '?')} | "
                f"category={final.get('category', '?')} | "
                f"priority={final.get('priority', '?')} | "
                f"validation_issues={len(validation_issues)}"
            )

    metrics = evaluate_predictions(tickets, predictions)
    error_analysis = analyze_errors(tickets, predictions)
    validation_failure_rate = validation_issue_count / max(len(tickets), 1)

    if verbose:
        print("\n--- Metrics ---")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print(f"  validation_failure_rate: {round(validation_failure_rate, 4)}")

    return predictions, metrics, error_analysis, round(validation_failure_rate, 4)


def run_pipeline(
    verbose: bool = False,
    plot: bool = True,
    *,
    output_name_suffix: str = "",
    llm_mode: str = "rerun",
    llm_path: Optional[Path] = None,
) -> Dict[str, object]:
    # kb_rows = load_knowledge_base(DATA_DIR / "knowledge_base.csv")
    # tickets = load_eval_set(DATA_DIR / "eval_set.json")
    kb_rows = load_knowledge_base(DATA_DIR / "train_0.csv")
    tickets = load_eval_set(DATA_DIR / "test_0.json")
    kb_index = preprocess_knowledge_base(kb_rows)

    if verbose:
        print(f"Loaded {len(kb_rows)} knowledge base rows and {len(tickets)} tickets.")

    run_output_dir = _build_run_output_dir(name_suffix=output_name_suffix)

    llm_output_dir: Optional[Path] = None
    llm_cache: Optional[Dict[str, Dict[str, object]]] = None
    if llm_mode == "rerun":
        llm_output_dir = run_output_dir
    elif llm_mode == "last":
        cache_file = _find_latest_llm_outputs_file()
        if verbose:
            print(f"Using cached LLM outputs from: {cache_file}")
        llm_cache = load_llm_outputs_jsonl(cache_file)
    elif llm_mode == "path":
        assert llm_path is not None
        if verbose:
            print(f"Using cached LLM outputs from: {llm_path}")
        llm_cache = load_llm_outputs_jsonl(llm_path)

    predictions, metrics, error_analysis, validation_failure_rate = _run_single_pass(
        tickets,
        kb_index,
        verbose=verbose,
        llm_output_dir=llm_output_dir,
        llm_cache=llm_cache,
    )

    eval_payload = {
        "validation_failure_rate": validation_failure_rate,
        "metrics": metrics,
        "predictions": predictions,
    }
    if plot:
        render_pipeline_visualizations(
            tickets,
            predictions,
            metrics,
            error_analysis,
            [run_output_dir, OUTPUT_DIR],
        )
    _write_pipeline_outputs(run_output_dir, eval_payload, error_analysis)
    _write_pipeline_outputs(OUTPUT_DIR, eval_payload, error_analysis)

    if verbose:
        print(f"\nOutputs written to: {run_output_dir}")

    return {
        "eval_results": eval_payload,
        "error_analysis": error_analysis,
    }


def _parse_pipeline_args(args: List[str]) -> PipelineCliOptions:
    if "--help" in args or "-h" in args:
        print(
            "Usage: python -m src.pipeline [options]\n\n"
            "Options:\n"
            "  --verbose              Print per-ticket progress and metrics\n"
            "  --plot                 Write visualization PNGs\n"
            "  --output-name-suffix S Append S to the timestamped folder name under outputs/\n"
            "  --llm-rerun            Call the LLM and write llm_outputs.jsonl in the new run dir (default)\n"
            "  --llm-use-last         Reuse LLM rows from the newest outputs/*/llm_outputs.jsonl\n"
            "  --llm-from PATH        Reuse LLM rows from a specific jsonl file or run directory\n"
            "  -h, --help             Show this message\n"
        )
        raise SystemExit(0)

    verbose = "--verbose" in args
    plot = "--plot" in args

    has_from = "--llm-from" in args
    has_last = "--llm-use-last" in args
    has_rerun = "--llm-rerun" in args
    if sum([has_from, has_last, has_rerun]) > 1:
        print(
            "Error: use only one of --llm-from, --llm-use-last, --llm-rerun.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    llm_mode = "rerun"
    llm_path: Optional[Path] = None
    if has_from:
        try:
            idx = args.index("--llm-from")
        except ValueError:
            idx = -1
        if idx < 0 or idx + 1 >= len(args):
            print("Error: --llm-from requires a path.", file=sys.stderr)
            raise SystemExit(2)
        llm_mode = "path"
        llm_path = Path(args[idx + 1]).expanduser().resolve()
    elif has_last:
        llm_mode = "last"

    output_name_suffix = ""
    if "--output-name-suffix" in args:
        idx = args.index("--output-name-suffix")
        if idx + 1 >= len(args):
            print("Error: --output-name-suffix requires a value.", file=sys.stderr)
            raise SystemExit(2)
        output_name_suffix = args[idx + 1]

    return PipelineCliOptions(
        verbose=verbose,
        plot=plot,
        output_name_suffix=output_name_suffix,
        llm_mode=llm_mode,
        llm_path=llm_path,
    )


if __name__ == "__main__":
    opts = _parse_pipeline_args(sys.argv[1:])
    opts.output_name_suffix = "_new_LLM_claude-opus-4-6"
    result = run_pipeline(
        verbose=opts.verbose,
        plot=opts.plot,
        output_name_suffix=opts.output_name_suffix,
        llm_mode=opts.llm_mode,
        llm_path=opts.llm_path,
    )
    eval_metrics = cast(Dict[str, Any], cast(Dict[str, Any], result["eval_results"])["metrics"])
    print(
        "Pipeline complete | "
        f"Category exact: {_to_float(eval_metrics.get('category_accuracy_exact', 0.0)):.3f}, "
        f"Priority: {_to_float(eval_metrics.get('priority_accuracy', 0.0)):.3f}, "
        f"Response quality: {_to_float(eval_metrics.get('response_quality_score', 0.0)):.3f}"
    )
