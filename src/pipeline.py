"""Steadfast support ticket triage pipeline entry point."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

try:
    from src.agent import classify_ticket
    from src.analyze import analyze_errors
    from src.evaluate import evaluate_predictions
    from src.postprocess import apply_heuristics
    from src.preprocess import load_knowledge_base, preprocess_knowledge_base
    from src.validate import validate_prediction
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.agent import classify_ticket
    from src.analyze import analyze_errors
    from src.evaluate import evaluate_predictions
    from src.postprocess import apply_heuristics
    from src.preprocess import load_knowledge_base, preprocess_knowledge_base
    from src.validate import validate_prediction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def load_eval_set(eval_path: Path) -> List[Dict[str, str]]:
    with eval_path.open("r", encoding="utf-8") as handle:
        data = cast(List[Dict[str, str]], json.load(handle))
    return [dict(item) for item in data]


def _build_run_output_dir(now: datetime | None = None) -> Path:
    run_time = now or datetime.now()
    run_dir_name = run_time.strftime("%Y_%m_%d_%H_%M")
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
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object], float]:
    predictions: List[Dict[str, object]] = []
    validation_issue_count = 0

    for i, ticket in enumerate(tickets):
        raw = classify_ticket(ticket, kb_index)
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


def run_pipeline(verbose: bool = False) -> Dict[str, object]:
    kb_rows = load_knowledge_base(DATA_DIR / "knowledge_base.csv")
    tickets = load_eval_set(DATA_DIR / "eval_set.json")
    kb_index = preprocess_knowledge_base(kb_rows)

    if verbose:
        print(f"Loaded {len(kb_rows)} knowledge base rows and {len(tickets)} tickets.")

    predictions, metrics, error_analysis, validation_failure_rate = _run_single_pass(
        tickets,
        kb_index,
        verbose=verbose,
    )

    eval_payload = {
        "validation_failure_rate": validation_failure_rate,
        "metrics": metrics,
        "predictions": predictions,
    }

    run_output_dir = _build_run_output_dir()
    _write_pipeline_outputs(run_output_dir, eval_payload, error_analysis)
    _write_pipeline_outputs(OUTPUT_DIR, eval_payload, error_analysis)

    if verbose:
        print(f"\nOutputs written to: {run_output_dir}")

    return {
        "eval_results": eval_payload,
        "error_analysis": error_analysis,
    }


if __name__ == "__main__":
    result = run_pipeline(verbose=True)
    eval_metrics = cast(Dict[str, Any], cast(Dict[str, Any], result["eval_results"])["metrics"])
    print(
        "Pipeline complete | "
        f"Category exact: {_to_float(eval_metrics.get('category_accuracy_exact', 0.0)):.3f}, "
        f"Priority: {_to_float(eval_metrics.get('priority_accuracy', 0.0)):.3f}, "
        f"Response quality: {_to_float(eval_metrics.get('response_quality_score', 0.0)):.3f}"
    )
