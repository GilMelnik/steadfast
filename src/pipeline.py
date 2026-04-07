"""Steadfast support ticket triage pipeline entry point."""

from __future__ import annotations

import json
import sys
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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def load_eval_set(eval_path: Path) -> List[Dict[str, str]]:
    with eval_path.open("r", encoding="utf-8") as handle:
        data = cast(List[Dict[str, str]], json.load(handle))
    return [dict(item) for item in data]


def _run_single_pass(
    tickets: List[Dict[str, str]],
    kb_index: Dict[str, object],
    aggressive_priority: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object], float]:
    predictions: List[Dict[str, object]] = []
    validation_issue_count = 0

    for ticket in tickets:
        raw = classify_ticket(ticket, kb_index)
        validated, validation_issues = validate_prediction(raw)
        validation_issue_count += len(validation_issues)
        final = apply_heuristics(ticket, validated, aggressive_priority=aggressive_priority)
        predictions.append(final)

    metrics = evaluate_predictions(tickets, predictions)
    error_analysis = analyze_errors(tickets, predictions)
    validation_failure_rate = validation_issue_count / max(len(tickets), 1)
    return predictions, metrics, error_analysis, round(validation_failure_rate, 4)


def run_pipeline() -> Dict[str, object]:
    kb_rows = load_knowledge_base(DATA_DIR / "knowledge_base.csv")
    tickets = load_eval_set(DATA_DIR / "eval_set.json")
    kb_index = preprocess_knowledge_base(kb_rows)

    baseline = _run_single_pass(tickets, kb_index, aggressive_priority=False)
    baseline_predictions, baseline_metrics, baseline_analysis, baseline_validation_rate = baseline
    second_metrics: Dict[str, Any] = {}
    second_validation_rate = 0.0

    baseline_priority_accuracy = _to_float(baseline_metrics.get("priority_accuracy", 0.0))
    should_iterate = baseline_priority_accuracy < 0.82
    if should_iterate:
        second = _run_single_pass(tickets, kb_index, aggressive_priority=True)
        second_predictions, second_metrics, second_analysis, second_validation_rate = second

        baseline_score = _to_float(baseline_metrics.get("category_accuracy_with_partial_credit", 0.0)) + _to_float(
            baseline_metrics.get("priority_accuracy", 0.0)
        ) + _to_float(baseline_metrics.get("response_quality_score", 0.0))
        second_score = _to_float(second_metrics.get("category_accuracy_with_partial_credit", 0.0)) + _to_float(
            second_metrics.get("priority_accuracy", 0.0)
        ) + _to_float(second_metrics.get("response_quality_score", 0.0))

        if second_score > baseline_score:
            chosen_predictions = second_predictions
            chosen_metrics = second_metrics
            chosen_analysis = second_analysis
            chosen_validation_rate = second_validation_rate
            selected_iteration = "iteration_2_aggressive_priority"
        else:
            chosen_predictions = baseline_predictions
            chosen_metrics = baseline_metrics
            chosen_analysis = baseline_analysis
            chosen_validation_rate = baseline_validation_rate
            selected_iteration = "iteration_1_baseline"
    else:
        chosen_predictions = baseline_predictions
        chosen_metrics = baseline_metrics
        chosen_analysis = baseline_analysis
        chosen_validation_rate = baseline_validation_rate
        selected_iteration = "iteration_1_baseline"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_payload = {
        "selected_iteration": selected_iteration,
        "validation_failure_rate": chosen_validation_rate,
        "metrics": chosen_metrics,
        "predictions": chosen_predictions,
        "iteration_log": [
            {
                "name": "iteration_1_baseline",
                "aggressive_priority": False,
                "metrics": baseline_metrics,
                "validation_failure_rate": baseline_validation_rate,
            }
        ],
    }
    if should_iterate:
        eval_payload["iteration_log"].append(
            {
                "name": "iteration_2_aggressive_priority",
                "aggressive_priority": True,
                "metrics": second_metrics,
                "validation_failure_rate": second_validation_rate,
            }
        )

    with (OUTPUT_DIR / "eval_results.json").open("w", encoding="utf-8") as handle:
        json.dump(eval_payload, handle, indent=2)

    with (OUTPUT_DIR / "error_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(chosen_analysis, handle, indent=2)

    return {
        "eval_results": eval_payload,
        "error_analysis": chosen_analysis,
    }


if __name__ == "__main__":
    result = run_pipeline()
    eval_metrics = cast(Dict[str, Any], cast(Dict[str, Any], result["eval_results"])["metrics"])
    print(
        "Pipeline complete | "
        f"Category exact: {_to_float(eval_metrics.get('category_accuracy_exact', 0.0)):.3f}, "
        f"Priority: {_to_float(eval_metrics.get('priority_accuracy', 0.0)):.3f}, "
        f"Response quality: {_to_float(eval_metrics.get('response_quality_score', 0.0)):.3f}"
    )
