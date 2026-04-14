"""Visualization helpers for evaluation and error analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from src.evaluate import PRIORITY_ORDER, _category_score, _priority_score, _response_quality

VISUALIZATION_FILENAMES = (
    "overall_metric_scores.png",
    "category_scores.png",
    "priority_scores.png",
    "category_confusion_matrix.png",
    "priority_confusion_matrix.png",
    "score_distributions.png",
)
ERROR_ANALYSIS_FILENAMES = (
    "error_root_causes.png",
    "error_category_confusions.png",
    "error_priority_confusions.png",
    "error_flag_frequency.png",
)
OutputPathInput = Path | Sequence[Path] | str | List[str]


def _normalize_output_dirs(output_dirs: OutputPathInput) -> List[Path]:
    if isinstance(output_dirs, Path):
        return [output_dirs]
    if isinstance(output_dirs, str):
        return [Path(output_dirs)]

    normalized = [Path(output_dir) for output_dir in output_dirs]
    if not normalized:
        raise ValueError("At least one output directory must be provided.")
    return normalized


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Visualization output requires matplotlib. Install dependencies from requirements.txt "
            "before running the evaluation pipeline."
        ) from exc

    return plt


def _save_figure(fig: object, output_path: Path) -> None:
    plt = _load_pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_bar_chart(
        labels: Sequence[str],
        values: Sequence[float],
        counts: Sequence[int],
        title: str,
        ylabel: str,
        output_path: Path,
) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.7)))
    positions = list(range(len(labels)))
    bars = ax.barh(positions, values, color="#2c7fb8")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, max(1.0, max(values, default=0.0) * 1.1))
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.invert_yaxis()

    for bar, value, count in zip(bars, values, counts):
        ax.text(
            min(value + 0.02, ax.get_xlim()[1] - 0.02),
            bar.get_y() + (bar.get_height() / 2.0),
            f"{value:.2f} ({count})",
            va="center",
            ha="left" if value < ax.get_xlim()[1] - 0.1 else "right",
            fontsize=9,
        )

    _save_figure(fig, output_path)


def _plot_vertical_bar_chart(
        labels: Sequence[str],
        values: Sequence[float],
        title: str,
        ylabel: str,
        output_path: Path,
) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.85), 5))
    bars = ax.bar(range(len(labels)), values, color="#7a9e9f")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    upper_bound = max(values, default=0.0)
    ax.set_ylim(0.0, upper_bound * 1.15 if upper_bound else 1.0)
    for bar, value in zip(bars, values):
        label = f"{int(value)}" if float(value).is_integer() else f"{value:.2f}"
        ax.text(
            bar.get_x() + (bar.get_width() / 2.0),
            value + (upper_bound * 0.03 if upper_bound else 0.03),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save_figure(fig, output_path)


def _plot_confusion_matrix(
        matrix: List[List[int]],
        labels: Sequence[str],
        title: str,
        output_path: Path,
) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), max(5, len(labels) * 1.0)))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    max_value = max((max(row) for row in matrix), default=0)
    threshold = max_value / 2.0 if max_value else 0.0
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    _save_figure(fig, output_path)


def _plot_overall_metrics(metrics: Dict[str, float], output_path: Path) -> None:
    plt = _load_pyplot()
    labels = list(metrics.keys())
    values = list(metrics.values())
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values, color=["#22577a", "#38a3a5", "#57cc99", "#80ed99", "#c7f9cc"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Overall Evaluation Metrics")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + (bar.get_width() / 2.0),
            min(value + 0.03, 0.98),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save_figure(fig, output_path)


def _plot_score_distributions(
        category_scores: Sequence[float],
        priority_scores: Sequence[float],
        response_scores: Sequence[float],
        output_path: Path,
) -> None:
    plt = _load_pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    series = (
        ("Category", category_scores, "#386641"),
        ("Priority", priority_scores, "#bc6c25"),
        ("Response", response_scores, "#6a4c93"),
    )
    bins = [0.0, 0.25, 0.5, 0.75, 1.0]

    for ax, (title, values, color) in zip(axes, series):
        ax.hist(values, bins=bins, color=color, edgecolor="white", rwidth=0.9)
        ax.set_title(f"{title} Score Distribution")
        ax.set_xlabel("Score")
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    axes[0].set_ylabel("Ticket Count")
    _save_figure(fig, output_path)


def _labels_in_order(expected_labels: Sequence[str], predicted_labels: Sequence[str], priority: bool = False) -> List[
    str]:
    ordered: List[str] = []
    seen = set()

    if priority:
        for label in PRIORITY_ORDER:
            if label in expected_labels or label in predicted_labels:
                ordered.append(label)
                seen.add(label)

    for label in sorted(set(expected_labels) | set(predicted_labels)):
        if label not in seen:
            ordered.append(label)
            seen.add(label)

    return ordered


def render_pipeline_visualizations(
        tickets: List[Dict[str, str]],
        predictions: List[Dict[str, object]],
        metrics: Dict[str, object],
        error_analysis: Dict[str, object],
        output_paths: OutputPathInput,
) -> List[str]:
    files = render_evaluation_visualizations(tickets, predictions, metrics, output_paths)
    files.extend(render_error_analysis_visualizations(error_analysis, output_paths))
    return files


def render_evaluation_visualizations(
        tickets: List[Dict[str, str]],
        predictions: List[Dict[str, object]],
        metrics: Dict[str, object],
        output_dirs: OutputPathInput,
) -> List[str]:
    resolved_output_dirs = _normalize_output_dirs(output_dirs)
    by_ticket = {str(pred["ticket_id"]): pred for pred in predictions}
    category_scores: List[float] = []
    priority_scores: List[float] = []
    response_scores: List[float] = []
    expected_categories: List[str] = []
    predicted_categories: List[str] = []
    expected_priorities: List[str] = []
    predicted_priorities: List[str] = []

    for ticket in tickets:
        ticket_id = ticket["ticket_id"]
        expected_category = ticket.get("expected_category", "unknown")
        expected_priority = ticket.get("expected_priority", "medium")
        prediction = by_ticket.get(
            ticket_id,
            {"ticket_id": ticket_id, "category": "unknown", "priority": "medium", "response": ""},
        )
        predicted_category = str(prediction.get("category", "unknown"))
        predicted_priority = str(prediction.get("priority", "medium"))

        expected_categories.append(expected_category)
        predicted_categories.append(predicted_category)
        expected_priorities.append(expected_priority)
        predicted_priorities.append(predicted_priority)
        category_scores.append(_category_score(expected_category, predicted_category))
        priority_scores.append(_priority_score(expected_priority, predicted_priority))
        response_scores.append(_response_quality(ticket, prediction, expected_category))

    per_category = metrics.get("per_category", {})
    if isinstance(per_category, dict) and per_category:
        category_labels = list(per_category.keys())
        category_values = [float(dict(value).get("avg_category_score", 0.0)) for value in per_category.values()]
        category_counts = [int(dict(value).get("count", 0)) for value in per_category.values()]
        for output_dir in resolved_output_dirs:
            _plot_bar_chart(
                category_labels,
                category_values,
                category_counts,
                title="Average Category Scores by Expected Category",
                ylabel="Average score",
                output_path=output_dir / "category_scores.png",
            )

    per_priority = metrics.get("per_priority", {})
    if isinstance(per_priority, dict) and per_priority:
        priority_labels = list(per_priority.keys())
        priority_values = [float(dict(value).get("avg_priority_score", 0.0)) for value in per_priority.values()]
        priority_counts = [int(dict(value).get("count", 0)) for value in per_priority.values()]
        for output_dir in resolved_output_dirs:
            _plot_bar_chart(
                priority_labels,
                priority_values,
                priority_counts,
                title="Average Priority Scores by Expected Priority",
                ylabel="Average score",
                output_path=output_dir / "priority_scores.png",
            )

    overall_metrics = {
        "Category Exact": float(metrics.get("category_accuracy_exact", 0.0)),
        "Category Partial": float(metrics.get("category_accuracy_with_partial_credit", 0.0)),
        "Priority Exact": float(metrics.get("priority_accuracy_exact", 0.0)),
        "Priority": float(metrics.get("priority_accuracy", 0.0)),
        "Response": float(metrics.get("response_quality_score", 0.0)),
    }
    for output_dir in resolved_output_dirs:
        _plot_overall_metrics(overall_metrics, output_dir / "overall_metric_scores.png")
        _plot_score_distributions(category_scores, priority_scores, response_scores,
                                  output_dir / "score_distributions.png")

    category_labels = _labels_in_order(expected_categories, predicted_categories)
    category_index = {label: idx for idx, label in enumerate(category_labels)}
    category_matrix = [[0 for _ in category_labels] for _ in category_labels]
    for expected, predicted in zip(expected_categories, predicted_categories):
        category_matrix[category_index[expected]][category_index[predicted]] += 1
    for output_dir in resolved_output_dirs:
        _plot_confusion_matrix(
            category_matrix,
            category_labels,
            title="Category Confusion Matrix",
            output_path=output_dir / "category_confusion_matrix.png",
        )

    priority_labels = _labels_in_order(expected_priorities, predicted_priorities, priority=True)
    priority_index = {label: idx for idx, label in enumerate(priority_labels)}
    priority_matrix = [[0 for _ in priority_labels] for _ in priority_labels]
    for expected, predicted in zip(expected_priorities, predicted_priorities):
        priority_matrix[priority_index[expected]][priority_index[predicted]] += 1
    for output_dir in resolved_output_dirs:
        _plot_confusion_matrix(
            priority_matrix,
            priority_labels,
            title="Priority Confusion Matrix",
            output_path=output_dir / "priority_confusion_matrix.png",
        )

    return list(VISUALIZATION_FILENAMES)


def render_error_analysis_visualizations(
        error_analysis: Dict[str, object],
        output_dirs: OutputPathInput,
) -> List[str]:
    resolved_output_dirs = _normalize_output_dirs(output_dirs)
    root_causes = error_analysis.get("root_causes", {})
    if isinstance(root_causes, dict) and root_causes:
        for output_dir in resolved_output_dirs:
            _plot_vertical_bar_chart(
                list(root_causes.keys()),
                [float(value) for value in root_causes.values()],
                title="Error Root Causes",
                ylabel="Ticket count",
                output_path=output_dir / "error_root_causes.png",
            )

    top_category_confusions = error_analysis.get("top_category_confusions", [])
    if isinstance(top_category_confusions, list) and top_category_confusions:
        category_labels = [
            f"{str(item.get('expected', '?'))} -> {str(item.get('predicted', '?'))}"
            for item in top_category_confusions
            if isinstance(item, dict)
        ]
        category_counts = [
            float(item.get("count", 0))
            for item in top_category_confusions
            if isinstance(item, dict)
        ]
        if category_labels:
            for output_dir in resolved_output_dirs:
                _plot_vertical_bar_chart(
                    category_labels,
                    category_counts,
                    title="Top Category Confusions",
                    ylabel="Ticket count",
                    output_path=output_dir / "error_category_confusions.png",
                )

    top_priority_confusions = error_analysis.get("top_priority_confusions", [])
    if isinstance(top_priority_confusions, list) and top_priority_confusions:
        priority_labels = [
            f"{str(item.get('expected', '?'))} -> {str(item.get('predicted', '?'))}"
            for item in top_priority_confusions
            if isinstance(item, dict)
        ]
        priority_counts = [
            float(item.get("count", 0))
            for item in top_priority_confusions
            if isinstance(item, dict)
        ]
        if priority_labels:
            for output_dir in resolved_output_dirs:
                _plot_vertical_bar_chart(
                    priority_labels,
                    priority_counts,
                    title="Top Priority Confusions",
                    ylabel="Ticket count",
                    output_path=output_dir / "error_priority_confusions.png",
                )

    flag_frequency = error_analysis.get("flag_frequency", {})
    if isinstance(flag_frequency, dict) and flag_frequency:
        top_flags = sorted(
            ((str(key), float(value)) for key, value in flag_frequency.items()),
            key=lambda item: (-item[1], item[0]),
        )[:12]
        for output_dir in resolved_output_dirs:
            _plot_bar_chart(
                [label for label, _ in top_flags],
                [value for _, value in top_flags],
                [int(value) for _, value in top_flags],
                title="Top Error Flags",
                ylabel="Flag count",
                output_path=output_dir / "error_flag_frequency.png",
            )

    return list(ERROR_ANALYSIS_FILENAMES)


def render_saved_output_visualizations(
        eval_results_path: Path,
        error_analysis_path: Path | None = None,
        output_dir: OutputPathInput | None = None,
) -> List[str]:
    eval_results = json.loads(eval_results_path.read_text(encoding="utf-8"))
    if not isinstance(eval_results, dict):
        raise ValueError(f"{eval_results_path} must contain a JSON object.")

    resolved_output_dirs = _normalize_output_dirs(output_dir or eval_results_path.parent)
    tickets_path = resolved_output_dirs[0] / "eval_set.json"
    if tickets_path.exists():
        tickets = json.loads(tickets_path.read_text(encoding="utf-8"))
    else:
        project_eval_set = Path(__file__).resolve().parents[1] / "data" / "eval_set.json"
        tickets = json.loads(project_eval_set.read_text(encoding="utf-8"))
    if not isinstance(tickets, list):
        raise ValueError("Ticket source for visualization must be a JSON array.")

    metrics = eval_results.get("metrics", {})
    predictions = eval_results.get("predictions", [])
    if not isinstance(metrics, dict):
        raise ValueError(f"{eval_results_path} is missing an object-valued 'metrics' field.")
    if not isinstance(predictions, list):
        raise ValueError(f"{eval_results_path} is missing a list-valued 'predictions' field.")

    files = render_evaluation_visualizations(
        [dict(item) for item in tickets if isinstance(item, dict)],
        [dict(item) for item in predictions if isinstance(item, dict)],
        metrics,
        resolved_output_dirs,
    )

    if error_analysis_path is not None:
        error_analysis = json.loads(error_analysis_path.read_text(encoding="utf-8"))
        if not isinstance(error_analysis, dict):
            raise ValueError(f"{error_analysis_path} must contain a JSON object.")
        files.extend(render_error_analysis_visualizations(error_analysis, resolved_output_dirs))

    return files


if __name__ == "__main__":
    outputs_path = Path("outputs")
    for output_path in outputs_path.iterdir():
        if not output_path.is_dir():
            continue
        eval_results_path = output_path / "eval_results.json"
        error_analysis_path = output_path / "error_analysis.json"
        files = render_saved_output_visualizations(
            eval_results_path=eval_results_path,
            error_analysis_path=error_analysis_path,
            output_dir=output_path,
        )
        print("Generated visualization files:")
        for file_name in files:
            print(f"  {file_name}")
