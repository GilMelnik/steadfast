"""Stage 6: evaluate predictions against expected labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

from src.preprocess import tokenize
from src.utils import coerce_non_negative_float, coerce_non_negative_int

PARTIAL_CATEGORY_MATCHES = {
	("bug", "performance"),
	("performance", "bug"),
	("onboarding", "account"),
	("account", "onboarding"),
	("integration", "onboarding"),
	("onboarding", "integration"),
}

PRIORITY_ORDER = ("low", "medium", "high", "critical")


def _format_hh_mm_ss(seconds: float) -> str:
	total_seconds = int(round(max(seconds, 0.0)))
	hours, remainder = divmod(total_seconds, 3600)
	minutes, secs = divmod(remainder, 60)
	return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _category_score(expected: str, predicted: str) -> float:
	if expected == predicted:
		return 1.0
	if (expected, predicted) in PARTIAL_CATEGORY_MATCHES:
		return 0.5
	return 0.0


def _priority_rank(priority: str) -> int:
	"""Map a priority label to its ordinal position, defaulting unknown labels to medium."""
	if priority in PRIORITY_ORDER:
		return PRIORITY_ORDER.index(priority)
	return PRIORITY_ORDER.index("medium")


def _priority_score(expected: str, predicted: str) -> float:
	"""Return a normalized distance score where farther priority mismatches get less credit."""
	max_distance = len(PRIORITY_ORDER) - 1
	distance = abs(_priority_rank(expected) - _priority_rank(predicted))
	return round(1.0 - (distance / max_distance), 4)


def _response_quality(ticket: Dict[str, str], prediction: Dict[str, object], expected_category: str) -> float:
	response = str(prediction.get("response", "")).strip()
	if not response:
		return 0.0

	ticket_tokens = tokenize(" ".join((ticket.get("subject", ""), ticket.get("body", ""))))
	response_tokens = tokenize(response)
	overlap = len(ticket_tokens.intersection(response_tokens)) / max(len(ticket_tokens), 1)

	has_action_step = any(
		marker in response.lower()
		for marker in (
			"please",
			"settings >",
			"next",
			"share",
			"enable",
			"resync",
			"guide",
		)
	)
	category_alignment = 1.0 if prediction.get("category") == expected_category else 0.65

	score = 0.45 * min(1.0, overlap * 3.0) + 0.25 * float(has_action_step) + 0.30 * category_alignment
	return round(max(0.0, min(score, 1.0)), 4)


def evaluate_predictions(tickets: List[Dict[str, str]], predictions: List[Dict[str, object]]) -> Dict[str, object]:
	by_ticket = {pred["ticket_id"]: pred for pred in predictions}

	per_category = defaultdict(lambda: {"count": 0, "category_score_sum": 0.0})
	per_priority = defaultdict(lambda: {"count": 0, "score_sum": 0.0})
	confusion = Counter()

	category_scores: List[float] = []
	priority_scores: List[float] = []
	priority_exact_hits = 0
	response_scores: List[float] = []
	total_input_tokens = 0
	total_output_tokens = 0
	total_time_taken_seconds = 0.0

	for ticket in tickets:
		ticket_id = ticket["ticket_id"]
		expected_category = ticket.get("expected_category", "unknown")
		expected_priority = ticket.get("expected_priority", "medium")
		pred = by_ticket.get(ticket_id, {"ticket_id": ticket_id, "category": "unknown", "priority": "medium", "response": ""})

		category_score = _category_score(expected_category, str(pred.get("category", "unknown")))
		category_scores.append(category_score)

		priority_score = _priority_score(expected_priority, str(pred.get("priority", "medium")))
		priority_scores.append(priority_score)
		priority_exact_hits += int(str(pred.get("priority", "medium")) == expected_priority)

		response_score = _response_quality(ticket, pred, expected_category)
		response_scores.append(response_score)

		usage = pred.get("usage", {})
		if not isinstance(usage, dict):
			usage = {}
		total_input_tokens += coerce_non_negative_int(usage.get("input_tokens", 0))
		total_output_tokens += coerce_non_negative_int(usage.get("output_tokens", 0))
		total_time_taken_seconds += coerce_non_negative_float(usage.get("time_taken", 0.0))

		per_category[expected_category]["count"] += 1
		per_category[expected_category]["category_score_sum"] += category_score

		per_priority[expected_priority]["count"] += 1
		per_priority[expected_priority]["score_sum"] += priority_score

		confusion[(expected_category, str(pred.get("category", "unknown")))] += 1

	total = max(len(tickets), 1)
	category_accuracy = sum(1 for score in category_scores if score == 1.0) / total
	category_accuracy_with_partial = sum(category_scores) / total
	priority_accuracy_exact = priority_exact_hits / total
	priority_accuracy = sum(priority_scores) / total
	response_quality = sum(response_scores) / total
	avg_input_tokens = total_input_tokens / total
	avg_output_tokens = total_output_tokens / total
	avg_time_taken_seconds = total_time_taken_seconds / total

	category_breakdown = {
		key: {
			"count": value["count"],
			"avg_category_score": round(value["category_score_sum"] / max(value["count"], 1), 4),
		}
		for key, value in sorted(per_category.items())
	}

	priority_breakdown = {
		key: {
			"count": value["count"],
			"avg_priority_score": round(value["score_sum"] / max(value["count"], 1), 4),
		}
		for key, value in sorted(per_priority.items())
	}

	top_confusions = [
		{
			"expected": expected,
			"predicted": predicted,
			"count": count,
		}
		for (expected, predicted), count in confusion.most_common(10)
		if expected != predicted
	]

	return {
		"num_tickets": len(tickets),
		"category_accuracy_exact": round(category_accuracy, 4),
		"category_accuracy_with_partial_credit": round(category_accuracy_with_partial, 4),
		"priority_accuracy_exact": round(priority_accuracy_exact, 4),
		"priority_accuracy": round(priority_accuracy, 4),
		"response_quality_score": round(response_quality, 4),
		"per_category": category_breakdown,
		"per_priority": priority_breakdown,
		"top_confusions": top_confusions,
		"resource_usage": {
			"total_input_tokens": total_input_tokens,
			"total_output_tokens": total_output_tokens,
			"total_time_seconds": round(total_time_taken_seconds, 4),
			"total_time_hh_mm_ss": _format_hh_mm_ss(total_time_taken_seconds),
			"avg_input_tokens_per_ticket": round(avg_input_tokens, 4),
			"avg_output_tokens_per_ticket": round(avg_output_tokens, 4),
			"avg_time_seconds_per_ticket": round(avg_time_taken_seconds, 4),
			"avg_time_hh_mm_ss_per_ticket": _format_hh_mm_ss(avg_time_taken_seconds),
		},
	}
