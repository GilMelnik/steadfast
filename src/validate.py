"""Stage 4: validate classifier output schema and values."""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.utils import coerce_non_negative_float, coerce_non_negative_int

ALLOWED_CATEGORIES = {
	"billing",
	"bug",
	"feature_request",
	"account",
	"integration",
	"onboarding",
	"security",
	"performance",
	"unknown",
}
ALLOWED_PRIORITIES = {"low", "medium", "high", "critical"}


def _ensure_flags(value: object) -> List[str]:
	if isinstance(value, list):
		return [str(item) for item in value]
	return []


def _sanitize_usage(value: object) -> Dict[str, object]:
	if not isinstance(value, dict):
		return {"input_tokens": 0, "output_tokens": 0, "time_taken": 0.0}
	return {
		"input_tokens": coerce_non_negative_int(value.get("input_tokens", 0)),
		"output_tokens": coerce_non_negative_int(value.get("output_tokens", 0)),
		"time_taken": round(coerce_non_negative_float(value.get("time_taken", 0.0)), 6),
	}


def validate_prediction(prediction: Dict[str, object]) -> Tuple[Dict[str, object], List[str]]:
	issues: List[str] = []

	ticket_id = str(prediction.get("ticket_id", "")).strip()
	category = str(prediction.get("category", "unknown")).strip().lower()
	priority = str(prediction.get("priority", "medium")).strip().lower()
	response = str(prediction.get("response", "")).strip()
	flags = _ensure_flags(prediction.get("flags", []))
	usage = _sanitize_usage(prediction.get("usage", {}))
	confidence_raw = prediction.get("confidence", 0.0)

	try:
		confidence = float(str(confidence_raw))
	except (TypeError, ValueError):
		confidence = 0.0
		issues.append("invalid_confidence")

	if category not in ALLOWED_CATEGORIES:
		issues.append("invalid_category")
		category = "unknown"
	if priority not in ALLOWED_PRIORITIES:
		issues.append("invalid_priority")
		priority = "medium"
	if not response:
		issues.append("empty_response")
		response = "Thanks for reporting this. We are reviewing your case and will follow up shortly."
	if not ticket_id:
		issues.append("missing_ticket_id")

	if confidence < 0.0 or confidence > 1.0:
		issues.append("confidence_out_of_range")
		confidence = max(0.0, min(confidence, 1.0))

	flags.extend(issues)
	validated = {
		"ticket_id": ticket_id,
		"category": category,
		"priority": priority,
		"response": response,
		"confidence": round(confidence, 3),
		"flags": sorted(set(flags)),
		"usage": usage,
	}
	return validated, issues
