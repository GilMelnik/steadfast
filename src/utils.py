"""Shared utility helpers for pipeline stages."""

from __future__ import annotations


def coerce_non_negative_int(value: object) -> int:
	"""Convert an arbitrary value to a non-negative integer."""
	try:
		parsed = int(float(str(value)))
	except (TypeError, ValueError):
		return 0
	return max(parsed, 0)


def coerce_non_negative_float(value: object) -> float:
	"""Convert an arbitrary value to a non-negative float."""
	try:
		parsed = float(str(value))
	except (TypeError, ValueError):
		return 0.0
	return max(parsed, 0.0)


def _contains_any_phrase(text: str, phrases: set[str]) -> bool:
	return any(phrase in text for phrase in phrases)
