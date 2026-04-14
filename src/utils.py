"""Shared utility helpers for pipeline stages."""

from __future__ import annotations

import re
from typing import Set

from classification_cues import STOPWORDS


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


def normalize_text(text: str) -> str:
	cleaned = text.lower().replace("\u2019", "'").replace("\u2014", "-")
	cleaned = re.sub(r"[^a-z0-9\s\-_/]", " ", cleaned)
	return normalize_whitespace(cleaned)


def normalize_whitespace(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> Set[str]:
	tokens = {tok for tok in normalize_text(text).split(" ") if tok and tok not in STOPWORDS}
	return {tok for tok in tokens if len(tok) > 2 or tok.isdigit()}
