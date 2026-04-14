"""Stage 5: rule-based corrections on top of classifier output."""

from __future__ import annotations

from typing import Dict


PRIORITIES = ["low", "medium", "high", "critical"]

def _escalate_priority(current_priority: str) -> str:
	if current_priority not in PRIORITIES:
		return "medium"
	idx = PRIORITIES.index(current_priority)
	return PRIORITIES[min(idx + 1, len(PRIORITIES) - 1)]

def _deescalate_priority(current_priority: str) -> str:
	if current_priority not in PRIORITIES:
		return "low"
	idx = PRIORITIES.index(current_priority)
	return PRIORITIES[max(idx - 1, 0)]


def apply_heuristics(
	ticket: Dict[str, str],
	prediction: Dict[str, object],
) -> Dict[str, object]:
	return prediction
