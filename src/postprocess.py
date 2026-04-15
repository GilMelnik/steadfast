"""Stage 5: rule-based corrections on top of classifier output."""

from __future__ import annotations

from typing import Dict, List


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


def _append_flag(prediction: Dict[str, object], flag: str) -> None:
	flags_raw = prediction.get("flags", [])
	flags: List[str] = [str(item) for item in flags_raw] if isinstance(flags_raw, list) else []
	if flag not in flags:
		flags.append(flag)
	prediction["flags"] = sorted(flags)


def apply_heuristics(
	ticket: Dict[str, str],
	prediction: Dict[str, object],
) -> Dict[str, object]:
	return prediction
