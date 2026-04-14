"""Stage 5: rule-based corrections on top of classifier output."""

from __future__ import annotations

from typing import Dict, List

from src.classification_cues import (
	CONFIRMED_SECURITY_CUES,
	DATA_LOSS_CUES,
	HIGH_PRIORITY_CUES,
	LOW_PRIORITY_CUES,
	MEDIUM_PRIORITY_CUES,
	PREVENTIVE_SECURITY_CUES,
)
from utils import normalize_text
from src.utils import _contains_any_phrase

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
	final_prediction = dict(prediction)
	text = normalize_text(" ".join((ticket.get("subject", ""), ticket.get("body", ""))))
	category = str(final_prediction.get("category", "")).strip().lower()
	priority = str(final_prediction.get("priority", "medium")).strip().lower()

	has_data_loss = _contains_any_phrase(text, DATA_LOSS_CUES)
	has_confirmed_security_issue = _contains_any_phrase(text, CONFIRMED_SECURITY_CUES)
	has_high_impact = _contains_any_phrase(text, HIGH_PRIORITY_CUES)
	has_medium_cap_signal = _contains_any_phrase(text, MEDIUM_PRIORITY_CUES)
	has_low_signal = _contains_any_phrase(text, LOW_PRIORITY_CUES)

	if has_data_loss or has_confirmed_security_issue:
		if priority != "critical":
			final_prediction["priority"] = "critical"
			_append_flag(final_prediction, "priority_postprocess_escalated_to_critical")
		return final_prediction

	if category == "security" and priority == "critical":
		if _contains_any_phrase(text, PREVENTIVE_SECURITY_CUES) and not has_confirmed_security_issue:
			final_prediction["priority"] = "high"
			_append_flag(final_prediction, "priority_postprocess_deescalated_to_high")
			return final_prediction

	if category in {"feature_request", "onboarding"} and priority != "low":
		if not has_high_impact and not has_data_loss and not has_confirmed_security_issue:
			final_prediction["priority"] = "low"
			_append_flag(final_prediction, "priority_postprocess_capped_to_low")
			return final_prediction

	if priority in {"high", "critical"} and has_low_signal and not has_high_impact:
		final_prediction["priority"] = "low"
		_append_flag(final_prediction, "priority_postprocess_deescalated_to_low")
		return final_prediction

	if priority in {"high", "critical"} and has_medium_cap_signal and not has_high_impact:
		final_prediction["priority"] = "medium"
		_append_flag(final_prediction, "priority_postprocess_capped_to_medium")
		return final_prediction

	if category in {"account", "billing", "integration"} and priority in {"high", "critical"}:
		if _contains_any_phrase(text, {"set up", "setup", "enable", "walk me through", "learning more", "what happens", "guest option"}) and not has_high_impact:
			final_prediction["priority"] = "medium"
			_append_flag(final_prediction, "priority_postprocess_deescalated_to_medium")

	return final_prediction
