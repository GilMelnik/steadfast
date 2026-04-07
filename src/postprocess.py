"""Stage 5: rule-based corrections on top of classifier output."""

from __future__ import annotations

from typing import Dict, List

from src.preprocess import normalize_text


def _escalate_priority(current_priority: str) -> str:
	order = ["low", "medium", "high", "critical"]
	if current_priority not in order:
		return "medium"
	idx = order.index(current_priority)
	return order[min(idx + 1, len(order) - 1)]


def apply_heuristics(
	ticket: Dict[str, str],
	prediction: Dict[str, object],
	aggressive_priority: bool = False,
) -> Dict[str, object]:
	subject = ticket.get("subject", "")
	body = ticket.get("body", "")
	plan = ticket.get("plan", "")
	normalized = normalize_text(f"{subject} {body}")

	category = str(prediction.get("category", "unknown"))
	priority = str(prediction.get("priority", "medium"))
	raw_flags = prediction.get("flags", [])
	flags: List[str] = [str(flag) for flag in raw_flags] if isinstance(raw_flags, list) else []

	if "invoice" in normalized or "charged" in normalized or "refund" in normalized:
		if category != "billing":
			category = "billing"
			flags.append("rule_override_billing")

	if "sso" in normalized or "webhook" in normalized or "sync" in normalized:
		if category in {"unknown", "bug", "onboarding"}:
			category = "integration"
			flags.append("rule_override_integration")

	if "security" in normalized or "suspicious" in normalized or "mfa" in normalized:
		category = "security"
		flags.append("rule_security_category")
		if priority != "critical":
			priority = _escalate_priority(priority)
			flags.append("rule_escalate_security")

	if "cannot" in normalized or "can\'t" in normalized or "urgent" in normalized:
		priority = _escalate_priority(priority)
		flags.append("rule_escalate_blocking")

	if aggressive_priority and plan.lower() == "enterprise" and priority in {"low", "medium"}:
		priority = _escalate_priority(priority)
		flags.append("rule_escalate_enterprise")

	if "not urgent" in normalized or "whenever" in normalized:
		priority = "low"
		flags.append("rule_deescalate_non_urgent")

	updated = dict(prediction)
	updated["category"] = category
	updated["priority"] = priority
	updated["flags"] = sorted(set(flags))
	return updated
