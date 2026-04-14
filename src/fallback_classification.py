"""Rule-based fallback classification used when the LLM path is unavailable."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from src.classification_cues import (
	CRITICAL_PRIORITY_CUES,
	HIGH_PRIORITY_CUES,
	LOW_PRIORITY_CUES,
	MEDIUM_PRIORITY_CUES,
	NON_ESCALATION_CUES,
	PREVENTIVE_SECURITY_CUES,
)
from utils import normalize_text, tokenize
from src.utils import _contains_any_phrase


def guess_category(
	ticket_text: str,
	context_examples: List[Dict[str, object]],
	category_hints: Dict[str, set[str]],
	allowed_categories: set[str],
) -> str:
	category_scores: Dict[str, float] = defaultdict(float)
	normalized_text = normalize_text(ticket_text)
	ticket_tokens = tokenize(ticket_text)

	for category, hints in category_hints.items():
		for hint in hints:
			if hint in normalized_text:
				category_scores[category] += 1.5
			if hint in ticket_tokens:
				category_scores[category] += 1.0

	for idx, example in enumerate(context_examples):
		decay = 1.0 / (idx + 1)
		category_scores[str(example["category"])] += 2.0 * decay

	if not category_scores:
		return "unknown"

	best_category = max(category_scores.items(), key=lambda kv: kv[1])[0]
	return best_category if best_category in allowed_categories else "unknown"


def guess_priority(
	ticket_text: str,
	category: str,
	context_examples: List[Dict[str, object]],
	priority_hints: Dict[str, set[str]],
	allowed_priorities: set[str],
) -> str:
	priority_scores: Dict[str, float] = defaultdict(float)
	normalized_text = normalize_text(ticket_text)

	for priority, hints in priority_hints.items():
		for hint in hints:
			if hint in normalized_text:
				priority_scores[priority] += 1.2

	if _contains_any_phrase(normalized_text, LOW_PRIORITY_CUES):
		priority_scores["low"] += 2.0
	if _contains_any_phrase(normalized_text, MEDIUM_PRIORITY_CUES):
		priority_scores["medium"] += 1.8
	if _contains_any_phrase(normalized_text, HIGH_PRIORITY_CUES):
		priority_scores["high"] += 2.4
	if _contains_any_phrase(normalized_text, CRITICAL_PRIORITY_CUES):
		priority_scores["critical"] += 4.0
	if _contains_any_phrase(normalized_text, NON_ESCALATION_CUES):
		priority_scores["low"] += 1.0
		priority_scores["medium"] += 0.5
		priority_scores["high"] -= 0.8
		priority_scores["critical"] -= 1.2

	for idx, example in enumerate(context_examples):
		decay = 1.0 / (idx + 1)
		priority_scores[str(example["priority"])] += 1.1 * decay

	if category in {"feature_request", "onboarding"} and not _contains_any_phrase(
		normalized_text,
		HIGH_PRIORITY_CUES.union(CRITICAL_PRIORITY_CUES),
	):
		priority_scores["low"] += 2.5
	if category in {"billing", "account", "integration"} and _contains_any_phrase(
		normalized_text,
		{"set up", "setup", "enable", "walk me through", "learning more", "guest option", "what happens"},
	):
		priority_scores["medium"] += 1.3
		priority_scores["high"] -= 0.7
	if category == "security" and _contains_any_phrase(normalized_text, PREVENTIVE_SECURITY_CUES) and not _contains_any_phrase(
		normalized_text,
		{"session i don't recognize", "session i dont recognize", "session i do not recognize", "unauthorized access", "compromised"},
	):
		priority_scores["high"] += 1.1
		priority_scores["critical"] -= 1.5
	if _contains_any_phrase(normalized_text, {"intermittent", "sometimes", "other sections work fine"}):
		priority_scores["medium"] += 1.2
		priority_scores["high"] -= 0.6

	if "critical" in priority_scores and priority_scores["critical"] >= 2:
		return "critical"
	if not priority_scores:
		return "medium"

	best_priority = max(priority_scores.items(), key=lambda kv: kv[1])[0]
	return best_priority if best_priority in allowed_priorities else "medium"


def build_grounded_response(ticket: Dict[str, str], category: str, context_examples: List[Dict[str, object]]) -> str:
	greeting = f"Hi {ticket.get('customer_name', 'there')} - thanks for reaching out."
	if not context_examples:
		return f"{greeting} We are reviewing this now and will follow up with concrete next steps shortly."

	best = context_examples[0]
	anchor = best.get("resolution_summary", "")
	subject = ticket.get("subject", "your issue")

	if category == "integration":
		next_step = "Please re-authorize the integration in Settings > Integrations and then run a manual resync."
	elif category == "security":
		next_step = "As an immediate safeguard, rotate admin credentials, enforce MFA, and review active sessions."
	elif category == "performance":
		next_step = "Please share the affected project/workspace IDs and timestamp so we can profile the slow queries."
	elif category == "billing":
		next_step = "Please share the invoice ID and charge date so we can verify line-item details and apply corrections if needed."
	elif category == "onboarding":
		next_step = "We can share the relevant setup guide and suggest a quick rollout plan tailored to your workspace."
	elif category == "feature_request":
		next_step = "I've logged this with our product team and can attach your use case to the request for prioritization."
	elif category == "account":
		next_step = "We can guide you through the admin settings path and complete verification steps if an ownership change is involved."
	else:
		next_step = "Please include a screenshot or exact error text so we can narrow this down quickly."

	return (
		f"{greeting} For '{subject}', this looks consistent with a known {category.replace('_', ' ')} pattern we've handled before. "
		f"Similar prior fix context: {anchor}. {next_step}"
	)


def classify_with_fallback(
	ticket: Dict[str, str],
	context_examples: List[Dict[str, object]],
	category_hints: Dict[str, set[str]],
	priority_hints: Dict[str, set[str]],
	allowed_categories: set[str],
	allowed_priorities: set[str],
	llm_error_flag: Optional[str],
) -> Dict[str, object]:
	ticket_text = " ".join((ticket.get("subject", ""), ticket.get("body", "")))
	category = guess_category(ticket_text, context_examples, category_hints, allowed_categories)
	priority = guess_priority(ticket_text, category, context_examples, priority_hints, allowed_priorities)
	response = build_grounded_response(ticket, category, context_examples)

	confidence = 0.45 + 0.1 * min(len(context_examples), 4)
	if context_examples:
		confidence += 0.1 if context_examples[0]["category"] == category else 0.0
	confidence = max(0.0, min(round(confidence, 2), 0.98))

	return {
		"ticket_id": ticket.get("ticket_id", ""),
		"category": category,
		"priority": priority,
		"response": response,
		"confidence": confidence,
		"flags": [llm_error_flag] if llm_error_flag else [],
		"context_ticket_ids": [example.get("ticket_id", "") for example in context_examples],
		"usage": {
			"input_tokens": 0,
			"output_tokens": 0,
		},
	}
