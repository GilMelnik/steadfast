"""Stage 3: ticket classification and response generation."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, cast

from src.preprocess import normalize_text, tokenize

ALLOWED_CATEGORIES = {
	"billing",
	"bug",
	"feature_request",
	"account",
	"integration",
	"onboarding",
	"security",
	"performance",
}
ALLOWED_PRIORITIES = {"low", "medium", "high", "critical"}

CATEGORY_HINTS = {
	"billing": {"invoice", "charged", "refund", "credit", "payment", "billing", "seat"},
	"bug": {"broken", "error", "not working", "fails", "stale", "wrong", "issue", "bug"},
	"feature_request": {"feature", "request", "would love", "wish", "add support", "ability"},
	"account": {"account", "owner", "role", "permission", "users", "workspace", "deactivate"},
	"integration": {"api", "webhook", "sso", "slack", "okta", "jira", "github", "sync"},
	"onboarding": {"how to", "setup", "set up", "getting started", "guide", "best practices"},
	"security": {"security", "mfa", "2fa", "suspicious", "breach", "allowlist", "compromised"},
	"performance": {"slow", "lag", "timeout", "504", "performance", "sluggish", "latency"},
}

PRIORITY_HINTS = {
	"critical": {"urgent", "production down", "can't log in", "data loss", "security", "504"},
	"high": {"asap", "important", "blocking", "cannot", "team affected", "stopped"},
	"medium": {"inconsistent", "intermittent", "confusing", "question", "today"},
	"low": {"not urgent", "whenever", "nice to have", "would like", "quick question"},
}

DEFAULT_LLM_BASE_URL = "https://lsp-proxy.cave.latent.build/v1"
DEFAULT_LLM_MODEL = "claude-sonnet-4-6"


def _get_llm_config() -> Dict[str, str]:
	api_key = os.getenv("LSP_API_KEY") or os.getenv("API_KEY") or ""
	base_url = os.getenv("LSP_API_BASE") or os.getenv("BASE_URL") or DEFAULT_LLM_BASE_URL
	model = os.getenv("LSP_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_LLM_MODEL
	return {
		"api_key": api_key,
		"base_url": base_url,
		"model": model,
	}


def _make_llm_messages(ticket: Dict[str, str], context_examples: List[Dict[str, object]]) -> List[Dict[str, str]]:
	context_lines: List[str] = []
	for example in context_examples[:4]:
		context_lines.append(
			" | ".join(
				[
					f"ticket_id={example.get('ticket_id', '')}",
					f"category={example.get('category', '')}",
					f"priority={example.get('priority', '')}",
					f"subject={example.get('subject', '')}",
					f"resolution={example.get('resolution_summary', '')}",
				]
			)
		)

	allowed_categories = ", ".join(sorted(ALLOWED_CATEGORIES))
	allowed_priorities = ", ".join(sorted(ALLOWED_PRIORITIES))

	system_prompt = (
		"You are a Steadfast support triage assistant. "
		"Use the provided KB context to classify the ticket and draft a grounded, actionable response. "
		"Return strict JSON only with keys: category, priority, response, confidence, flags. "
		f"category must be one of [{allowed_categories}]. "
		f"priority must be one of [{allowed_priorities}]. "
		"confidence must be a number between 0 and 1. flags must be an array of strings."
	)

	user_prompt = (
		f"Ticket ID: {ticket.get('ticket_id', '')}\n"
		f"Customer: {ticket.get('customer_name', '')}\n"
		f"Plan: {ticket.get('plan', '')}\n"
		f"Subject: {ticket.get('subject', '')}\n"
		f"Body: {ticket.get('body', '')}\n\n"
		"Nearest knowledge base examples:\n"
		+ "\n".join(context_lines)
	)

	return [
		{"role": "user", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]


def _extract_json_payload(raw_text: str) -> Optional[Dict[str, object]]:
	text = raw_text.strip()
	if not text:
		return None
	try:
		obj = json.loads(text)
		return obj if isinstance(obj, dict) else None
	except json.JSONDecodeError:
		left = text.find("{")
		right = text.rfind("}")
		if left == -1 or right == -1 or right <= left:
			return None
		try:
			obj = json.loads(text[left : right + 1])
			return obj if isinstance(obj, dict) else None
		except json.JSONDecodeError:
			return None


def _classify_with_llm(ticket: Dict[str, str], context_examples: List[Dict[str, object]]) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
	config = _get_llm_config()
	if not config["api_key"]:
		return None, "llm_skipped_missing_api_key"

	try:
		from openai import OpenAI
	except ImportError:
		return None, "llm_skipped_missing_openai_dependency"

	messages = _make_llm_messages(ticket, context_examples)
	try:
		client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
		response = client.chat.completions.create(
			model=config["model"],
			messages=messages,
			temperature=0.1,
		)
	except Exception as e:
		return None, f"llm_call_failed: {e}"

	content = ""
	usage = {}

	if response.choices and response.choices[0].message:
		content = response.choices[0].message.content or ""
	elif hasattr(response, "model_extra") and isinstance(response.model_extra, dict):
		content = response.model_extra['content'][0]['text']

	if hasattr(response, "usage") and hasattr(response.usage, "model_extra") and \
		isinstance(response.usage.model_extra, dict):

		usage = {
			'input_tokens': response.usage.model_extra.get('input_tokens', 0),
			'output_tokens': response.usage.model_extra.get('output_tokens', 0),
		}
	payload = _extract_json_payload(content)
	if payload is None:
		return None, "llm_invalid_json"

	category = str(payload.get("category", "unknown")).strip().lower()
	priority = str(payload.get("priority", "medium")).strip().lower()
	response_text = str(payload.get("response", "")).strip()
	confidence_raw = payload.get("confidence", 0.75)
	try:
		confidence = float(str(confidence_raw))
	except (TypeError, ValueError):
		confidence = 0.75

	if category not in ALLOWED_CATEGORIES:
		return None, "llm_invalid_category"
	if priority not in ALLOWED_PRIORITIES:
		return None, "llm_invalid_priority"
	if not response_text:
		return None, "llm_empty_response"

	raw_flags = payload.get("flags", [])
	flags = [str(flag) for flag in raw_flags] if isinstance(raw_flags, list) else []
	return (
		{
			"ticket_id": ticket.get("ticket_id", ""),
			"category": category,
			"priority": priority,
			"response": response_text,
			"confidence": max(0.0, min(confidence, 1.0)),
			"flags": sorted(set(flags + ["llm_used"])),
			"usage": usage,
		},
		None,
	)


def _score_overlap(ticket_tokens: set[str], example_tokens: set[str], subject: str, body: str) -> float:
	if not ticket_tokens or not example_tokens:
		overlap = 0.0
	else:
		overlap = len(ticket_tokens.intersection(example_tokens)) / max(len(ticket_tokens), 1)
	semantic_ratio = SequenceMatcher(None, normalize_text(subject + " " + body), " ".join(example_tokens)).ratio()
	return 0.75 * overlap + 0.25 * semantic_ratio


def retrieve_context(ticket: Dict[str, str], kb_index: Dict[str, object], top_k: int = 4) -> List[Dict[str, object]]:
	ticket_tokens = tokenize(" ".join((ticket.get("subject", ""), ticket.get("body", ""))))
	scored: List[Tuple[float, Dict[str, object]]] = []
	examples = cast(List[Dict[str, Any]], kb_index.get("examples", []))
	for example_any in examples:
		example = cast(Dict[str, object], example_any)
		score = _score_overlap(
			ticket_tokens,
			cast(set[str], example["tokens"]),
			ticket.get("subject", ""),
			ticket.get("body", ""),
		)
		scored.append((score, example))
	scored.sort(key=lambda pair: pair[0], reverse=True)
	return [example for _, example in scored[:top_k]]


def _guess_category(ticket_text: str, context_examples: List[Dict[str, object]]) -> str:
	category_scores: Dict[str, float] = defaultdict(float)
	normalized_text = normalize_text(ticket_text)
	ticket_tokens = tokenize(ticket_text)

	for category, hints in CATEGORY_HINTS.items():
		for hint in hints:
			if hint in normalized_text:
				category_scores[category] += 1.5
			if hint in ticket_tokens:
				category_scores[category] += 1.0

	for idx, ex in enumerate(context_examples):
		decay = 1.0 / (idx + 1)
		category_scores[str(ex["category"])] += 2.0 * decay

	if not category_scores:
		return "unknown"
	best_category = max(category_scores.items(), key=lambda kv: kv[1])[0]
	return best_category if best_category in ALLOWED_CATEGORIES else "unknown"


def _guess_priority(ticket_text: str, context_examples: List[Dict[str, object]]) -> str:
	priority_scores: Dict[str, float] = defaultdict(float)
	normalized_text = normalize_text(ticket_text)

	for priority, hints in PRIORITY_HINTS.items():
		for hint in hints:
			if hint in normalized_text:
				priority_scores[priority] += 1.2

	for idx, ex in enumerate(context_examples):
		decay = 1.0 / (idx + 1)
		priority_scores[str(ex["priority"])] += 1.6 * decay

	if "critical" in priority_scores and priority_scores["critical"] >= 2:
		return "critical"
	if not priority_scores:
		return "medium"
	best_priority = max(priority_scores.items(), key=lambda kv: kv[1])[0]
	return best_priority if best_priority in ALLOWED_PRIORITIES else "medium"


def _build_grounded_response(ticket: Dict[str, str], category: str, context_examples: List[Dict[str, object]]) -> str:
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


def _classify_ticket(ticket: Dict[str, str], kb_index: Dict[str, object]) -> Dict[str, object]:
	context_examples = retrieve_context(ticket, kb_index)
	llm_result, llm_error_flag = _classify_with_llm(ticket, context_examples)
	if llm_result is not None:
		llm_result["context_ticket_ids"] = [example.get("ticket_id", "") for example in context_examples]
		return llm_result

	ticket_text = " ".join((ticket.get("subject", ""), ticket.get("body", "")))

	category = _guess_category(ticket_text, context_examples)
	priority = _guess_priority(ticket_text, context_examples)
	response = _build_grounded_response(ticket, category, context_examples)

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
			'input_tokens': 0,
			'output_tokens': 0,
		}
	}


def classify_ticket(ticket: Dict[str, str], kb_index: Dict[str, object]) -> Dict[str, object]:
	start_time = time.time()
	result = _classify_ticket(ticket, kb_index)
	end_time = time.time()
	time_taken = end_time - start_time
	usage = result.get("usage", {})
	usage["time_taken"] = time_taken
	return result
