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
DEFAULT_CONTEXT_RERANK_CANDIDATES = 6


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
			messages=cast(Any, messages),
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
	ticket_text = normalize_text(subject + " " + body)
	if not ticket_tokens or not example_tokens:
		overlap = 0.0
	else:
		overlap = len(ticket_tokens.intersection(example_tokens)) / max(len(ticket_tokens), 1)
	semantic_ratio = SequenceMatcher(None, ticket_text, " ".join(sorted(example_tokens))).ratio()
	return 0.65 * overlap + 0.35 * semantic_ratio


def _bm25_like_score(query_tokens: set[str], example_tokens: set[str], idf: Dict[str, float], avg_doc_len: float) -> float:
	if not query_tokens or not example_tokens:
		return 0.0
	k1 = 1.5
	b = 0.75
	doc_len = max(len(example_tokens), 1)
	normalization = 1.0 - b + b * (doc_len / max(avg_doc_len, 1.0))
	score = 0.0
	for token in query_tokens.intersection(example_tokens):
		idf_value = float(idf.get(token, 0.35))
		tf = 1.0
		score += idf_value * ((tf * (k1 + 1.0)) / (tf + (k1 * normalization)))
	return score


def _intent_signal_score(ticket_text: str, example: Dict[str, object]) -> float:
	intent_tokens = tokenize(ticket_text)
	if not intent_tokens:
		return 0.0

	category = str(example.get("category", ""))
	priority = str(example.get("priority", ""))
	score = 0.0

	for hint in CATEGORY_HINTS.get(category, set()):
		if hint in ticket_text:
			score += 0.22

	for hint in PRIORITY_HINTS.get(priority, set()):
		if hint in ticket_text:
			score += 0.14

	common = len(intent_tokens.intersection(cast(set[str], example.get("tokens", set()))))
	if common >= 3:
		score += 0.2
	return score


def _semantic_score(ticket_text: str, example: Dict[str, object]) -> float:
	example_text = str(example.get("search_text", ""))
	if not example_text:
		example_text = normalize_text(
			" ".join(
				(
					str(example.get("subject", "")),
					str(example.get("body", "")),
					str(example.get("resolution", "")),
				)
			)
		)
	return SequenceMatcher(None, ticket_text, example_text).ratio()


def _rank_context_examples(
	ticket: Dict[str, str],
	kb_index: Dict[str, object],
	candidate_pool_size: int,
) -> List[Tuple[float, Dict[str, object]]]:
	ticket_text = normalize_text(" ".join((ticket.get("subject", ""), ticket.get("body", ""))))
	ticket_tokens = tokenize(ticket_text)

	idf = cast(Dict[str, float], kb_index.get("idf", {}))
	avg_doc_len_value = kb_index.get("avg_doc_len", 1.0)
	try:
		avg_doc_len = float(str(avg_doc_len_value))
	except (TypeError, ValueError):
		avg_doc_len = 1.0
	avg_doc_len = max(avg_doc_len, 1.0)
	scored: List[Tuple[float, Dict[str, object]]] = []
	examples = cast(List[Dict[str, Any]], kb_index.get("examples", []))

	for example_any in examples:
		example = cast(Dict[str, object], example_any)
		example_tokens = cast(set[str], example.get("tokens", set()))
		lexical = _bm25_like_score(ticket_tokens, example_tokens, idf, avg_doc_len)
		semantic = _semantic_score(ticket_text, example)
		overlap = _score_overlap(ticket_tokens, example_tokens, ticket.get("subject", ""), ticket.get("body", ""))
		intent = _intent_signal_score(ticket_text, example)
		score = (0.44 * lexical) + (0.24 * overlap) + (0.22 * semantic) + (0.10 * intent)
		scored.append((score, example))

	scored.sort(key=lambda pair: pair[0], reverse=True)
	return scored[:candidate_pool_size]


def _is_context_rerank_enabled() -> bool:
	flag = (os.getenv("LSP_CONTEXT_RERANK") or "").strip().lower()
	return flag in {"1", "true", "yes", "on"}


def _make_context_rerank_messages(ticket: Dict[str, str], candidates: List[Dict[str, object]]) -> List[Dict[str, str]]:
	candidate_lines: List[str] = []
	for candidate in candidates:
		candidate_lines.append(
			" | ".join(
				[
					f"ticket_id={candidate.get('ticket_id', '')}",
					f"category={candidate.get('category', '')}",
					f"priority={candidate.get('priority', '')}",
					f"subject={candidate.get('subject', '')}",
					f"resolution={candidate.get('resolution_summary', '')}",
				]
			)
		)

	system_prompt = (
		"You rank historical support examples by relevance for grounding a response. "
		"Return strict JSON only with key ranked_ticket_ids as an array of ticket IDs in descending relevance."
	)
	user_prompt = (
		f"Ticket subject: {ticket.get('subject', '')}\n"
		f"Ticket body: {ticket.get('body', '')}\n\n"
		"Candidates:\n"
		+ "\n".join(candidate_lines)
	)
	return [
		{"role": "user", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]


def _rerank_with_llm(
	ticket: Dict[str, str],
	candidates: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], Optional[str]]:
	if not candidates:
		return candidates, None

	config = _get_llm_config()
	if not config["api_key"]:
		return candidates, "context_rerank_skipped_missing_api_key"

	try:
		from openai import OpenAI
	except ImportError:
		return candidates, "context_rerank_skipped_missing_openai_dependency"

	try:
		client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
		response = client.chat.completions.create(
			model=config["model"],
			messages=cast(Any, _make_context_rerank_messages(ticket, candidates)),
			temperature=0.0,
		)
	except Exception as e:
		return candidates, f"context_rerank_failed: {e}"

	content = ""
	if response.choices and response.choices[0].message:
		content = response.choices[0].message.content or ""

	payload = _extract_json_payload(content)
	if payload is None:
		return candidates, "context_rerank_invalid_json"

	ranked_ids_raw = payload.get("ranked_ticket_ids", [])
	if not isinstance(ranked_ids_raw, list):
		return candidates, "context_rerank_invalid_payload"

	ranked_ids = [str(item) for item in ranked_ids_raw]
	if not ranked_ids:
		return candidates, "context_rerank_empty_ranking"

	by_id = {str(candidate.get("ticket_id", "")): candidate for candidate in candidates}
	reordered: List[Dict[str, object]] = []
	seen = set()
	for ticket_id in ranked_ids:
		candidate = by_id.get(ticket_id)
		if candidate is None:
			continue
		reordered.append(candidate)
		seen.add(ticket_id)

	for candidate_any in candidates:
		candidate = cast(Dict[str, object], candidate_any)
		ticket_id = str(candidate.get("ticket_id", ""))
		if ticket_id in seen:
			continue
		reordered.append(candidate)

	return reordered, None


def retrieve_context(ticket: Dict[str, str], kb_index: Dict[str, object], top_k: int = 5) -> List[Dict[str, object]]:
	top_k = max(top_k, 1)
	candidate_pool_size = max(top_k * 3, DEFAULT_CONTEXT_RERANK_CANDIDATES)
	ranked = _rank_context_examples(ticket, kb_index, candidate_pool_size=candidate_pool_size)
	selected = [example for _, example in ranked]

	if _is_context_rerank_enabled():
		rerank_window = selected[:DEFAULT_CONTEXT_RERANK_CANDIDATES]
		reranked, _ = _rerank_with_llm(ticket, rerank_window)
		selected = reranked + selected[DEFAULT_CONTEXT_RERANK_CANDIDATES:]

	return selected[:top_k]


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
	usage_any = result.get("usage", {})
	usage = cast(Dict[str, object], usage_any if isinstance(usage_any, dict) else {})
	usage["time_taken"] = time_taken
	result["usage"] = usage
	return result
