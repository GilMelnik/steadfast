"""Stage 3: ticket classification and response generation."""

from __future__ import annotations

import copy
import json
import os
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from src.fallback_classification import classify_with_fallback
from utils import normalize_text, tokenize

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
    "billing": {"invoice", "charged", "refund", "credit", "payment", "billing", "seat", "tax", "address", "downgrade"},
    "bug": {"broken", "error", "not working", "fails", "stale", "wrong", "issue", "bug", "glitch", "incorrect"},
    "feature_request": {"feature", "request", "would love", "wish", "add support", "ability", "improvement",
                        "suggestion"},
    "account": {"account", "owner", "role", "permission", "users", "workspace", "deactivate", "export", "transfer",
                "ownership", "profile"},
    "integration": {"api", "webhook", "sso", "slack", "okta", "jira", "github", "sync", "connection", "token"},
    "onboarding": {"how to", "setup", "set up", "getting started", "guide", "best practices", "migration", "import",
                   "first time", "onboard"},
    "security": {"security", "mfa", "2fa", "suspicious", "breach", "allowlist", "compromised", "unauthorized",
                 "compliance", "audit", "rbac"},
    "performance": {"slow", "lag", "timeout", "504", "performance", "sluggish", "latency", "freezes", "loading"},
}

PRIORITY_HINTS = {
    "critical": {"urgent", "production down", "can't log in", "data loss", "security", "504", "board meeting",
                 "immediate", "now", "breach"},
    "high": {"asap", "important", "blocking", "cannot", "team affected", "stopped", "enterprise", "migration",
             "ownership", "admin", "today", "deadline"},
    "medium": {"inconsistent", "intermittent", "confusing", "question", "workaround", "glitch", "minor"},
    "low": {"not urgent", "whenever", "nice to have", "would like", "quick question", "future", "cosmetic"},
}

DEFAULT_LLM_BASE_URL = "https://lsp-proxy.cave.latent.build/v1"
# DEFAULT_LLM_MODEL = "claude-sonnet-4-6"
DEFAULT_LLM_MODEL = "claude-opus-4-6"

LLM_OUTPUTS_FILENAME = "llm_outputs.jsonl"


def append_llm_output_record(output_dir: Path, record: Dict[str, object]) -> None:
    """Append one JSON object as a line to the run's LLM log file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / LLM_OUTPUTS_FILENAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_llm_outputs_jsonl(path: Path) -> Dict[str, Dict[str, object]]:
    """
    Load a JSONL file of LLM records; later lines override earlier ones per ticket_id.
    `path` may be a file or a directory containing llm_outputs.jsonl.
    """
    if path.is_dir():
        path = path / LLM_OUTPUTS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"LLM outputs not found: {path}")

    by_ticket: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            tid = str(obj.get("ticket_id", "")).strip()
            if tid:
                by_ticket[tid] = cast(Dict[str, object], obj)
    return by_ticket


def _get_llm_config() -> Dict[str, str]:
    api_key = os.getenv("LSP_API_KEY") or os.getenv("API_KEY") or ""
    base_url = os.getenv("LSP_API_BASE") or os.getenv("BASE_URL") or DEFAULT_LLM_BASE_URL
    model = os.getenv("LSP_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_LLM_MODEL
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


def _make_llm_message(ticket: Dict[str, str], context_examples: List[Dict[str, object]]) -> Dict[str, str]:
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

    category_guidance = (
        "- billing: Invoices, charges, refunds, payments, billing cycles, seats/licenses, plan changes/downgrades, and billing profile updates (address, tax info).\n"
        "- bug: Broken features, errors, unexpected behavior, UI glitches, 'not working', or functional regressions.\n"
        "- feature_request: Suggestions for new functionality, 'would love to see', missing features, or UI improvements.\n"
        "- account: User management, roles, permissions, workspace settings, data exports, ownership transfers, and login issues (non-security).\n"
        "- integration: API, webhooks, third-party apps (Slack, Jira, Github), SSO (Okta), and sync failures.\n"
        "- onboarding: Setup help for new teams, 'how to' questions, best practices, getting started, and initial migrations from other tools.\n"
        "- security: MFA/2FA, suspicious activity, data breaches, unauthorized access, allowlisting, and security compliance (RBAC, audits).\n"
        "- performance: Slowness, lag, timeouts, high latency, 504 errors, and UI freezing."
    )

    priority_guidance = (
        "- critical: Production down, data loss (including silent failures), security breach, total login failure, 504 errors, or business-critical deadlines (e.g., 'board meeting', 'immediate audit').\n"
        "- high: Workflow-blocking issues where a user/team cannot proceed, major feature failures, sensitive administrative changes (ownership, billing), onboarding/migration for new teams, or issues affecting Enterprise account productivity.\n"
        "- medium: Intermittent bugs, confusing behavior, feature requests with defined but non-urgent deadlines, or issues affecting single users with available workarounds.\n"
        "- low: General 'how-to' questions, minor cosmetic issues, suggestions for future features, or 'nice to have' requests with no immediate urgency."
    )

    message = (
        "You are a senior Steadfast support triage assistant. Your goal is to accurately classify tickets and draft initial responses.\n\n"
        "### Classification Guidelines\n"
        "Categories:\n"
        f"{category_guidance}\n\n"
        "Priorities:\n"
        f"{priority_guidance}\n\n"
        "### Strategic Escalation Rules\n"
        "1. **Enterprise Impact:** Issues affecting multiple users or core workflows on an Enterprise plan should be escalated to High or Critical.\n"
        "2. **Onboarding/Migration:** Initial setup and data migrations are time-sensitive blockers for new customers; treat these as High priority.\n"
        "3. **Sensitive Operations:** Ownership transfers, billing updates, and security configurations are high-stakes and should be High priority.\n"
        "4. **Silent Failures:** Issues where the system fails without an error message (e.g., data disappearing, silent sync failures) are high risk and should be at least High.\n\n"
        "### Grounding Context\n"
        "Use these historical examples to identify Steadfast product details, known issues, and appropriate resolution summaries.\n"
        "Nearest knowledge base examples:\n"
    )

    if context_lines:
        message += "\n".join(context_lines)
    else:
        message += "(none)"

    message += (
        "\n\n### Instructions\n"
        "1. Analyze the ticket subject and body, paying close attention to the customer's plan and any urgency cues.\n"
        "2. Select the most appropriate category and priority based on the guidelines and escalation rules.\n"
        "3. For category and priority, provide a brief internal explanation of your reasoning (mentioning specific guidelines/rules applied) and a confidence score (0.0 to 1.0).\n"
        "4. Draft a specific, professional customer response grounded in the historical examples. If no exact match is found, use general product knowledge from the examples to provide a helpful path forward.\n"
        "5. Return strict JSON ONLY with the keys: category, category_explanation, category_confidence, priority, priority_explanation, priority_confidence, response, confidence, flags.\n\n"
        "### Flags Guidance\n"
        "Include relevant flags in the 'flags' array, such as: 'enterprise-impact', 'onboarding-blocker', 'time-sensitive', 'data-loss-risk', 'sensitive-operation', 'silent-failure', 'workaround-available'.\n\n"
        f"Allowed Categories: [{allowed_categories}]\n"
        f"Allowed Priorities: [{allowed_priorities}]\n\n"
        f"Ticket ID: {ticket.get('ticket_id', '')}\n"
        f"Customer: {ticket.get('customer_name', '')} (Plan: {ticket.get('plan', '')})\n"
        f"Subject: {ticket.get('subject', '')}\n"
        f"Body: {ticket.get('body', '')}\n"
    )
    return {"role": "user", "content": message}


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
            obj = json.loads(text[left: right + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def _classify_with_llm(
        ticket: Dict[str, str],
        context_examples: List[Dict[str, object]],
        save_dir: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    ticket_id = str(ticket.get("ticket_id", "")).strip()
    config = _get_llm_config()
    model_name = config["model"]

    def _log(
            status: str,
            *,
            raw_content: str = "",
            usage: Optional[Dict[str, object]] = None,
            error: Optional[str] = None,
            result: Optional[Dict[str, object]] = None,
    ) -> None:
        if save_dir is None:
            return
        record: Dict[str, object] = {
            "ticket_id": ticket_id,
            "model": model_name,
            "status": status,
            "usage": usage if usage is not None else {},
        }
        if raw_content:
            record["raw_content"] = raw_content
        if error:
            record["error"] = error
        if result is not None:
            record["result"] = result
        append_llm_output_record(save_dir, record)

    if not config["api_key"]:
        _log("skipped", error="llm_skipped_missing_api_key")
        return None, "llm_skipped_missing_api_key"

    try:
        from openai import OpenAI
    except ImportError:
        _log("skipped", error="llm_skipped_missing_openai_dependency")
        return None, "llm_skipped_missing_openai_dependency"

    message = _make_llm_message(ticket, context_examples)
    try:
        client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        response = client.chat.completions.create(
            model=config["model"],
            messages=cast(Any, [message]),
            temperature=0.1,
        )
    except Exception as e:
        _log("error", error=f"llm_call_failed: {e}")
        return None, f"llm_call_failed: {e}"

    content = ""
    usage: Dict[str, object] = {}

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
        _log("error", raw_content=content, usage=usage, error="llm_invalid_json")
        return None, "llm_invalid_json"

    category = str(payload.get("category", "unknown")).strip().lower()
    priority = str(payload.get("priority", "medium")).strip().lower()
    response_text = str(payload.get("response", "")).strip()

    # Extract per-choice explanations and confidence
    category_explanation = str(payload.get("category_explanation", "")).strip()
    priority_explanation = str(payload.get("priority_explanation", "")).strip()
    category_confidence = payload.get("category_confidence", 0.75)
    priority_confidence = payload.get("priority_confidence", 0.75)

    confidence_raw = payload.get("confidence")
    if confidence_raw is None:
        try:
            confidence_raw = (float(str(category_confidence)) + float(str(priority_confidence))) / 2
        except (TypeError, ValueError):
            confidence_raw = 0.75

    try:
        confidence = float(str(confidence_raw))
    except (TypeError, ValueError):
        confidence = 0.75

    if category not in ALLOWED_CATEGORIES:
        _log("error", raw_content=content, usage=usage, error="llm_invalid_category")
        return None, "llm_invalid_category"
    if priority not in ALLOWED_PRIORITIES:
        _log("error", raw_content=content, usage=usage, error="llm_invalid_priority")
        return None, "llm_invalid_priority"
    if not response_text:
        _log("error", raw_content=content, usage=usage, error="llm_empty_response")
        return None, "llm_empty_response"

    raw_flags = payload.get("flags", [])
    flags = [str(flag) for flag in raw_flags] if isinstance(raw_flags, list) else []
    result = {
        "ticket_id": ticket.get("ticket_id", ""),
        "category": category,
        "category_explanation": category_explanation,
        "category_confidence": category_confidence,
        "priority": priority,
        "priority_explanation": priority_explanation,
        "priority_confidence": priority_confidence,
        "response": response_text,
        "confidence": max(0.0, min(confidence, 1.0)),
        "flags": sorted(set(flags + ["llm_used"])),
        "usage": usage,
    }
    _log("ok", raw_content=content, usage=usage, result=result)
    return result, None


def _score_overlap(ticket_tokens: set[str], example_tokens: set[str], subject: str, body: str) -> float:
    ticket_text = normalize_text(subject + " " + body)
    if not ticket_tokens or not example_tokens:
        overlap = 0.0
    else:
        overlap = len(ticket_tokens.intersection(example_tokens)) / max(len(ticket_tokens), 1)
    semantic_ratio = SequenceMatcher(None, ticket_text, " ".join(sorted(example_tokens))).ratio()
    return 0.65 * overlap + 0.35 * semantic_ratio


def _bm25_like_score(query_tokens: set[str], example_tokens: set[str], idf: Dict[str, float],
                     avg_doc_len: float) -> float:
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
    return scored


def retrieve_context(ticket: Dict[str, str], kb_index: Dict[str, object], top_k: int = 5) -> List[Dict[str, object]]:
    top_k = max(top_k, 1)
    ranked = _rank_context_examples(ticket, kb_index)
    selected = [example for _, example in ranked]

    return selected[:top_k]


def _classify_ticket(
        ticket: Dict[str, str],
        kb_index: Dict[str, object],
        *,
        llm_output_dir: Optional[Path] = None,
        llm_cache: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    context_examples = retrieve_context(ticket, kb_index)
    context_ids = [example.get("ticket_id", "") for example in context_examples]
    ticket_id = str(ticket.get("ticket_id", "")).strip()

    if llm_cache is not None:
        cached = llm_cache.get(ticket_id)
        if cached is not None and str(cached.get("status", "")) == "ok":
            result_any = cached.get("result")
            if isinstance(result_any, dict):
                result = copy.deepcopy(cast(Dict[str, object], result_any))
                result["context_ticket_ids"] = context_ids
                return result
        return classify_with_fallback(
            ticket=ticket,
            context_examples=context_examples,
            category_hints=CATEGORY_HINTS,
            priority_hints=PRIORITY_HINTS,
            allowed_categories=ALLOWED_CATEGORIES,
            allowed_priorities=ALLOWED_PRIORITIES,
            llm_error_flag="llm_cache_miss",
        )

    llm_result, llm_error_flag = _classify_with_llm(
        ticket, context_examples, save_dir=llm_output_dir,
    )
    if llm_result is not None:
        llm_result["context_ticket_ids"] = context_ids
        return llm_result

    return classify_with_fallback(
        ticket=ticket,
        context_examples=context_examples,
        category_hints=CATEGORY_HINTS,
        priority_hints=PRIORITY_HINTS,
        allowed_categories=ALLOWED_CATEGORIES,
        allowed_priorities=ALLOWED_PRIORITIES,
        llm_error_flag=llm_error_flag,
    )


def classify_ticket(
        ticket: Dict[str, str],
        kb_index: Dict[str, object],
        *,
        llm_output_dir: Optional[Path] = None,
        llm_cache: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    start_time = time.time()
    result = _classify_ticket(
        ticket,
        kb_index,
        llm_output_dir=llm_output_dir,
        llm_cache=llm_cache,
    )
    end_time = time.time()
    time_taken = end_time - start_time
    usage_any = result.get("usage", {})
    usage = cast(Dict[str, object], usage_any if isinstance(usage_any, dict) else {})
    usage["time_taken"] = time_taken
    result["usage"] = usage
    return result
