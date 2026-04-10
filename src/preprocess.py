"""Stage 2: preprocess and normalize the knowledge base."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Set

STOPWORDS = {
	"a",
	"an",
	"and",
	"are",
	"as",
	"at",
	"be",
	"by",
	"for",
	"from",
	"has",
	"have",
	"in",
	"is",
	"it",
	"of",
	"on",
	"or",
	"that",
	"the",
	"this",
	"to",
	"we",
	"with",
}


def normalize_whitespace(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
	cleaned = text.lower().replace("\u2019", "'").replace("\u2014", "-")
	cleaned = re.sub(r"[^a-z0-9\s\-_/]", " ", cleaned)
	return normalize_whitespace(cleaned)


def tokenize(text: str) -> Set[str]:
	tokens = {tok for tok in normalize_text(text).split(" ") if tok and tok not in STOPWORDS}
	return {tok for tok in tokens if len(tok) > 2 or tok.isdigit()}


def load_knowledge_base(kb_path: Path) -> List[Dict[str, str]]:
	with kb_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		return [dict(row) for row in reader]


def preprocess_knowledge_base(kb_rows: List[Dict[str, str]]) -> Dict[str, object]:
	examples: List[Dict[str, object]] = []
	category_keywords: Dict[str, Set[str]] = {}
	priority_keywords: Dict[str, Set[str]] = {}
	document_frequency: Dict[str, int] = {}
	total_token_count = 0

	for row in kb_rows:
		subject = row.get("subject", "")
		body = row.get("body", "")
		resolution = row.get("resolution", "")
		category = row.get("category", "").strip().lower()
		priority = row.get("priority", "").strip().lower()

		combined = " ".join((subject, body, resolution))
		normalized_combined = normalize_text(combined)
		tokens = tokenize(combined)
		resolution_summary = normalize_whitespace(resolution)[:320]
		total_token_count += len(tokens)
		for token in tokens:
			document_frequency[token] = document_frequency.get(token, 0) + 1

		example = {
			"ticket_id": row.get("ticket_id", ""),
			"plan": row.get("plan", ""),
			"category": category,
			"priority": priority,
			"subject": subject,
			"body": body,
			"resolution": resolution,
			"resolution_summary": resolution_summary,
			"search_text": normalized_combined,
			"tokens": tokens,
		}
		examples.append(example)

		category_keywords.setdefault(category, set()).update(tokens)
		priority_keywords.setdefault(priority, set()).update(tokens)

	num_examples = max(len(examples), 1)
	avg_doc_len = total_token_count / num_examples
	idf = {
		token: math.log(1.0 + (num_examples - freq + 0.5) / (freq + 0.5))
		for token, freq in document_frequency.items()
	}

	return {
		"examples": examples,
		"category_keywords": category_keywords,
		"priority_keywords": priority_keywords,
		"idf": idf,
		"avg_doc_len": avg_doc_len,
	}
