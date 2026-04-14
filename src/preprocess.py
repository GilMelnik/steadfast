"""Stage 2: preprocess and normalize the knowledge base."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Set

from utils import normalize_text, normalize_whitespace, tokenize


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
