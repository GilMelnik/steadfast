"""Stage 7: error analysis over evaluation results."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List


def analyze_errors(tickets: List[Dict[str, str]], predictions: List[Dict[str, object]]) -> Dict[str, object]:
	by_ticket = {pred["ticket_id"]: pred for pred in predictions}

	root_causes = Counter()
	category_confusions = Counter()
	priority_confusions = Counter()
	flagged_tickets = defaultdict(list)
	examples: List[Dict[str, object]] = []

	for ticket in tickets:
		ticket_id = ticket["ticket_id"]
		expected_category = ticket.get("expected_category")
		expected_priority = ticket.get("expected_priority")
		pred = by_ticket.get(ticket_id, {})

		predicted_category = pred.get("category", "unknown")
		predicted_priority = pred.get("priority", "medium")
		raw_flags = pred.get("flags", [])
		flags = [str(flag) for flag in raw_flags] if isinstance(raw_flags, list) else []

		category_wrong = expected_category != predicted_category
		priority_wrong = expected_priority != predicted_priority

		if category_wrong and priority_wrong:
			root_causes["category_and_priority_mismatch"] += 1
		elif category_wrong:
			root_causes["category_mismatch"] += 1
		elif priority_wrong:
			root_causes["priority_mismatch"] += 1

		if category_wrong:
			category_confusions[(expected_category, predicted_category)] += 1
		if priority_wrong:
			priority_confusions[(expected_priority, predicted_priority)] += 1

		if flags:
			flagged_tickets[ticket_id].extend(flags)

		if category_wrong or priority_wrong:
			examples.append(
				{
					"ticket_id": ticket_id,
					"subject": ticket.get("subject", ""),
					"expected_category": expected_category,
					"predicted_category": predicted_category,
					"expected_priority": expected_priority,
					"predicted_priority": predicted_priority,
					"flags": sorted(set(flags)),
				}
			)

	return {
		"total_tickets": len(tickets),
		"root_causes": dict(root_causes),
		"top_category_confusions": [
			{"expected": e, "predicted": p, "count": c}
			for (e, p), c in category_confusions.most_common(10)
		],
		"top_priority_confusions": [
			{"expected": e, "predicted": p, "count": c}
			for (e, p), c in priority_confusions.most_common(10)
		],
		"flag_frequency": dict(Counter(flag for flags in flagged_tickets.values() for flag in flags)),
		"error_examples": examples[:20],
	}
