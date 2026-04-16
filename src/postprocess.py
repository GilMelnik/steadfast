"""Stage 5: rule-based corrections on top of classifier output."""

from __future__ import annotations

from typing import Dict


def apply_heuristics(
	ticket: Dict[str, str],
	prediction: Dict[str, object],
) -> Dict[str, object]:
	subject = ticket.get("subject", "").lower()
	body = ticket.get("body", "").lower()
	combined = f"{subject} {body}"
	
	current_category = str(prediction.get("category", "")).lower()
	current_priority = str(prediction.get("priority", "")).lower()
	plan = ticket.get("plan", "").lower()
	flags = prediction.get("flags", [])

	# 1. Category Corrections
	# SLA/Contractual -> Account (often misclassified as billing)
	if current_category == "billing":
		if any(word in combined for word in ["sla", "contract", "legal", "terms of service", "agreement"]):
			if "invoice" not in subject and "charge" not in subject:
				prediction["category"] = "account"
				current_category = "account"

	# 500 Error -> Bug (often misclassified as integration if it's an API)
	if "500" in combined or "internal server error" in combined:
		if current_category == "integration":
			prediction["category"] = "bug"
			current_category = "bug"

	# 2. Priority Corrections
	# Onboarding Blocker for Enterprise/Growth should be at least High
	if "onboarding-blocker" in flags:
		if plan in ["enterprise", "growth"]:
			if current_priority in ["low", "medium"]:
				prediction["priority"] = "high"
				current_priority = "high"

	# Silent Failure on Enterprise should be at least High
	if "silent-failure" in flags and plan == "enterprise":
		if current_priority in ["low", "medium"]:
			prediction["priority"] = "high"
			current_priority = "high"

	# Actual Data Loss should be Critical
	if "data-loss-risk" in flags:
		if any(word in combined for word in ["gone", "disappeared", "vanished", "deleted", "lost"]):
			if current_priority != "critical":
				prediction["priority"] = "critical"
				current_priority = "critical"

	# De-escalate simple "how to" on Starter plan
	if plan == "starter":
		if any(phrase in combined for phrase in ["how to", "is there a way", "where can i", "question about"]):
			if current_priority == "high":
				prediction["priority"] = "medium"
				current_priority = "medium"
			elif current_priority == "critical":
				prediction["priority"] = "high"
				current_priority = "high"

	# Churn risk (Cancellation/Downgrade) for high-value plans should be Critical
	if any(word in combined for word in ["cancel", "downgrade", "stop using", "closing my account"]):
		if plan in ["enterprise", "growth"]:
			if current_priority != "critical":
				prediction["priority"] = "critical"
				current_priority = "critical"

	# feature_request vs integration confusion
	if current_category == "integration":
		if any(phrase in combined for phrase in ["would love to see", "would be great", "is it possible to add", "is there a way to"]):
			# If it's asking for new functionality in an integration
			prediction["category"] = "feature_request"
			current_category = "feature_request"

	# Security configuration (not breach) shouldn't be High/Critical if it's just enablement and has a workaround
	if current_category == "security":
		if any(word in combined for word in ["enable", "setup", "configure", "allowlist", "sso"]):
			if "breach" not in combined and "hacked" not in combined and "unauthorized" not in combined:
				if "workaround-available" in flags:
					if current_priority in ["high", "critical"]:
						prediction["priority"] = "medium"
						current_priority = "medium"

	# Billing: Discount/Promo code issues are usually Medium
	if current_category == "billing" and any(word in combined for word in ["discount", "promo", "coupon"]):
		if current_priority == "high":
			prediction["priority"] = "medium"
			current_priority = "medium"

	return prediction
