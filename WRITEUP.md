# Write-Up

## Data Exploration

I started by inspecting the knowledge base schema (`ticket_id`, `plan`, `subject`, `body`, `category`, `priority`, `resolution`) and sampling representative rows.

What stood out:
- The KB includes repeated issue patterns with concrete Steadfast product terms (e.g., HubSpot sync mapping, PDF export service, IP allowlisting, workflow guides).
- Priority is not purely category-driven: similar issues can appear with different severities depending on urgency wording and plan context.
- Several onboarding/account/integration intents are semantically close, making them a likely confusion cluster.

How this informed the design:
- I built preprocessing to normalize text and extract retrieval tokens from `subject + body + resolution`.
- I used retrieval of similar KB tickets before classification so customer responses can include specific, grounded next steps.
- I added post-processing heuristics for known ambiguous boundaries (billing/account, onboarding/integration, bug/performance) and urgency escalations.

## Pipeline Design Decisions

### Stage 1 - Load Data
- Implemented in `src/pipeline.py` (`load_eval_set`) and `src/preprocess.py` (`load_knowledge_base`).
- Uses `pathlib.Path` and typed dictionaries for consistent downstream processing.

### Stage 2 - Preprocess
- Implemented in `src/preprocess.py`:
  - lowercasing and punctuation cleanup,
  - whitespace normalization,
  - lightweight stopword filtering,
  - token set construction for retrieval.
- Output includes an index with examples and token maps.

### Stage 3 - Classification and Response Generation
- Implemented in `src/agent.py` as a retrieval-first classifier.
- For each eval ticket, it retrieves top KB neighbors by token overlap + similarity ratio.
- Category and priority are scored from:
  1) keyword hints,
  2) nearest-neighbor label priors.
- Response generation is grounded using top retrieved resolution summary plus category-specific actionable next step.

### Stage 4 - Validation
- Implemented in `src/validate.py`.
- Enforces required fields, enum values, non-empty response, confidence bounds.
- Invalid outputs are corrected to safe defaults and tagged in `flags`.

### Stage 5 - Post-Processing Heuristics
- Implemented in `src/postprocess.py`.
- Adds rule-based corrections for high-signal terms (invoice/refund -> billing, SSO/webhook/sync -> integration, suspicious/MFA -> security).
- Escalates/de-escalates priority based on urgency cues and optional enterprise-sensitive iteration mode.

### Stage 6 - Evaluation
- Implemented in `src/evaluate.py`.
- Metrics:
  - category exact accuracy,
  - category partial-credit accuracy,
  - priority accuracy,
  - response quality score.
- Includes per-category and per-priority breakdown plus top confusion pairs.

### Stage 7 - Error Analysis
- Implemented in `src/analyze.py`.
- Produces root-cause buckets, confusion tables, flag frequency, and sampled error examples.

### Stage 8 - Iterate
- Implemented in `src/pipeline.py` as a second-pass option with more aggressive priority escalation.
- Chooses best iteration by aggregate objective:
  `category_partial + priority_accuracy + response_quality`.

## Iteration Log

What I tried:
1. Baseline retrieval + scoring + grounded response.
2. Second pass with stronger priority escalation for enterprise/urgency-heavy tickets.

What worked:
- Retrieval grounding improved response quality and reduced generic replies.
- Validation + flags produced clean, schema-consistent output with zero validation failures.

What did not materially improve:
- Aggressive priority escalation increased over-prediction of high/critical for some medium/low tickets.

Latest run selected: `iteration_1_baseline`.

| Iteration | Change | Category Acc | Priority Acc | Response Quality |
|-----------|--------|-------------|-------------|-----------------|
| v1 | Baseline retrieval classifier + validation + heuristics | 0.6522 (exact), 0.7065 (partial) | 0.5435 | 0.8681 |
| v2 | Aggressive priority escalation | lower combined objective vs v1 | lower than v1 | similar |
| v3 | Not selected | - | - | - |

## Response Quality Metric

I used a bounded heuristic score in `src/evaluate.py`:

`0.45 * lexical_overlap + 0.25 * actionability + 0.30 * category_alignment`

Where:
- `lexical_overlap`: overlap between ticket tokens and response tokens,
- `actionability`: binary signal from markers like `please`, `settings >`, `share`, `enable`, `resync`,
- `category_alignment`: full credit if predicted category matches expected, partial otherwise.

Why this metric:
- It rewards relevance and concrete next steps rather than politeness alone.
- It is deterministic and fast, useful for iterative local tuning.

Limitations:
- It is not a semantic judge; it can miss nuanced correctness.
- Token overlap can reward superficial mirroring.
- A future upgrade should use an LLM-as-judge rubric aligned to hidden-test criteria.

## What I'd Do Differently

With more time, I would:
1. Replace heuristic Stage 3 scoring with a real LLM call (structured JSON output) and retrieval-augmented prompt.
2. Add embeddings-based retrieval for stronger semantic match than token overlap.
3. Improve priority modeling with calibrated probabilities and richer urgency features.
4. Expand error-analysis-driven rules for known confusion groups (`bug` vs `performance`, `onboarding` vs `account`).
5. Add unit tests for each stage and snapshot tests on eval artifacts to protect regressions.
