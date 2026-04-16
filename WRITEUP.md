# Write-Up

## Data Exploration

I started by inspecting the knowledge base schema (`ticket_id`, `plan`, `subject`, `body`, `category`, `priority`,
`resolution`) and sampling representative rows.

What stood out:

- The KB includes repeated issue patterns with concrete Steadfast product terms (e.g., HubSpot sync mapping, PDF export
  service, IP allowlisting, workflow guides).
- Priority is not purely category-driven: similar issues can appear with different severities depending on urgency
  wording and plan context.

How this informed the design:

- I built preprocessing to normalize text and extract retrieval tokens from `subject + body + resolution`.
- I used retrieval of similar KB tickets before classification so customer responses can include specific, grounded next
  steps.
- I added post-processing heuristics for known ambiguous boundaries (billing/account, onboarding/integration,
  bug/performance) and urgency escalations.

## Pipeline Design Decisions

### Stage 1 - Load Data

- Implemented in `src/pipeline.py` (`load_eval_set`) and `src/preprocess.py` (`load_knowledge_base`).

### Stage 2 - Preprocess

- Implemented in `src/preprocess.py`:
    - lowercasing and punctuation cleanup,
    - whitespace normalization,
    - lightweight stopword filtering,
    - token set construction for retrieval.
- Output includes an index with examples and token maps.

### Stage 3 - Classification and Response Generation

- Implemented in `src/agent.py` as a retrieval-augmented classifier.
- For each eval ticket, it retrieves the top-K knowledge base neighbors using a weighted ensemble ranking:
    - **BM25-like Lexical Score (44%)**: Rewards matching rare, high-signal tokens using Inverse Document Frequency (
      IDF).
    - **Token Overlap (24%)**: Measures the raw intersection density between ticket and KB tokens.
    - **Sequence Similarity (22%)**: Uses structural matching (`SequenceMatcher`) to capture text patterns and flow.
    - **Intent Signal (10%)**: Injects domain-specific bonuses for matching known category/priority keyword hints.
- Classification and Response:
    - If configured, an LLM pass uses the retrieved context for grounded, structured output.
    - A fallback heuristic scoring system handles cases without LLM access, using nearest-neighbor priors.
    - Responses are grounded in the resolution summaries of the top retrieved matches.

### Stage 4 - Validation

- Implemented in `src/validate.py`.
- Enforces required fields, enum values, non-empty response, confidence bounds.
- Invalid outputs are corrected to safe defaults and tagged in `flags`.

### Stage 5 - Post-Processing Heuristics

- Implemented in `src/postprocess.py`.
- **Category Corrections**:
    - Re-routes `billing` tickets to `account` if they focus on legal/SLA terms rather than specific charges.
    - Moves `integration` hits to `bug` when explicit 500 errors are present.
    - Converts `integration` to `feature_request` for requests framed as "would love to see" functionality.
- **Priority Adjustments**:
    - **Escalations**: Forces `high` or `critical` for onboarding blockers, silent failures, data loss signals, or churn
      risks (cancellations) on Enterprise and Growth plans.
    - **De-escalations**: Dampens priority for simple "how-to" questions on Starter plans, non-breach security setups (
      SSO/allowlisting) with available workarounds, and billing discount inquiries.

### Stage 6 - Evaluation

- Implemented in `src/evaluate.py`.
- Metrics:
    - **Category Accuracy**: Both exact match and partial-credit (0.5) for related pairs (e.g., bug/performance,
      account/onboarding, integration/onboarding).
    - **Priority Accuracy**: Both exact match and a normalized distance score (ordinal penalty) where closer misses (
      e.g., low vs medium) receive more credit than far misses (e.g., low vs critical).
    - **Response Quality Score**: A weighted heuristic measuring relevance, actionability, and alignment.
- Includes per-category/priority breakdowns, top confusion pairs, and resource usage tracking (tokens and latency).

### Stage 7 - Error Analysis

- Implemented in `src/analyze.py`.
- Produces root-cause buckets, confusion tables, flag frequency, and sampled error examples.

## Iteration Log

- I started by getting a baseline with running without an LLM and with an LLM, just to understand what is the playground.
- I tried a rerank logic for retrieving better context, but it did not help much.
- I removed the too generic post processing logic, which improved the priority accuracy.
- I then focused on improving the prompt, focusing more on priority.
- I used the mistakes from the previous iteration to implement a post process logic that focuses on priority, which improved the priority accuracy.
- I suspected that post process is overfitted to the test set, so I sampled a new test set from the kb. The priority accuracy dropped significantly, which made me realize that the post
  process logic is overfitting to the original test set. 
- I removed the post process logic, and simplified the prompt further to avoid overfitting. 
- The priority accuracy was still low, which made me suspect that the distribution of the test set is still not matching 
the distribution of the kb, so I resampled a new small test set with distribution matching the kb. 
- I changed the prompt to request a scoring to the prediction. It improved the priority accuracy.
- I realize that the post process logic is still overfitting to the original test set, so I reimplemented the post process


| Iteration | change                                                     | num_tickets | category_accuracy_exact | priority_accuracy_exact | response_quality_score |
|-----------|------------------------------------------------------------|-------------|-------------------------|-------------------------|------------------------|
| 1         | First run without an LLM                                   | 46          | 0.6522                  | 0.587                   | 0.8693                 |
| 2         | Run with LLM                                               | 46          | 0.8696                  | 0.5                     | 0.9668                 |
| 3         | Tried a rerank logic with LLM                              | 46          | 0.8913                  | 0.5                     | 0.9697                 |
| 4         | Removed post process                                       | 46          | 0.8913                  | 0.5217                  | 0.972                  |
| 5         | Tried a new prompt                                         | 46          | 0.8913                  | 0.6087                  | 0.9609                 |
| 6         | Reimplement a post process logic that focuses on priority  | 46          | 0.8913                  | 0.6957                  | 0.9525                 |
| 7         | Run on a new test set sampled from kb                      | 62          | 0.9194                  | 0.2581                  | 0.9593                 |
| 8         | Removed post process                                       | 62          | 0.9194                  | 0.3065                  | 0.9625                 |
| 9         | Simplified prompt to aviod overfitting                     | 62          | 0.9194                  | 0.3226                  | 0.9647                 |
| 10        | Sampled a new small test set with distribution matching kb | 50          | 0.8                     | 0.28                    | 0.9641                 |
| 11        | Added the prompt a request for scoring their prediction    | 50          | 0.86                    | 0.34                    | 0.965                  |
| 12        | Reimplement post process based on sampled test set         | 50          | 0.84                    | 0.34                    | 0.9729                 |
| 14        | Run on eval_set.json                                       | 46          | 0.8478                  | 0.6087                  | 0.9764                 |

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

1. Add embeddings-based retrieval for stronger semantic match than token overlap.
2. Improve priority modeling with calibrated probabilities and richer urgency features.
