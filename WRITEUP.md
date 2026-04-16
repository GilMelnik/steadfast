# Write-Up

## Data Exploration

I started by inspecting the knowledge base schema (`ticket_id`, `plan`, `subject`, `body`, `category`, `priority`, `resolution`) and sampling representative rows.

What stood out:

* The KB contains recurring issue patterns tied to specific Steadfast platform features, such as HubSpot sync mapping, PDF export services, IP allowlisting, and workflow guides.
* Priority is not determined solely by category: similar issue types can appear with different severities depending on urgency wording, customer impact, and plan context.

How this informed the design:

* I built preprocessing to normalize text and extract retrieval tokens from `subject + body + resolution`.
* I used retrieval of similar historical tickets before classification, allowing customer responses to include specific, grounded next steps based on prior resolutions.
* I kept post-processing heuristics intentionally minimal, because multiple iterations showed that manually engineered rules quickly converged to test-set-specific overfitting.

---

## Pipeline Design Decisions

### Stage 1 - Load Data

* Implemented in `src/pipeline.py` (`load_eval_set`) and `src/preprocess.py` (`load_knowledge_base`).
* **This stage validates the expected schema early and standardizes the in-memory format used by downstream stages.**

### Stage 2 - Preprocess

Implemented in `src/preprocess.py`:

* lowercasing and punctuation cleanup
* whitespace normalization
* lightweight stopword filtering
* token set construction for retrieval

**Design rationale:**

* The goal was fast and deterministic lexical retrieval without introducing external embedding dependencies.
* The output includes an index with examples and token maps to support efficient top-K lookup during inference.

### Stage 3 - Classification and Response Generation

* Implemented in `src/agent.py` as a retrieval-augmented classifier and responder.

* For each eval ticket, the pipeline retrieves the top-K KB neighbors using a weighted ensemble ranking:

* **BM25-like Lexical Score (44%)**: rewards matching rare, high-signal tokens using inverse document frequency (IDF).

* **Token Overlap (24%)**: measures raw token intersection density.

* **Sequence Similarity (22%)**: uses `SequenceMatcher` to capture structural similarity and recurring phrasing patterns.

* **Intent Signal (10%)**: injects domain-specific bonuses for known category and priority hints.

Classification and response generation:

* If configured, an LLM pass uses the retrieved context for grounded structured output.
* A fallback heuristic scoring system handles cases without LLM access using nearest-neighbor priors.
* Responses are grounded in the resolution summaries of the highest-ranked retrieved matches, which improves specificity and customer usefulness.

### Stage 4 - Validation

* Implemented in `src/validate.py`.
* Enforces:

  * required fields
  * valid enum values
  * non-empty response
  * confidence bounds
* Invalid outputs are corrected to safe defaults and tagged in `flags`.

### Stage 5 - Post-Processing Heuristics

* Implemented in `src/postprocess.py`.
* Final version intentionally kept minimal.

I experimented with multiple rule-based corrections, especially around priority, but repeated testing on newly sampled KB-derived sets showed these rules were overfitting to the dev distribution rather than learning robust business logic.


### Stage 6 - Evaluation

* Implemented in `src/evaluate.py`.

Metrics:

* **Category Accuracy**: exact match + partial credit (0.5) for related pairs (e.g. bug/performance, account/onboarding, integration/onboarding).
* **Priority Accuracy**: exact match + normalized ordinal distance score.
* **Response Quality Score**: weighted heuristic measuring relevance, actionability, and alignment.

The evaluation also includes:

* per-category breakdowns
* per-priority breakdowns
* top confusion pairs
* latency and token usage tracking


### Stage 7 - Error Analysis

* Implemented in `src/analyze.py`.
* Produces:

  * root-cause buckets
  * confusion tables
  * validation flag frequency
  * sampled error examples

This stage was used as the main driver for prompt refinement and heuristic rollback decisions.

---

## Iteration Log

My iteration process focused on separating real improvements from dev-set overfitting.

### What worked

* Adding an LLM significantly improved category accuracy and response quality.
* Prompt iterations that explicitly asked the model to score its own confidence before selecting priority improved priority performance.
* Claude Opus produced the strongest category accuracy on the official eval set.

### What did not work

* Retrieval reranking produced negligible gains relative to complexity.
* Rule-based priority post-processing initially improved dev-set metrics, but failed badly on newly sampled KB-derived test sets.
* Aggressive prompt specialization also showed signs of overfitting and did not generalize.

### Iteration narrative

* I first established baselines with and without an LLM (Claude Sonnet 4.6) to understand the improvement ceiling.
* I then tested retrieval reranking, which gave only marginal gains.
* Priority became the main weakness, so I focused iterations on prompt wording and confidence estimation.
* A temporary priority-specific post-processing layer improved the official dev score.
* To test robustness, I sampled fresh validation sets from the KB.
* The sharp priority drop on these new sets revealed that the heuristic layer was overfitting.
* I then removed most post-processing logic and simplified the prompt to improve generalization.
* Finally, after gaining access to additional models, I tested **Claude Opus 4.6**, which gave the strongest final eval-set results.


| Iteration | change                                                                   | num_tickets | category_accuracy_exact | priority_accuracy_exact | response_quality_score |
|-----------|--------------------------------------------------------------------------|-------------|-------------------------|-------------------------|------------------------|
| 1         | Baseline heuristic retrieval pipeline without LLM classification         | 46          | 0.6522                  | 0.587                   | 0.8693                 |
| 2         | Added LLM-based classification                                           | 46          | 0.8696                  | 0.5                     | 0.9668                 |
| 3         | Tested retrieval reranking to improve top-K KB context quality           | 46          | 0.8913                  | 0.5                     | 0.9697                 |
| 4         | Removed generic post-processing after weak impact on robustness          | 46          | 0.8913                  | 0.5217                  | 0.972                  |
| 5         | Refined prompt to improve priority reasoning and issue grounding         | 46          | 0.8913                  | 0.6087                  | 0.9609                 |
| 6         | Added priority-focused heuristic correction layer                        | 46          | 0.8913                  | 0.6957                  | 0.9525                 |
| 7         | Validated generalization on a newly sampled KB-derived test set          | 62          | 0.9194                  | 0.2581                  | 0.9593                 |
| 8         | Removed priority heuristics after detecting dev-set overfitting          | 62          | 0.9194                  | 0.3065                  | 0.9625                 |
| 9         | Simplified prompt to improve generalization and reduce overfitting       | 62          | 0.9194                  | 0.3226                  | 0.9647                 |
| 10        | Created smaller distribution-matched validation set for faster iteration | 50          | 0.8                     | 0.28                    | 0.9641                 |
| 11        | Added the prompt a self-scored confidence and explanation step           | 50          | 0.86                    | 0.34                    | 0.965                  |
| 12        | Switched model to Claude Opus 4.6 for stronger reasoning quality         | 50          | 0.9                     | 0.3                     | 0.9671                 |
| 13        | Final Claude Opus 4.6 evaluation on official dev set                     | 46          | 0.913                   | 0.6522                  | 0.9521                 |


---

## Response Quality Metric

I used a bounded heuristic score in `src/evaluate.py`:

`0.45 * lexical_overlap + 0.25 * actionability + 0.30 * category_alignment`

Where:

* `lexical_overlap`: overlap between ticket tokens and response tokens
* `actionability`: binary signal from markers such as `please`, `settings >`, `share`, `enable`, `resync`
* `category_alignment`: full credit if predicted category matches expected, partial otherwise

Why this metric:

* It rewards issue-specific relevance and actionable next steps, not just polite phrasing.
* It is deterministic, fast, and suitable for rapid local iteration loops.

Limitations:

* It is not a semantic judge and can miss nuanced correctness.
* Token overlap may reward superficial mirroring.

---

## What I’d Do Differently

With more time, I would:

1. Add embedding-based retrieval for stronger semantic matching than lexical overlap.
2. Replace the heuristic quality metric with an LLM judge aligned to the hidden rubric.
3. Add cross-validation over multiple KB-derived sampled dev sets to detect overfitting earlier.
4. Expand model comparisons across more providers and reasoning-focused models.

