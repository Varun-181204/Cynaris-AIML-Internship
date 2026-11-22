# CIA Full-Stack Mentor Review Log — W8D2

## Review 1 — Architecture and Implementation
Status: Preliminary mentor review received.

Areas reviewed:
- Project layout and Git hygiene
- Document ingestion and metadata
- Text chunking and overlap
- ChromaDB persistence and embeddings
- Retrieval and prompt configuration
- 10-question evaluation dataset
- Ragas metrics

Mentor feedback:
- Preserve source metadata for evaluation.
- Verify chunk size and overlap.
- Verify persistent vector-store configuration.
- Check retrieval results and source documents.
- Ensure all 10 questions have valid ground truths.

Follow-up requested: Share implementation snippets for detailed line-by-line feedback.

## Review 2 — Evaluation and Optimisation
Status: Methodology review received.

Baseline:
- Faithfulness: 0.5258
- Answer Relevancy: 0.6447
- Context Precision: 0.7500
- Context Recall: 0.7600

Optimisation:
- Changed top_k from 3 to 2.
- Faithfulness improved to 0.5750.
- Improvement: 0.0492.

Mentor recommendations:
- Expand the evaluation dataset to 30–50 questions.
- Record retrieved chunk IDs and sources.
- Perform error analysis on low-faithfulness answers.
- Consider additional retrieval configurations.
- Document experiment limitations.

Important evaluation limitation:
Only faithfulness was re-evaluated after optimisation. The other three optimised metric values are not independently measured.

## Review outcome
Both mentor interactions received and documented.
Recommendations retained for future improvements.
