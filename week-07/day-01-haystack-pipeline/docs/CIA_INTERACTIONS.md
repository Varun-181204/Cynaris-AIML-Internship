# W7D1 — CIA Full Stack Mentor Mode Interaction Log

## Interaction 1 — Implementation and Architecture Review

**Review status:** CIA response received. The source files were not attached during this interaction, so the mentor provided a template-style review and a checklist rather than a direct inspection of the implementation.

**Review scope:**
- Haystack pipeline architecture.
- PDF conversion and preprocessing.
- Document splitting and indexing.
- Document store and retriever configuration.
- Custom Reader component.
- Error handling and Python best practices.
- Performance and scalability considerations.

**CIA feedback and findings:**

1. Recommended a modular architecture separating document loading, splitting, embedding, storage, retrieval and reading responsibilities.
2. Identified the expected pipeline components: PDF converter, text splitter, embedding generator, document store, BM25 retriever, dense retriever and Reader.
3. Recommended verifying that both retrievers use the same document store and that dense retrieval uses compatible document and query embedding models.
4. Recommended preserving source filenames and page metadata for document traceability.
5. Recommended handling missing files, empty documents, embedding failures and empty retrieval results.
6. Recommended improving code quality through type hints, docstrings, logging, pinned dependencies and configuration management.
7. Suggested persistence and indexing reuse as future scalability improvements.

**Actions / follow-up:**

- Used the checklist to review the W7D1 implementation.
- Verified the indexing pipeline and retrieval pipeline structure.
- Identified the need to retain source metadata and handle limitations transparently.
- The response was advisory; no direct code inspection or CIA approval was claimed.

## Interaction 2 — Retrieval Evaluation Review

**Review status:** CIA evaluation review received. The mentor examined the described five-PDF, 22-chunk, ten-question BM25 versus dense retrieval evaluation.

**Review scope:**
- Precision@3 calculation.
- Relevance-judgment methodology.
- Repeated chunks from the same PDF.
- Fairness of BM25 and dense retrieval comparison.
- Query latency.
- Dataset limitations and evaluation metrics.

**CIA feedback and findings:**

1. The mean Precision@3 formula is appropriate when each query's relevant retrieved results are divided by three and then averaged.
2. If fewer than three results are returned, missing positions should count as non-relevant when using a fixed Precision@3 denominator.
3. Using expected PDF filenames as relevance labels is too coarse because not every chunk from a PDF necessarily answers the question.
4. Counting multiple chunks from the same PDF as separate relevant hits can over-reward a retriever under file-level relevance judgments.
5. The mentor recommended either passage-level relevance annotations or deduplicating retrieved results by source before calculating file-level precision.
6. Both retrieval methods use the same top-K setting and document store, which is appropriate for the comparison.
7. Average query latency should be reported alongside retrieval quality. Initial model-loading time should be considered when interpreting latency.
8. The small dataset and ten questions do not establish statistical significance or prove that dense retrieval is generally better.
9. The five-PDF dataset, chunk overlap, English-only content and absence of no-answer questions limit the evaluation.
10. Recall@K, MAP, nDCG@K, MRR and additional latency measurements were suggested as optional future metrics.

**Observed results before addressing the evaluation feedback:**

| Metric | BM25 | Dense |
|---|---:|---:|
| Mean Precision@3 | 0.5000 | 0.5667 |
| Mean query latency | 4.80 ms | 293.61 ms |
| Queries | 10 | 10 |

The original 0.0667 difference is preliminary and must not be interpreted as statistically significant.

**Actions / follow-up:**

- Reviewed the existing evaluation implementation against the CIA feedback.
- Identified the mismatch between source-level relevance labels and chunk-level duplicate counting.
- Evaluation methodology requires correction before treating the scores as final.
- Latency results require interpretation, particularly the first dense query and model-loading overhead.

## Review Integrity

Both interactions are based on responses received from CIA Full Stack Mentor Mode.

Interaction 1 was a general review checklist because implementation files were not attached. Interaction 2 provided specific feedback on the retrieval evaluation methodology.

Neither response is represented as an external approval or certification of the implementation. Follow-up corrections and test results are recorded below.
## Follow-up — Evaluation Corrections

After receiving the second CIA review, the evaluation was revised:

1. Retrieved results are now deduplicated by PDF filename before source-level Precision@3 is calculated.
2. The dense embedding pipeline is warmed up before the measured evaluation queries.
3. The evidence JSON was regenerated.
4. The previous preliminary scores (BM25 0.5000, Dense 0.5667) were superseded.
5. The corrected scores are BM25 0.3333 and Dense 0.3333, with mean query latencies of 3.88 ms and 51.29 ms respectively.
6. The test suite was updated for source deduplication and all 12 tests passed.

The remaining limitation is that relevance labels are PDF-level proxies rather than independent passage-level annotations.

