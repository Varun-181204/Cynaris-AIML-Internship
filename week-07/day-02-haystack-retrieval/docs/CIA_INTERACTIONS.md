# CIA Mentor Interactions - W7D2

## Interaction 1 - Pipeline implementation checklist

**Review focus:** Haystack pipeline architecture and implementation completeness.

**Feedback received:**
- Checklist covering PDF conversion, document stores, splitting,
  embeddings, BM25, dense retrieval, extractive reading, pipeline
  orchestration, evaluation, and CI.
- Highlighted common issues involving metadata propagation, persistence,
  chunk overlap, path handling, error handling, and reproducibility.

**Assessment and action:**
The response included APIs and classes associated with older Haystack
versions, such as FARMReader, FAISSDocumentStore, and
ElasticsearchDocumentStore. The project uses Haystack 3.5.0 and its
current implementation, so these suggestions were treated as a general
checklist rather than copied directly. Version compatibility must be
verified before adopting them.

## Interaction 2 - Evaluation fairness audit

**Review focus:** Source-level Precision@3, latency, and answer quality.

**Feedback received:**
- Passage-level relevance would provide finer-grained evaluation.
- Deduplication by PDF can hide multiple useful passages from one source.
- Fixed-denominator Precision@3 should be clearly explained.
- Add recall and broader ranking metrics.
- Separate retrieval precision from answer correctness.
- Expand the question set and repeat latency measurements.
- Document ground-truth creation and evaluation reproducibility.

**Assessment and action:**
The current metric is retained as a source-level proxy because the
experiment explicitly evaluates relevant PDF sources. It is not
presented as passage-level precision or answer accuracy. The README
and self-review document the metric definition, small dataset, and
answer-quality limitations. Additional metrics and human answer grading
are recorded as future improvements.

## Summary
Two CIA interactions were completed and their feedback was reviewed.
Suggestions were assessed against the actual Haystack 3.5.0
implementation rather than adopting incompatible APIs.
