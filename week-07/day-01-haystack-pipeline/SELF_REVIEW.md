# W7D1 Self-Review — Haystack Pipeline Architecture

## Assignment Requirements

- [x] Installed Haystack AI.
- [x] Installed Ollama Haystack integration.
- [x] Created five PDF documents.
- [x] Converted PDFs into Haystack Documents.
- [x] Split documents into smaller chunks.
- [x] Indexed 22 document chunks.
- [x] Implemented BM25 retrieval.
- [x] Implemented dense retrieval using nomic-embed-text.
- [x] Implemented an extractive Reader component.
- [x] Executed ten questions using BM25.
- [x] Executed the same ten questions using dense retrieval.
- [x] Calculated Precision@3.
- [x] Compared both retrieval approaches.
- [x] Saved JSON execution evidence.

## Testing

- [x] Verified five PDF documents.
- [x] Verified ten evaluation questions.
- [x] Tested Reader answer extraction.
- [x] Tested empty retrieval handling.
- [x] Tested top-three chunk preservation.
- [x] Verified both retrieval methods in evidence.
- [x] Verified precision score boundaries.
- [x] Verified comparison consistency.
- [x] All 11 automated tests passed.

## Evaluation Results

| Metric | BM25 | Dense |
|---|---:|---:|
| Precision@3 | 0.5000 | 0.5667 |
| Questions | 10 | 10 |

Dense retrieval achieved an absolute improvement of 0.0667.

## Documentation and Evidence

- [x] README created.
- [x] Self-review checklist created.
- [x] Retrieval evidence JSON generated.
- [x] Five source PDFs included.
- [ ] Evidence screenshots captured.
- [ ] CIA interactions finalized.
- [ ] Final Git review completed.
- [ ] Two descriptive commits completed.
- [ ] Branch pushed.
- [ ] Weekly PR raised.

## Known Limitations

1. Evaluation uses expected source PDF labels as initial relevance judgments.
2. Independent human chunk-level relevance judgments are not yet available.
3. The extractive Reader selects a sentence using query-term overlap.
4. InMemoryDocumentStore does not persist data between runs.
5. Two original PDFs generate non-blocking parser warnings.

## Final Review

The indexing and retrieval workflows execute successfully.
Both retrieval approaches use the same dataset and ten questions.
All 11 automated tests passed.

**Status:** Implementation and automated testing complete.
Documentation complete. Submission and CIA verification pending.

## Final Evaluation Update

- [x] Incorporated CIA feedback on repeated source chunks.
- [x] Deduplicated retrieved results by PDF filename before source-level Precision@3.
- [x] Added a dense retrieval warm-up before latency measurement.
- [x] Regenerated retrieval evidence.
- [x] Python compilation passed.
- [x] All 12 automated tests passed.

| Metric | BM25 | Dense |
|---|---:|---:|
| Source-deduplicated Precision@3 | 0.3333 | 0.3333 |
| Mean query latency | 3.88 ms | 51.29 ms |

**Final interpretation:** Tie on this small dataset. Source-level relevance is only a proxy; independent passage-level judgments and a larger corpus are future improvements.
