# W7D2 Self Review

## Completed work
- Loaded five PDF reference documents.
- Indexed 22 document chunks.
- Implemented BM25 and dense retrieval using Haystack 3.5.0.
- Evaluated both methods on the same ten questions.
- Saved query-level retrieval and answer evidence.
- Added tests for dataset, retrieval, and evaluation evidence.

## Results
- BM25 mean Precision@3: 0.3333
- Dense mean Precision@3: 0.3333
- BM25 mean observed latency: 5.79 ms
- Dense mean observed latency: 61.56 ms
- Automated tests: 12 passed.

## What went well
The pipeline successfully indexed the dataset, executed both retrieval
methods, generated evidence, and passed all 12 tests.

## Limitations identified
1. Source-level relevance does not measure passage-level relevance.
2. Deduplication can hide multiple useful passages from one PDF.
3. A fixed top-3 denominator limits the maximum score when fewer than
   three relevant sources exist.
4. Ten questions provide limited evaluation coverage.
5. Retrieval precision and extractive answer correctness are different.
6. Latency should be measured repeatedly and split into retrieval,
   reader, and end-to-end timings for a stronger benchmark.

## Improvements for future work
- Create passage-level relevance annotations.
- Expand and diversify the question set.
- Add Recall@3 and other ranking metrics.
- Manually grade answer correctness.
- Repeat latency measurements and report variability.
- Compare retrieval results with a baseline.

## Reflection
The experiment showed equal source-level Precision@3 for BM25 and dense
retrieval on this dataset. BM25 had lower observed latency in this run.
The results are limited to the current dataset, questions, and evaluation
method and should not be generalized beyond this experiment.
