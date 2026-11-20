# CIA Full Stack Mentor Mode Interactions — W7D3

## Interaction 1 — Implementation review
Date: 2026-11-19

Prompt/context: Requested a senior AI/ML review of the LlamaIndex, Ollama, ChromaDB indexing and query comparison implementation.

Actual mentor feedback: The mentor approved the overall architecture and recommended explicit UTF-8 loading, embedding warm-up, empty-document filtering, clearer Ollama errors, progress logging, and median/p95 latency. The mentor also raised concerns about embedding reuse, collection cleanup, persistence path, and latency definition.

Changes made: Added embedding warm-up, empty-document filtering, UTF-8 loading, Ollama health/model checks, logging, and additional latency statistics. The existing implementation already reused the shared Settings embedding model, deleted/recreated the Chroma collection, used an explicit PersistentClient path, and measured end-to-end query latency. An unverified seed parameter suggestion was not applied.

## Interaction 2 — Evaluation and quality review
Date: 2026-11-20

Prompt/context: Requested a review of evaluation fairness, latency interpretation, source retrieval verification, testing quality, and limitations.

Actual mentor feedback: The mentor considered the comparison largely controlled because both engines use the same five documents, ten questions, embedding model, and machine. It highlighted the small corpus and question set, recommended reporting retrieval depth and reproducibility controls, and suggested negative/edge-case tests, larger datasets, and repeated latency measurements with statistical analysis. It cautioned against generalizing the observed ChromaDB speed advantage.

Changes made: Documented the evaluation configuration, measured latency results, and study limitations. The existing code uses similarity_top_k=2. Expected-source tracking confirms that the expected source appears among the retrieved top two; it does not establish top-one accuracy or complete relevance. Larger datasets, paraphrased questions, negative queries, and repeated statistical benchmarks are future improvements.
