# W8D2 - RAG Evaluation and Optimisation

## Objective
Build a LangChain and ChromaDB RAG pipeline, evaluate it using Ragas, and optimise the weakest evaluation metric.

## Implementation
- Loaded five source documents covering RAG, ChromaDB, embeddings, evaluation and LlamaIndex.
- Split documents using RecursiveCharacterTextSplitter.
- Stored document chunks in persistent ChromaDB.
- Generated a 10-question evaluation dataset.
- Added 10 ground-truth reference answers.
- Used Ollama models for generation, embeddings and evaluation.
- Tracked baseline metrics using MLflow.

## Baseline Evaluation

| Metric | Score |
|---|---:|
| Faithfulness | 0.5258 |
| Answer Relevancy | 0.6447 |
| Context Precision | 0.7500 |
| Context Recall | 0.7600 |

The lowest baseline metric was Faithfulness.

## Retrieval Optimisation

Changed retrieval top_k from 3 to 2 while retaining chunk size 800 and overlap 120.

| Configuration | Faithfulness |
|---|---:|
| Baseline: top_k=3 | 0.5258 |
| Optimised: top_k=2 | 0.5750 |

Faithfulness improved by 0.0492 (4.92 percentage points).

The optimised result was evaluated on the same 10 questions using the Faithfulness metric. Other metrics were not re-evaluated after optimisation.

## Evidence
- outputs/baseline_results.json
- outputs/optimized_results.json
- outputs/mlflow.db

## Conclusion
The experiment showed a measurable improvement in faithfulness after reducing the number of retrieved chunks from three to two. The results are specific to the selected dataset and local Ollama models.
