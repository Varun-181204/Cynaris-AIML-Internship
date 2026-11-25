# W8D1 - Production RAG Monitoring Strategy

## 1. Objective

Monitor the production RAG API to maintain reliability, response quality,
retrieval accuracy, performance and resource efficiency.

## 2. Metrics to Track

| Category | Metric | Purpose |
|---|---|---|
| Availability | API uptime | Measure service availability |
| Availability | HTTP 5xx error rate | Detect server-side failures |
| Latency | End-to-end query latency | Measure complete response time |
| Latency | Retrieval latency | Measure document retrieval time |
| Retrieval | Retrieved chunks per query | Detect retrieval anomalies |
| Retrieval | Precision@K and Recall@K | Evaluate retrieval relevance |
| Generation | Faithfulness | Measure whether answers are supported by context |
| Generation | Answer relevancy | Measure relevance to the question |
| Generation | Unknown-answer rate | Track questions the system cannot answer |
| Infrastructure | CPU and memory usage | Detect resource pressure |
| Infrastructure | Container restarts | Detect application instability |
| Dependencies | Ollama availability | Monitor model service connectivity |
| Storage | ChromaDB collection size | Monitor indexed data growth |
| Experiment tracking | MLflow run metrics | Compare model and configuration performance |

## 3. Alert Conditions

| Alert | Trigger | Recommended Action |
|---|---|---|
| API unavailable | Health endpoint fails repeatedly | Restart or investigate API container |
| High server errors | HTTP 5xx rate exceeds 5% over 5 minutes | Inspect application logs |
| High latency | P95 query latency exceeds 10 seconds | Investigate retrieval and generation bottlenecks |
| Ollama unavailable | Health check cannot reach Ollama | Check Ollama service and connectivity |
| Empty retrieval index | Retrieval reports an empty collection | Run document ingestion |
| Low faithfulness | Evaluation score falls below 0.80 | Review prompts, context and retrieval quality |
| High memory usage | Container memory exceeds 85% | Investigate resource consumption |
| Repeated restarts | Multiple unexpected restarts | Inspect container logs and resource limits |

These thresholds are initial operational recommendations and should be tuned using production baselines.

## 4. Retraining and Re-indexing Triggers

A RAG system does not necessarily require model retraining whenever its knowledge changes.

The following actions should be considered:

- Document updates: Re-ingest changed documents and refresh ChromaDB embeddings.
- Retrieval quality degradation: Review chunk size, overlap, embedding model and top-K settings.
- Faithfulness degradation: Review retrieved context and generation prompts.
- Persistent answer-quality degradation: Evaluate alternative language models or fine-tuning where justified.
- Embedding model changes: Re-embed the document collection using the new embedding model.

Any model or configuration change should be evaluated against a fixed evaluation dataset before deployment.

## 5. Monitoring and Evaluation Tools

- FastAPI: API health and HTTP response monitoring.
- Docker: Container status, resource usage and logs.
- Ollama: Model service availability.
- ChromaDB: Document index and retrieval data.
- MLflow: Experiment parameters, latency and retrieved chunk metrics.
- Ragas: Retrieval and generation quality evaluation.

## 6. Current Implementation and Limitations

Implemented:
- FastAPI health endpoint with Ollama connectivity check.
- Query latency recorded through MLflow.
- Retrieved chunk count recorded through MLflow.
- Docker container logs available for inspection.
- ChromaDB document indexing and retrieval.

Planned production enhancements:
- Centralized metrics collection and dashboards.
- Automated alert notifications.
- Scheduled Ragas evaluation.
- Production traffic monitoring and historical trend analysis.
- Automated evaluation gates before model or prompt deployment.

## 7. Incident Response

1. Detect the issue through health checks, metrics or alerts.
2. Inspect Docker and application logs.
3. Check Ollama connectivity and model availability.
4. Verify ChromaDB index health.
5. Run a known evaluation query.
6. Apply the required fix and rerun automated tests.
7. Record the incident and corrective action.

## 8. Conclusion

The monitoring strategy combines API health, infrastructure monitoring,
retrieval evaluation and generation-quality measurements. MLflow provides
experiment tracking, while Ragas can be used for systematic RAG evaluation.
