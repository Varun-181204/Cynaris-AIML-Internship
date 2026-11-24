# W8D3 - Haystack BM25 and Dense Retrieval API Deployment

## Objective
Build a Haystack retrieval pipeline, compare BM25 and dense retrieval using 10 questions, and expose both methods through a FastAPI service prepared for Railway deployment.

## Implementation
- Indexed five PDF reference documents (`data/pdfs/`).
- Converted and split documents into 22 chunks (`split_length=180`, `split_overlap=30`).
- Implemented BM25 retrieval using `InMemoryBM25Retriever`.
- Implemented dense retrieval using Ollama embeddings with `nomic-embed-text`.
- Added an extractive reader (`ExtractiveReader`) to return answers and source documents.
- Evaluated both retrieval methods using the same 10 questions (`outputs/retrieval_evidence.json`).
- Exposed FastAPI `GET /`, `GET /health`, and `POST /query` endpoints.
- Hardened startup, `Dockerfile`, `.dockerignore`, `requirements.txt`, and `railway.toml` for Railway deployment.

## Recommended Railway Architecture
Deploy the FastAPI service (`week-08/day-03-haystack-railway`) as a standalone Docker service on Railway:
- **BM25-first resilient startup**: At startup, the API converts and indexes the 5 PDFs (22 chunks) in memory for BM25 immediately with zero external dependencies, so `/health` passes reliably even if Ollama is unavailable.
- **Dense retrieval via `OLLAMA_URL`**:
  - **Recommended for full BM25 + Dense on Railway**: Run a second `ollama` service in the same Railway project (with `nomic-embed-text` pulled) and set `OLLAMA_URL=http://<ollama-service>.railway.internal:11434` on the API service.
  - **Graceful fallback**: If `OLLAMA_URL` is unreachable at boot, BM25 remains active (`bm25_ready=true`, `dense_ready=false`). Dense queries lazily attempt embedding indexing when Ollama becomes reachable or return HTTP `503` with an actionable diagnostic message.

See [docs/railway_deployment_guide.md](docs/railway_deployment_guide.md) for full setup and validation steps.

## Evaluation Results

| Metric | BM25 | Dense |
|---|---:|---:|
| Mean Precision@3 | 0.3333 | 0.3333 |
| Mean latency (ms) | 5.63 | 55.05 |

Result: Tie in mean Precision@3.

Precision@3 uses a fixed denominator of three. Retrieved documents are deduplicated by PDF filename. Relevance judgments are source-level proxy judgments, not independent passage-level annotations.

## API Endpoints
- `GET /` - returns service metadata, readiness flags (`ready`, `bm25_ready`, `dense_ready`), indexed chunk count, and endpoint paths.
- `GET /health` - reports API readiness, retrieval mode availability, indexed chunk count, embedding model, and sanitized Ollama URL.
- `POST /query` - accepts `question` (3-1000 non-whitespace characters) and `method` (`bm25` or `dense`).

## Validation
- Five PDFs indexed successfully (22 chunks).
- Ten evaluation questions executed for each retrieval method (`outputs/retrieval_evidence.json`).
- FastAPI `GET /`, `GET /health`, and `POST /query` (`bm25`, `dense` lazy recovery, `503` fallback, `422` validation) verified via `TestClient`.
- Pytest: 19 passed (12 pipeline/evidence tests + 7 API/configuration tests).
- Ruff lint: passed.
- Ruff format check: passed.
- Live Railway deployment and live Railway endpoint testing: pending deployment.

## Deployment Notes
- Set the Railway service **Root Directory** to `week-08/day-03-haystack-railway`.
- Configure `OLLAMA_URL` (e.g., `http://ollama.railway.internal:11434` or a public HTTPS Ollama endpoint) and optionally `OLLAMA_EMBEDDING_MODEL` (default `nomic-embed-text`) and `OLLAMA_TIMEOUT` (default `30`).
- Localhost Ollama (`http://localhost:11434`) is only suitable for local testing.

## Limitations
- Evaluation uses five reference PDFs and ten questions.
- Source-level relevance judgments are proxies.
- Dense retrieval latency includes embedding generation.
- `InMemoryDocumentStore` re-indexes the 5 bundled PDFs on container startup (22 chunks, fast startup, non-persistent across restarts).
- Live Railway deployment has not yet been executed or verified against a live Railway URL.

## Conclusion
Both retrieval methods achieved equal mean Precision@3 on the evaluation set, with lower local latency for BM25. The FastAPI service and Railway configuration now support resilient BM25-first startup, dynamic `OLLAMA_URL` configuration, and clear diagnostics when Ollama is unavailable.
