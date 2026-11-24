# W8D3 Railway Deployment & Validation Guide

## 1. Recommended Railway Architecture

The simplest reliable Railway architecture for this Haystack API separates stateless API serving from optional Ollama embedding inference:

```
+-------------------------------------------------------------------+
| Railway Project Environment                                       |
|                                                                   |
|  +-----------------------------+     Private Network (optional)   |
|  | haystack-api (Dockerfile)   |    OLLAMA_URL=http://ollama...   |
|  | - FastAPI + Uvicorn         | ------------------------------>  |
|  | - 5 PDFs (22 chunks in RAM) |                                  |
|  | - BM25 always ready         |     +-------------------------+  |
|  | - Dense lazy/startup index  |     | ollama (ollama/ollama)  |  |
|  +-----------------------------+     | model: nomic-embed-text |  |
|                 |                    +-------------------------+  |
|                 v                                                 |
|      Public HTTPS Endpoint                                        |
|      GET /, GET /health, POST /query                              |
+-------------------------------------------------------------------+
```

### Why this architecture is recommended
1. **No single-container bloat**: Bundling the Ollama server and `nomic-embed-text` weights inside the FastAPI image slows Docker builds, increases cold-start time, and risks exceeding small container memory limits.
2. **Reliable `/health` startup**: Because `api.py` runs `index_pdfs(require_dense=False)` during `lifespan`, all 5 PDFs (22 chunks) are indexed into `InMemoryDocumentStore` for BM25 immediately without waiting on Ollama. Railway's `/health` check returns HTTP `200` within seconds.
3. **Flexible Dense Retrieval**:
   - **Option A (Two Railway services - full BM25 + Dense)**: Deploy `ollama/ollama` as a companion service in the same Railway project, pull `nomic-embed-text`, and set `OLLAMA_URL=http://<ollama-service>.railway.internal:11434`.
   - **Option B (Single Railway service - BM25 + external Ollama or BM25-only)**: Deploy only `haystack-api`. Point `OLLAMA_URL` to a reachable Ollama instance if available; otherwise BM25 queries return `200 OK` and dense queries return a clear `503 Service Unavailable`.

---

## 2. Deployment Blockers Identified and Fixed

1. **Startup crash when Ollama is unavailable**: `lifespan` previously called `index_pdfs()`, which ran `OllamaDocumentEmbedder` unconditionally and crashed Uvicorn when `localhost:11434` was unreachable. Fixed by splitting BM25 indexing from dense embedding generation (`require_dense=False`) and adding lazy dense recovery (`ensure_dense_ready()`).
2. **Import-time `OLLAMA_URL` evaluation and trailing slashes**: `OLLAMA_URL` is now resolved dynamically via `resolve_ollama_url()` (strips whitespace and trailing `/`), `OLLAMA_EMBEDDING_MODEL`, and `OLLAMA_TIMEOUT`, with credential sanitization before displaying in `/health` or error messages.
3. **Railway `startCommand` shell expansion**: Updated `railway.toml` `startCommand` to `sh -c 'uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}'` so Railway's `$PORT` (or local fallback `8000`) always expands properly.
4. **Docker hardening & reproducibility**: Added `.dockerignore`, pinned `pypdf==6.21.2` in `requirements.txt`, added `PYTHONDONTWRITEBYTECODE=1`, `EXPOSE 8000`, and a non-root `appuser` in `Dockerfile`.
5. **Root route & health diagnostics**: Added `GET /` and expanded `GET /health` with `bm25_ready`, `dense_ready`, `indexed_chunks`, `embedding_model`, sanitized `ollama_url`, and `dense_error`.
6. **UTF-8 encoding in `README.md`**: Replaced a Windows-1252 byte in `README.md` so `ruff format --check` runs cleanly across the folder.

---

## 3. Railway Service Configuration

### Build & Service Settings
- **Repository**: `Cynaris-AIML-Internship`
- **Branch**: `feat/aiml-W8-Varun-K-S`
- **Root Directory**: `week-08/day-03-haystack-railway`
- **Config File**: `week-08/day-03-haystack-railway/railway.toml` (auto-detected under Root Directory)
- **Builder**: `DOCKERFILE` (`dockerfilePath = "Dockerfile"`)
- **Healthcheck Path**: `/health`
- **Healthcheck Timeout**: `300` seconds
- **Restart Policy**: `ON_FAILURE` (`restartPolicyMaxRetries = 3`)

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `PORT` | Injected by Railway | `8000` | Port Uvicorn binds to |
| `OLLAMA_URL` | Required for Dense | `http://localhost:11434` | Base URL of reachable Ollama service (e.g. `http://ollama.railway.internal:11434`) |
| `OLLAMA_EMBEDDING_MODEL` | Optional | `nomic-embed-text` | Ollama embedding model name |
| `OLLAMA_TIMEOUT` | Optional | `30` | Timeout in seconds for Ollama requests |

---

## 4. Local Pre-Deploy Verification Commands

### Python tests and lint checks (executed locally)
```powershell
pytest week-08/day-03-haystack-railway/tests -v
ruff check week-08/day-03-haystack-railway
ruff format --check week-08/day-03-haystack-railway/api.py week-08/day-03-haystack-railway/src week-08/day-03-haystack-railway/tests
```

### Local Docker container check (run with Docker Desktop started)
```powershell
cd week-08/day-03-haystack-railway
docker build -t cynaris-w8d3-haystack-api:local .
docker run --rm -p 8000:8000 -e OLLAMA_URL=http://host.docker.internal:11434 cynaris-w8d3-haystack-api:local
```

---

## 5. Live Railway Validation Checklist (Run After Deployment)

Replace `<RAILWAY_URL>` with your deployed service URL (for example, `https://cynaris-w8d3.up.railway.app`).

1. **Check root endpoint**:
   ```bash
   curl -sS https://<RAILWAY_URL>/
   ```
   Expected: HTTP `200`, `ready: true`, `bm25_ready: true`, `indexed_chunks: 22`.

2. **Check health endpoint**:
   ```bash
   curl -sS https://<RAILWAY_URL>/health
   ```
   Expected: HTTP `200`, `status: "ok"`, `bm25_ready: true`, `indexed_chunks: 22`. If Ollama is connected, `dense_ready: true`.

3. **Test BM25 query**:
   ```bash
   curl -sS -X POST https://<RAILWAY_URL>/query \
     -H "Content-Type: application/json" \
     -d '{"question": "How are components connected in a Haystack pipeline?", "method": "bm25"}'
   ```
   Expected: HTTP `200` with `answer`, `source`, `retrieved_sources`, and `latency_ms`.

4. **Test Dense query**:
   ```bash
   curl -sS -X POST https://<RAILWAY_URL>/query \
     -H "Content-Type: application/json" \
     -d '{"question": "How does dense retrieval use embeddings to find documents?", "method": "dense"}'
   ```
   Expected: HTTP `200` when `OLLAMA_URL` is configured with `nomic-embed-text`, or HTTP `503` with a clear `Dense retrieval is unavailable` diagnostic if Ollama is not attached.
