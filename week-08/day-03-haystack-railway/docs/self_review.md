# W8D3 Self-Review Checklist

## Implementation
- [x] Haystack BM25 retrieval implemented (`InMemoryBM25Retriever`)
- [x] Haystack dense retrieval implemented (`OllamaDocumentEmbedder`, `OllamaTextEmbedder`, `InMemoryEmbeddingRetriever`)
- [x] Five reference PDFs indexed into 22 document chunks
- [x] Ten evaluation questions executed across BM25 and dense retrieval (`outputs/retrieval_evidence.json`)
- [x] FastAPI `GET /`, `GET /health`, and `POST /query` endpoints implemented
- [x] Startup hardened so BM25 is ready without blocking on Ollama and dense retrieval degrades cleanly (`503`) or recovers lazily
- [x] Dynamic `OLLAMA_URL`, `OLLAMA_EMBEDDING_MODEL`, and `OLLAMA_TIMEOUT` resolution with credential sanitization

## Deployment & Quality
- [x] `Dockerfile` hardened with non-root user, `EXPOSE 8000`, and `PYTHONDONTWRITEBYTECODE=1`
- [x] `.dockerignore` added for clean image build context
- [x] `railway.toml` configured with shell-safe `startCommand` and `/health` healthcheck
- [x] `requirements.txt` pinned (`pypdf==6.21.2`) and `requirements-dev.txt` added
- [x] `pytest` suite expanded to 19 tests (12 pipeline/evidence + 7 API/config tests), all passing
- [x] Ruff lint and formatting checks passing
- [x] Deployment and validation documentation added (`docs/railway_deployment_guide.md`)
- [ ] Local Docker build & container smoke test executed (Docker Desktop daemon not running during check)
- [ ] Live Railway deployment executed and live endpoints verified
- [ ] Changes pushed to `feat/aiml-W8-Varun-K-S` and PR #9 updated (pending user approval)
