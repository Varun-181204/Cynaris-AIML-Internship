FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_URL=http://host.docker.internal:11434
ENV PYTHONPATH=/app/week-08/day-01-production-rag/src
ENV GIT_PYTHON_REFRESH=quiet

WORKDIR /app

COPY week-07/day-05-multi-document-rag/src/ /app/week-07/day-05-multi-document-rag/src/
COPY week-08/day-01-production-rag/ /app/week-08/day-01-production-rag/

WORKDIR /app/week-08/day-01-production-rag

RUN pip install --no-cache-dir -r requirements.txt \
    && mkdir -p outputs/chroma_db outputs/mlruns \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app/week-08/day-01-production-rag

USER appuser

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
