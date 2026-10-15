FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_LLM_PROVIDER=fake \
    RAG_AUTO_INGEST=true

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app
COPY sample_docs ./sample_docs

RUN pip install --no-cache-dir .

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
