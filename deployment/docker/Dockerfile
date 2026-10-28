# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY atlas ./atlas
COPY benchmarks/__init__.py ./benchmarks/__init__.py
COPY benchmarks/simulator ./benchmarks/simulator
RUN pip install --upgrade pip && pip install .

COPY data/docs ./data/docs
COPY data/web ./data/web
COPY data/logs ./data/logs

# Unprivileged user; only the state directory is writable.
RUN useradd --system --uid 10001 atlas && mkdir -p /app/data/state && chown -R atlas /app/data
USER atlas

ENV ATLAS_DATA_DIR=/app/data
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "atlas.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
