FROM python:3.11-slim

LABEL maintainer="MARKYMARK55"
LABEL description="Arxiv RAG Proxy — OpenAI-compatible FastAPI proxy with LangGraph RAG pipeline"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Ensure the pipeline package is in PYTHONPATH
ENV PYTHONPATH=/app

# Non-root execution
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["uvicorn", "api.rag_proxy:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1", "--log-level", "info"]
