FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

LABEL maintainer="MARKYMARK55"
LABEL description="BGE-M3 Sparse Embedder — SPLADE-style sparse vectors via FlagEmbedding"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY sparse_embed.py .

# Pre-download the model to /model cache at build time
# (set HF_TOKEN as a build arg if the model is gated)
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
ENV SPARSE_MODEL="BAAI/bge-m3"
ENV HF_HOME="/app/.hf_cache"

# Download model during build to avoid startup delay at runtime
RUN python -c "\
from FlagEmbedding import BGEM3FlagModel; \
import os; \
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True); \
print('Model downloaded successfully'); \
" || echo "Model download will happen at runtime"

ENV SPARSE_EMBED_HOST=0.0.0.0
ENV SPARSE_EMBED_PORT=8035
ENV SPARSE_BATCH_SIZE=32
ENV SPARSE_MAX_LENGTH=512

EXPOSE 8035

HEALTHCHECK --interval=30s --timeout=20s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8035/health || exit 1

CMD ["uvicorn", "sparse_embed:app", "--host", "0.0.0.0", "--port", "8035", "--workers", "1", "--log-level", "info"]
