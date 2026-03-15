# Dockerfile.blackwell
FROM --platform=linux/arm64 nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
    CUDAARCHS=120 \
    PYTORCH_JIT=0

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev curl git \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
        libgomp1 tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA for ARM64
RUN pip3 install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install "docling>=1.14.0"

# Create a small patch to fix JIT compilation
RUN echo 'import torch\n\
if not torch.cuda.get_arch_list():\n\
    print("Warning: No CUDA architectures detected, forcing sm_120")\n\
    torch.cuda.set_arch_list(["sm_120"])\n\
' > /app/patch_cuda.py

COPY server.py .

# Apply the patch before running
RUN echo '#!/bin/bash\n\
python3 /app/patch_cuda.py\n\
exec python3 -m uvicorn server:app --host 0.0.0.0 --port 5001\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 5001

ENTRYPOINT ["/entrypoint.sh"]