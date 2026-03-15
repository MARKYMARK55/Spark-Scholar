#!/bin/bash

echo "🔍 Testing GPU access in Docling container..."
echo "=========================================="

# Test 1: nvidia-smi
echo -e "\n📊 1. nvidia-smi output:"
docker exec docling nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Test 2: PyTorch CUDA check
echo -e "\n🎮 2. PyTorch CUDA check:"
docker exec docling python3 -c "
import torch
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
"

# Test 3: Check environment variables
echo -e "\n⚙️  3. GPU environment variables:"
docker exec docling env | grep -E "CUDA|NVIDIA"

echo -e "\n✅ Test complete!"
