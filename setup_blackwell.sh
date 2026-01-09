#!/bin/bash
# Optimized setup script for NVIDIA Blackwell GPUs (GB10, compute 12.1)
# Builds xformers v0.0.33 from source with Blackwell support
set -e

echo "=== Blackwell GPU Optimized Setup ==="
echo "This script builds xformers from source for optimal performance"
echo ""

# Environment variables for Blackwell
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST="12.1"
export MAX_JOBS=4  # Adjust based on your CPU cores

VENV_DIR="/home/gabriel/dev/gpt-2048/.venv"

# Clean up existing venv
rm -rf "$VENV_DIR"
echo "Creating virtual environment..."
uv venv --python 3.12 "$VENV_DIR"

echo "=== Step 1: Installing PyTorch with CUDA 13.0 ==="
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 --python "$VENV_DIR/bin/python"

echo "=== Step 2: Installing base dependencies ==="
uv pip install ninja cmake wheel setuptools numpy --python "$VENV_DIR/bin/python"

echo "=== Step 3: Building xformers v0.0.33 from source (Blackwell support) ==="
cd /tmp
rm -rf xformers
git clone --depth=1 --branch v0.0.33 --recursive https://github.com/facebookresearch/xformers.git
cd xformers
TORCH_CUDA_ARCH_LIST="12.1" MAX_JOBS=4 uv pip install --no-build-isolation -e . --python "$VENV_DIR/bin/python"
cd -

echo "=== Step 4: Installing ML stack ==="
uv pip install transformers==4.57.3 peft datasets trl==0.22.2 --python "$VENV_DIR/bin/python"

echo "=== Step 5: Installing unsloth (no deps to avoid conflicts) ==="
uv pip install --no-deps unsloth unsloth-zoo --python "$VENV_DIR/bin/python"

echo "=== Step 6: Installing bitsandbytes ==="
uv pip install --no-deps bitsandbytes --python "$VENV_DIR/bin/python"

echo "=== Step 7: Installing torchao ==="
uv pip install torchao --python "$VENV_DIR/bin/python"

echo "=== Step 8: Installing wandb for experiment tracking ==="
uv pip install wandb --python "$VENV_DIR/bin/python"

echo ""
echo "=== Verification ==="
"$VENV_DIR/bin/python" -c "
import os
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

import triton
print(f'Triton: {triton.__version__}')

import xformers
print(f'xformers: {xformers.__version__}')

# Test xformers
try:
    from xformers.ops import memory_efficient_attention
    x = torch.randn(1, 8, 64, 64, device='cuda', dtype=torch.float16)
    out = memory_efficient_attention(x, x, x)
    print('xformers memory_efficient_attention: WORKING')
except Exception as e:
    print(f'xformers memory_efficient_attention: FAILED ({e})')

print('Setup complete!')
"

echo ""
echo "=== Done! ==="
echo "To activate: source $VENV_DIR/bin/activate"
echo "To run training: python gpt_2048_rl.py"
