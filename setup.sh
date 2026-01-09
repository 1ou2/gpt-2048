#!/bin/bash
set -e
rm -rf ./.venv
echo "Creating virtual environment..."
uv venv --clear
source ./.venv/bin/activate
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST=12.1a
echo "Installing PyTorch with CUDA 130..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

echo "Installing transformers stack..."
uv pip install transformers peft datasets trl

echo "Installing unsloth (no deps)..."
uv pip install --no-deps unsloth unsloth-zoo

echo "Installing bitsandbytes (no deps)..."
uv pip install --no-deps bitsandbytes

echo "Installing torchao..."
uv pip install --upgrade torchao

echo "Installing wandb for experiment tracking..."
uv pip install wandb

echo "Upgrading unsloth stack..."
uv pip install --upgrade unsloth unsloth-zoo transformers

#uv pip install flash-attn --no-build-isolation
uv pip install xformers

#echo "Installing remaining dependencies..."
#uv pip install aiohttp requests tqdm jupyter

echo "Done! All packages installed successfully."
