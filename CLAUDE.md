# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project trains GPT-OSS-20B to play the 2048 game using reinforcement learning (GRPO - Group Relative Policy Optimization). The model learns to generate Python strategy functions that take a game board state and return optimal moves.

## Environment Setup

### Local Development (using uv)

```bash
# Run the setup script to create venv and install dependencies
./setup.sh
source .venv/bin/activate
```

The setup script installs:
- PyTorch with CUDA 13.0
- Unsloth (for efficient model fine-tuning)
- Transformers, PEFT, TRL, datasets
- xformers (for attention optimization)
- bitsandbytes (for quantization)

### Docker (for DGX Spark or H100 systems)

```bash
docker build -f Dockerfile -t unsloth-dgx-spark .
docker run -it --gpus=all --net=host --ipc=host \
    -v $(pwd):$(pwd) -w $(pwd) unsloth-dgx-spark
```

### GPU Compatibility Issues

This project has specific GPU architecture requirements:

**NVIDIA GB10 (Blackwell) or newer GPUs (CUDA capability 12.1+)**:
- PyTorch 2.9.1 only officially supports up to compute capability 12.0
- Flash Attention 2 will fail to install due to missing libcudart.so.12
- xformers requires Flash Attention to be uninstalled
- Triton may fail with PTXAS errors for `sm_121a` architecture

**Required workarounds**:
1. **Uninstall flash-attn**: `pip uninstall flash-attn -y`
2. **Set Triton environment variables** (for PTXAS errors):
   ```bash
   export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
   export TORCH_CUDA_ARCH_LIST=12.1a
   ```

**Expected output** after fixes:
```
FA [Xformers = 0.0.33.post2. FA2 = False]
```

## Running Training

### Main Training Script

```bash
# Set environment variables for Blackwell GPU support
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST=12.1a

# Run training
python gpt_2048_rl.py
```

**Training time**: ~25-27 hours for 1,000 steps on GB10 GPU with optimized settings.

### Performance Optimization

The training configuration in `gpt_2048_rl.py:429-447` has been optimized:
- `max_completion_length=200` (down from 548) - strategy functions are short
- `gradient_accumulation_steps=4` - best balance for GRPO with this hardware
- `num_generations=2` - **REQUIRED** by GRPO (cannot be reduced to 1)

Increasing `gradient_accumulation_steps` beyond 4 may **slow down** training due to overhead. The sweet spot is 4 for this GPU/model combination.

### Jupyter Notebook

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

The notebook version is available at: `gpt_oss_20B_RL_2048_Game.ipynb`

## Architecture

### Code Structure

The main file `gpt_2048_rl.py` is organized into sections:

1. **2048 Game Implementation** (lines 22-233)
   - Pure Python implementation of 2048 game logic
   - `GameBoard` class manages game state, moves, and rendering
   - Moves: "é" (up), "u" (down), "a" (left), "i" (right)
   - Board size: 6x6, target tile: 2048

2. **RL Environment Setup** (lines 235-254)
   - `execute_strategy()`: Runs a strategy function with 5-second timeout
   - Executes generated code in sandboxed environment

3. **Code Execution and Safety** (lines 256-269)
   - `extract_function()`: Parses strategy functions from model output
   - Expects functions in markdown code blocks starting with `def strategy(board):`

4. **Reward Functions** (lines 271-352)
   - `function_works()`: Checks if function syntax is valid (+1.0 or -2.0)
   - `no_cheating()`: Penalizes forbidden imports (-20.0 if cheating detected)
   - `strategy_succeeds()`: Rewards gameplay performance (+20.0 for success, +2.0 for valid attempt)

5. **Model Loading** (lines 354-382)
   - Uses `unsloth/gpt-oss-20b` with 4-bit quantization
   - LoRA rank: 4 (low rank for faster training)
   - Max sequence length: 768 tokens
   - Offloads embeddings to RAM to save 1GB VRAM

6. **Training Configuration** (lines 414-463)
   - GRPO (Group Relative Policy Optimization)
   - Learning rate: 5e-5 with linear schedule
   - AdamW 8-bit optimizer
   - Saves checkpoints every 100 steps to `outputs/`

### Key Design Decisions

**Why GRPO requires `num_generations=2`**: GRPO calculates advantages by comparing multiple generations per prompt. With only 1 generation, it cannot compute relative rewards, which breaks the algorithm.

**Why `gradient_accumulation_steps=4` is optimal**: While higher values reduce optimizer steps, the overhead of accumulating gradients and the staleness of the policy in RL make 4 the sweet spot for this 20B model on GB10.

**Why completions are limited to 200 tokens**: Strategy functions are typically 50-100 tokens. The original 548-token limit was wasting GPU cycles generating padding.

**Reward function balance**: The three reward functions work together:
- `function_works`: Ensures basic code validity
- `no_cheating`: Enforces constraints (no external libraries)
- `strategy_succeeds`: Drives actual gameplay improvement

### Training Monitoring

Watch for these metrics in logs:
- `rewards/strategy_succeeds/mean`: Should increase over time (starts near 0, target 20.0)
- `completions/clipped_ratio`: If 1.0, all completions hit the length limit (may need adjustment)
- `kl`: KL divergence should stay low (< 0.01) to prevent mode collapse
- `completion_length`: Monitor if functions are getting truncated

## Model Output

Trained models can be saved in three formats:
- `mxfp4`: 4-bit mixed precision (smallest, recommended)
- `merged_16bit`: Full 16-bit weights (large)
- `lora`: LoRA adapters only (requires base model)

Uncomment lines 535-536 in `main()` to enable saving after training.

## Known Issues

1. **Triton PTXAS errors on Blackwell GPUs**: Set `TRITON_PTXAS_PATH` and `TORCH_CUDA_ARCH_LIST` environment variables
2. **Flash Attention incompatibility**: Must be uninstalled for xformers to work
3. **Move key mappings**: The game uses French keyboard layout ("é" for up, "i" for right)
4. **First training step is slower**: Includes compilation time (~2-3 minutes vs ~90 seconds for subsequent steps)
