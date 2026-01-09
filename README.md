# Fine-tune GPT-OSS for the 2048 Game

This project uses reinforcement learning (GRPO - Group Relative Policy Optimization) to train GPT-OSS-20B to play the 2048 game by generating Python strategy functions.

## How It Works

### Algorithm Overview

**GRPO (Group Relative Policy Optimization)** is a reinforcement learning algorithm that:
1. Generates multiple completions per prompt (2 in this setup)
2. Evaluates each completion using reward functions
3. Calculates advantages by comparing rewards within each group
4. Updates the model to favor higher-reward strategies

The model learns to generate Python functions that take a game board state and return optimal moves (W/A/S/D).

### Training Dataset

The dataset contains a single prompt repeated 1,000 times:

```python
prompt = """
Create a new short 2048 strategy using only native Python code.
You are given a list of list of numbers for the current board state.
Output one action for "W", "A", "S", "D" on what is the optimal next step.
Output your new short function in backticks using the format below:
```python
def strategy(board):
    return "W" # Example
```
All helper functions should be inside def strategy. Only output the short function `strategy`.
"""
```

**Key constraints:**
- Functions must use only native Python (no imports allowed)
- Must return a single character: "W" (up), "A" (left), "S" (down), or "D" (right)
- Helper functions must be nested inside `strategy()`
- Board is a 6x6 grid passed as list of lists

### Reward Functions

Three reward functions guide the learning process (evaluated in sequence):

#### 1. `function_works()`
- **+1.0**: Valid Python syntax that can be executed
- **-2.0**: Syntax errors or cannot extract function
- **-0.5**: Extraction succeeded but execution failed

#### 2. `no_cheating()`
- **+1.0**: Uses only allowed Python (no imports)
- **-20.0**: Uses forbidden imports or modules (heavily penalized)
- **-1.0**: Failed to create function

#### 3. `strategy_succeeds()`
- **+20.0**: Strategy reaches 2048 tile (success!)
- **+2.0**: Strategy runs but doesn't reach 2048
- **-1.0**: Strategy times out (>5 seconds)
- **-3.0**: Strategy crashes with exception
- **0.0**: Invalid function

**Total possible reward range:** -25.0 (worst) to +22.0 (best)

### Training Process

1. **Warmup Phase (Steps 1-100)**
   - Learning rate increases from 0 to 5e-5
   - Model explores random strategy variations
   - Most strategies fail or timeout
   - Rewards are typically negative (-3.0 to -2.0)

2. **Learning Phase (Steps 100-500)**
   - Model starts generating valid strategies
   - Variance in rewards increases (essential for GRPO)
   - Occasional strategies play multiple moves
   - Rewards gradually improve toward 0.0 to +2.0

3. **Optimization Phase (Steps 500-1000)**
   - Model refines successful strategies
   - Focus on maximizing tile merges and score
   - Rewards target +4.0 (valid working strategy)
   - Rare successes may hit +20.0 (reaching 2048)

### What Happens During Training

Each training step (takes ~85-90 seconds):
1. Model generates 2 strategy functions (200 tokens each)
2. Each function is executed on a random 6x6 game board
3. Game plays out with 5-second timeout per strategy
4. Three reward functions evaluate results
5. GRPO calculates advantages and updates model weights
6. Checkpoint saved every 100 steps to `outputs/`

**Expected training time:** 25-27 hours for 1,000 steps on GB10 GPU

### Monitoring Training Progress

Watch these key metrics in the logs:

**Critical metrics:**
- `reward`: Total reward (target: increasing from -3.0 toward +4.0)
- `rewards/strategy_succeeds/mean`: Game performance (target: 2.0+)
- `reward_std`: Variance in rewards (must be >0 for GRPO to learn)

**Quality indicators:**
- `completions/clipped_ratio`: If 1.0, outputs hit 200-token limit (expected)
- `kl`: KL divergence (<0.01 is safe, >0.05 risks mode collapse)
- `frac_reward_zero_std`: Fraction of batches with no variance (target: <0.5)

**What you'll see:**
- First 50-100 steps: Timeouts, "return W" copies, negative rewards
- Steps 100-200: Some variance appears, occasional valid strategies
- Steps 200+: Rewards improve, strategies attempt actual gameplay

**According to the tutorial:** "You might have to wait 150 to 200 steps for any action. You'll probably get 0 reward for the first 100 steps. Please be patient!"

### Game Implementation

The 2048 game (gpt_2048_rl.py:22-233):
- **Board size:** 6x6 (larger than standard 4x4 for harder challenge)
- **Target tile:** 2048
- **Spawn rate:** 10% chance of spawning 4, 90% chance of spawning 2
- **Moves:** W (up), A (left), S (down), D (right)
- **Win condition:** Any tile reaches 2048
- **Lose condition:** No valid moves remaining

Each game starts with 2 random tiles and plays until win/loss or timeout.

## Setup & Installation

### Prerequisites
- NVIDIA GPU with CUDA support (tested on GB10/Blackwell)
- CUDA 12.1+ and cuDNN
- Python 3.10+
- 120GB+ GPU memory (for 20B model)

### Local Setup (using uv)

```bash
# Run the setup script
./setup.sh

# Activate virtual environment
source .venv/bin/activate
```

The setup script installs:
- PyTorch 2.9.1 with CUDA 13.0
- Unsloth (efficient fine-tuning)
- Transformers, PEFT, TRL, datasets
- xformers (attention optimization)
- bitsandbytes (quantization)
- wandb (experiment tracking)

**GPU Compatibility Note (Blackwell GB10 or newer):**
```bash
# Uninstall flash-attn if installation fails
pip uninstall flash-attn -y

# Set environment variables for Triton
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST=12.1a
```

## Reference
https://unsloth.ai/docs/basics/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth

### Docker Setup (Alternative)

```bash
# Build the Docker image
docker build -f Dockerfile -t unsloth-dgx-spark .

# Run with GPU support
docker run -it \
    --gpus=all \
    --net=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):$(pwd) \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -w $(pwd) \
    unsloth-dgx-spark
```

### Jupyter Notebook (Optional)

```bash
# Download the official notebook
NOTEBOOK_URL="https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/nb/gpt_oss_(20B)_Reinforcement_Learning_2048_Game_DGX_Spark.ipynb"
wget -O "gpt_oss_20B_RL_2048_Game.ipynb" "$NOTEBOOK_URL"

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## Running Training

### Standard Run (Recommended)

Use `screen` to run training in a persistent session that survives disconnections:

```bash
# Set environment variables (for Blackwell GPUs)
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST=12.1a

# Create a screen session
screen -S training2048

# Activate virtual environment
source .venv/bin/activate

# Run training with logging
python gpt_2048_rl.py 2>&1 | tee output.log

# Detach from screen: Press Ctrl+A then D
# Reattach later: screen -r training2048
```

### Monitoring Progress

```bash
# View live training output
screen -r training2048

# Or tail the log file from another terminal
tail -f output.log

# View recent training metrics
grep "reward.*mean" output.log | tail -20
```

### Weights & Biases Integration (Recommended)

Track your training experiments with [Weights & Biases](https://wandb.ai/) for real-time dashboards, metric visualization, and experiment comparison.

**Setup:**

1. Create a free account at [wandb.ai](https://wandb.ai/)
2. Get your API key from [wandb.ai/settings](https://wandb.ai/settings)
3. Create a `.env` file in the project root:

```bash
# .env
WANDB=your_wandb_api_key_here
```

4. Run training as normal - metrics will automatically sync to your dashboard

**Dashboard features:**
- Real-time loss and reward curves
- GPU utilization monitoring
- Comparison across training runs
- Hyperparameter tracking
- Model checkpoints

**View your dashboard:**
- Project URL: https://wandb.ai/YOUR_USERNAME/gpt-2048-rl
- Individual runs appear as clickable entries

**Key metrics logged:**
- `reward`: Total reward across all reward functions
- `rewards/function_works/mean`: Code validity score
- `rewards/no_cheating/mean`: Safety compliance score
- `rewards/strategy_succeeds/mean`: Game performance score
- `completions/clipped_ratio`: Percentage of truncated outputs
- `kl`: KL divergence from reference policy

**Disable logging:**
To disable wandb, either remove the `.env` file or set:
```bash
export WANDB_MODE=disabled
```

### Screen Commands Quick Reference

```bash
# Create session
screen -S training2048

# Detach from session
Ctrl+A then D

# Reattach to session
screen -r training2048

# List all sessions
screen -ls

# Kill session (from outside)
screen -X -S training2048 quit

# Scroll in screen (copy mode)
Ctrl+A then [
# Use arrow keys, PgUp/PgDn to scroll
# Press Esc to exit
```

## Saving the Trained Model

After training completes, uncomment lines 538-539 in `gpt_2048_rl.py`:

```python
# Save locally in mxfp4 format (4-bit, smallest size)
save_model(model, tokenizer, method="mxfp4")

# Or save in 16-bit format (larger but higher precision)
# save_model(model, tokenizer, method="merged_16bit")

# Or push to Hugging Face Hub
# push_to_hub(model, tokenizer, "username/gpt-oss-2048-rl", "your_hf_token", method="mxfp4")
```

**Save methods:**
- `mxfp4`: 4-bit mixed precision (recommended, ~5GB)
- `merged_16bit`: Full 16-bit weights (~40GB)
- `lora`: LoRA adapters only (requires base model, ~100MB)

## Testing the Trained Model

After training, test your model by running inference:

```python
from gpt_2048_rl import load_model, test_model, create_dataset

# Load the trained model
model, tokenizer, _ = load_model()

# Get the prompt
_, prompt = create_dataset()

# Generate a strategy
test_model(model, tokenizer, prompt)
```

The model will generate a strategy function that you can use to play 2048.

## Expected Results

**After 1,000 steps of training:**
- Model should generate syntactically valid Python functions
- Strategies should make multiple moves (10-100+ steps per game)
- Average reward should reach +2.0 to +4.0
- Occasional strategies may reach high tiles (512, 1024)
- Reaching 2048 on 6x6 board is rare but possible

**Typical progression:**
- Steps 0-100: Learning basic syntax, mostly failures
- Steps 100-300: Valid functions, simple strategies (always move one direction)
- Steps 300-600: More complex logic (checking for merges, avoiding corners)
- Steps 600-1000: Refined strategies attempting to maximize score

## Troubleshooting

**Training is slow (>2 minutes per step):**
- First step includes compilation (~2-3 minutes normal)
- Check GPU utilization: `nvidia-smi`
- Verify no other processes using GPU

**Model keeps generating invalid syntax:**
- Wait until step 150-200 before judging
- Check `rewards/function_works/mean` - should trend toward +1.0

**All strategies timeout:**
- Normal in early training (steps 0-100)
- Model is likely generating infinite loops
- Should improve after step 150-200

**Rewards stuck at -3.0:**
- Check if moves are valid (W/A/S/D not w/a/s/d)
- Verify game controls match prompt (we fixed this!)
- Ensure `reward_std > 0` - without variance, GRPO can't learn

**CUDA out of memory:**
- Reduce `per_device_train_batch_size` in gpt_2048_rl.py:437
- Reduce `gradient_accumulation_steps` (but keep ≥2 for GRPO)
- Enable `offload_embedding=True` (already enabled)

**Training interrupted:**
- Training auto-saves checkpoints every 100 steps to `outputs/`
- Resume by modifying `main()` to load from checkpoint
- See Hugging Face Trainer documentation for resume syntax

## Project Structure

```
.
├── gpt_2048_rl.py              # Main training script
├── setup.sh                     # Environment setup script
├── requirements.txt             # Python dependencies
├── CLAUDE.md                    # Developer guidance for Claude Code
├── README.md                    # This file
├── Dockerfile                   # Docker configuration
├── gpt_oss_20B_RL_2048_Game.ipynb  # Jupyter notebook (optional)
├── output.log                   # Training logs (generated)
├── outputs/                     # Model checkpoints (generated)
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── ...
└── finetuned_model/            # Final saved model (optional)
```

**Key files:**
- `gpt_2048_rl.py:22-233` - 2048 game implementation
- `gpt_2048_rl.py:235-254` - Strategy execution with timeout
- `gpt_2048_rl.py:271-352` - Three reward functions
- `gpt_2048_rl.py:358-382` - Model loading with LoRA
- `gpt_2048_rl.py:389-411` - Dataset creation
- `gpt_2048_rl.py:418-463` - GRPO training configuration

## Performance Metrics

**Hardware requirements:**
- GPU: NVIDIA GB10 (Blackwell) or equivalent with 120GB+ VRAM
- Training time: ~85-90 seconds per step
- Total training time: 25-27 hours for 1,000 steps
- Checkpoint size: ~500MB per checkpoint
- Final model size: ~5GB (mxfp4) or ~40GB (16-bit)

**Model specifications:**
- Base model: GPT-OSS-20B (20 billion parameters)
- Trainable parameters: ~2M (0.01% with LoRA)
- LoRA rank: 4
- Quantization: 4-bit loading
- Max sequence length: 768 tokens
- Completion length: 200 tokens

## Citation

If you use this project, please cite:

```bibtex
@misc{unsloth2024gpt2048rl,
  title={Fine-tuning GPT-OSS for 2048 Game with GRPO},
  author={Unsloth AI},
  year={2024},
  howpublished={\url{https://unsloth.ai/docs/basics/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth}}
}
```

## License

This project uses the Unsloth library and GPT-OSS model. Please refer to their respective licenses for usage terms.