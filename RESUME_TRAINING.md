# How to Resume Training from Checkpoint

## Summary of Fixes

### Bug #1: UnboundLocalError in `function_works()` (CRITICAL)
**Problem:** When `function is None`, the variable `info` was never defined, causing a crash when checking `"error" in info`.

**Fixed:** Added proper control flow with early returns using `continue` statements.

### Bug #2: Same issue in `strategy_succeeds()`
**Problem:** Same undefined variable bug.

**Fixed:** Added defensive checks and proper early returns.

### Enhancement: Better Logging
Added detailed logging to monitor training progress:
- Full completions printed every 5 steps (first 800 chars)
- Generation numbers (1/2 for GRPO)
- Clear emoji-based error messages
- Better game state reporting (Steps | State | Score)

---

## Resume Training from Checkpoint 200

You stopped at step 177. The nearest checkpoint is at step 100. However, since the bug was present during that training, it's recommended to **start fresh** or resume cautiously.

### Option 1: Start Fresh (Recommended)

```bash
# Set environment variables
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST=12.1a

# Create screen session
screen -S training2048

# Activate venv
source .venv/bin/activate

# Rename old outputs to preserve them
mv outputs outputs_buggy_run1
mv output.log output_buggy_run1.log

# Start fresh training with fixes
python gpt_2048_rl.py 2>&1 | tee output.log
```

### Option 2: Resume from Checkpoint 100

If you want to continue from checkpoint 100:

1. First, backup the current state:
```bash
cp -r outputs outputs_backup
cp output.log output_backup.log
```

2. Modify `gpt_2048_rl.py` to add resume functionality:

```python
def main():
    """Main training pipeline"""
    print("Loading model...")
    model, tokenizer, max_seq_length = load_model()

    print("Creating dataset...")
    dataset, prompt = create_dataset()

    print("Starting training...")
    trainer = train_model(
        model,
        tokenizer,
        dataset,
        max_seq_length,
        resume_from_checkpoint="outputs/checkpoint-100"  # ADD THIS LINE
    )

    print("\nTraining complete! Testing model...")
    test_model(model, tokenizer, prompt)
```

3. Then update `train_model()` to accept the resume parameter:

```python
def train_model(model, tokenizer, dataset, max_seq_length, resume_from_checkpoint=None):
    """Set up and run GRPO training"""
    # ... existing code ...

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            function_works,
            no_cheating,
            strategy_succeeds,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # Resume from checkpoint if provided
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer
```

---

## What to Monitor

With the new logging, watch for these signs of progress:

### Good Signs (Learning)
✅ Strategies playing **> 1 step** (currently all fail at step 1)
✅ `rewards/strategy_succeeds/mean` increasing from -1.0 toward 0.0 then +2.0
✅ `reward_std > 0` consistently (variance needed for GRPO)
✅ Generated functions evolving beyond `return "W"`
✅ Functions using logic: `if`, loops, board inspection
✅ KL divergence staying < 0.1

### Bad Signs (Stuck)
❌ Still outputting `return "W"` after 50+ steps
❌ All timeouts continuing
❌ `reward_std = 0` (no variance)
❌ KL divergence > 0.5 (mode collapse)
❌ Rewards not improving after 100 steps from resume point

---

## Expected Timeline

**With fixes:**
- Steps 0-100: Learning syntax, exploring variations
- Steps 100-200: First valid strategies that play > 1 step
- Steps 200-400: Strategies using board state, playing 5-20 steps
- Steps 400-600: More complex logic, playing 50-200 steps
- Steps 600-1000: Optimized strategies, occasional high tiles (512, 1024)

**Target at step 300 from resume (step 400 total):**
- `reward`: +2.0 to +3.0
- `rewards/strategy_succeeds/mean`: +2.0 (valid gameplay)
- Strategies playing 10-50 steps on average
- Clear evolution in generated code

---

## If Training Still Stalls

If after resuming from checkpoint 100 you reach step 250 (150 new steps) and see:
- Still outputting `return "W"`
- All timeouts
- No improvement in rewards

Then **stop and investigate:**
1. Check if the model checkpoint was corrupted
2. Consider starting completely fresh
3. Try adjusting hyperparameters:
   - Increase `temperature` to 1.5 for more exploration
   - Increase `max_completion_length` to 300
   - Increase `num_generations` to 4 (more advantage samples)

---

## Quick Start Command

```bash
# Safest approach - start fresh with fixes
screen -S training2048
source .venv/bin/activate
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST=12.1a
mv outputs outputs_buggy && python gpt_2048_rl.py 2>&1 | tee output.log
```

Press `Ctrl+A` then `D` to detach from screen.
Use `screen -r training2048` to reattach.
