# BUGFIX.md - Issues Solved During Development

This document summarizes all bugs and performance issues encountered and fixed during the development of the GPT-2048 reinforcement learning project on NVIDIA GB10 (Blackwell) GPU.

---

## 1. xformers Not Working on Blackwell GPU

**Problem:** The pip-installed xformers (0.0.33.post2) didn't have CUDA kernels compiled for Blackwell (compute capability 12.1). Training fell back to slow eager attention.

**Symptoms:**
- Slow training speed
- Log showed: `FA [Xformers = 0.0.33.post2. FA2 = False]`
- `memory_efficient_attention` failed with "requires device with capability <= (9, 0)"

**Solution:** Build xformers v0.0.33 from source with Blackwell support:
```bash
cd /tmp
git clone --depth=1 --branch v0.0.33 --recursive https://github.com/facebookresearch/xformers.git
cd xformers
TORCH_CUDA_ARCH_LIST="12.1" MAX_JOBS=4 uv pip install --no-build-isolation -e . --python /path/to/venv/bin/python
```

**Files Changed:** `setup_blackwell.sh` created

---

## 2. Slow Inference Speed

**Problem:** Token generation during `test_model()` was very slow compared to vLLM or llama.cpp.

**Symptoms:**
- Slow token-by-token generation
- High latency per token

**Solution:** Enable Unsloth's fast inference mode:
```python
# In test_model():
FastLanguageModel.for_inference(model)
```

**Files Changed:** `gpt_2048_rl.py` line ~505

---

## 3. Completions Truncated (clipped_ratio = 1.0)

**Problem:** All model completions were being truncated at the token limit, never finishing naturally.

**Symptoms:**
- `completions/clipped_ratio: 1.0` (100% truncated)
- `completions/mean_terminated_length: 0.0` (no natural completions)
- Functions cut off mid-code: `for col` (truncated)

**Root Cause:** `max_completion_length` was too small:
- Initially set to 200 tokens
- GPT-OSS outputs ~150-250 tokens of reasoning BEFORE code
- Qwen outputs ~100 tokens explanation before code

**Solution:** Increase completion length progressively:
```python
# For reasoning models (GPT-OSS):
max_completion_length = 450

# For instruction models (Qwen):
max_completion_length = 512

# Also increase max_seq_length:
max_seq_length = 768
```

**Files Changed:** `gpt_2048_rl.py` lines ~404, ~474

---

## 4. Chat Template Missing for Base Models

**Problem:** Base Qwen models don't have chat templates, causing tokenizer errors.

**Symptoms:**
```
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set
```

**Solution:** Use Instruct versions of models:
```python
# Wrong:
model_name = "unsloth/Qwen2.5-1.5B"

# Correct:
model_name = "unsloth/Qwen2.5-1.5B-Instruct"
```

**Files Changed:** `gpt_2048_rl.py` line ~412

---

## 5. Function Extraction Too Strict

**Problem:** Many valid functions weren't being extracted because they didn't have proper markdown backticks.

**Symptoms:**
- Model outputs valid `def strategy(board):` but without backticks
- `extract_function()` returns `None`
- "Failed to extract function from response"

**Solution:** Add fallback extraction that finds `def strategy(board):` anywhere in text:
```python
def extract_function(text):
    # Try backticks first
    if text.count("```") >= 2:
        # ... existing logic

    # Fallback: find function anywhere
    if "def strategy(board):" in text:
        start = text.find("def strategy(board):")
        # Extract until next top-level definition
        # ...
```

**Files Changed:** `gpt_2048_rl.py` lines ~260-285

---

## 6. Model Outputs Wrong Move Characters

**Problem:** Model outputs "U", "R", "L", "B", "NA" instead of "W", "A", "S", "D".

**Symptoms:**
- Functions return invalid moves
- Game immediately fails with `state = "failed"`

**Solution:** Improve prompt to be explicit:
```python
prompt = """```python
def strategy(board):
    # 2048 game: board is 6x6 list of lists (0=empty, 2/4/8/...=tiles)
    # Return EXACTLY one of: "W" "A" "S" "D" (up/left/down/right)
    # DO NOT use external functions. Keep it simple.
""".strip()
```

**Files Changed:** `gpt_2048_rl.py` lines ~447-452

---

## 7. Low GPU Utilization (Oscillating 0-86%)

**Problem:** GPU utilization dropped to 0% frequently, wasting compute.

**Symptoms:**
- Only 17GB/128GB memory used
- GPU utilization oscillating between 0% and 86%
- Long idle periods between batches

**Root Cause:** Small batch sizes and CPU-bound reward computation between GPU batches.

**Solution:** Increase batch size and parallelism:
```python
# Before:
per_device_train_batch_size = 1
num_generations = 2

# After:
per_device_train_batch_size = 8
num_generations = 2

# Also disable 4-bit quantization (you have memory):
load_in_4bit = False
dtype = torch.bfloat16
```

**Files Changed:** `gpt_2048_rl.py` lines ~417, ~503-505

---

## 8. Training Stuck - input() Blocking

**Problem:** Training hung indefinitely with no progress for over an hour.

**Symptoms:**
- GPU utilization at 0%
- No new log output
- Last log line: `Enter a direction (W, A, S, D):`

**Root Cause:** Model generated code containing `input()` which blocks waiting for user input. The timeout mechanism doesn't interrupt blocking I/O operations.

**Solution:** Block dangerous functions before execution:
```python
dangerous_calls = ["input(", "open(", "exec(", "eval(", "__import__"]
if any(call in function for call in dangerous_calls):
    scores.append(-20.0)  # Heavily penalize
    continue
```

**Files Changed:** `gpt_2048_rl.py` - Added checks in all 3 reward functions:
- `function_works()` lines ~294, ~304-307
- `no_cheating()` lines ~326, ~331-334
- `strategy_succeeds()` lines ~369-374

---

## 9. Slow Reward Computation (5s Timeouts)

**Problem:** Each step took minutes due to many 5-second timeouts.

**Symptoms:**
- Many "Timeout (5s exceeded)" messages
- Steps taking 5+ minutes
- 64 function evaluations per step × 5s = 320s worst case

**Solution:** Reduce timeout and batch size:
```python
# Reduce timeout from 5s to 2s:
@execute_with_time_limit(2)  # Was 5
def execute_strategy(strategy, game):
    ...

# Reduce evaluations per step:
per_device_train_batch_size = 8   # Was 16
num_generations = 2               # Was 4
# Total: 16 evaluations instead of 64
```

**Files Changed:** `gpt_2048_rl.py` lines ~251, ~503-505

---

## 10. Infinite Loops Bypass Timeout (while True)

**Problem:** Training hangs for 50+ minutes despite 2-second timeout. CPU at 96% with no progress.

**Symptoms:**
- Process running at high CPU
- No new log output for 50+ minutes
- Timeout mechanism not triggering

**Root Cause:** Python's signal-based timeout doesn't interrupt tight CPU loops. Generated code with `while True:` loops runs forever:
```python
def strategy(board):
    while True:  # This bypasses the timeout!
        if some_condition:
            return "W"
```

**Solution:**
1. Block `while True:` patterns before execution:
```python
infinite_loop_patterns = ["while True:", "while 1:", "while True :", "while(True)"]
if any(pattern in function for pattern in infinite_loop_patterns):
    scores.append(-20.0)
    continue
```

2. Add hard step limit in game execution:
```python
MAX_STEPS = 500
while game.state() == "ongoing":
    if steps >= MAX_STEPS:
        return steps, "failed"
    # ...
```

**Files Changed:** `gpt_2048_rl.py`:
- `_execute_strategy()` - Added MAX_STEPS=500 limit
- `function_works()` - Added infinite loop pattern check
- `no_cheating()` - Added infinite loop pattern check
- `strategy_succeeds()` - Added infinite loop pattern check

---

## Summary Table

| Issue | Root Cause | Impact | Fix |
|-------|------------|--------|-----|
| Slow attention | xformers not compiled for Blackwell | 2x slower training | Build from source |
| Slow inference | Missing `for_inference()` | 2-3x slower | Add call |
| Truncated completions | `max_completion_length` too small | No valid code | Increase to 512 |
| Chat template error | Using base model | Crash | Use Instruct model |
| Function extraction fails | Strict backtick requirement | Many failures | Add fallback |
| Wrong move characters | Unclear prompt | Invalid moves | Explicit prompt |
| Low GPU utilization | Small batches | Wasted compute | Increase batch size |
| Training hangs (input) | `input()` in generated code | Infinite hang | Block dangerous calls |
| Slow steps | 5s timeouts × many evals | 5+ min/step | Reduce timeout & batch |
| Training hangs (loops) | `while True:` bypasses timeout | 50+ min hang | Block loop patterns + step limit |

---

## Recommended Configuration

For NVIDIA GB10 (Blackwell) with 128GB unified memory:

```python
# Model
model_name = "unsloth/Qwen2.5-1.5B-Instruct"
max_seq_length = 768
load_in_4bit = False
dtype = torch.bfloat16

# Training
per_device_train_batch_size = 8
gradient_accumulation_steps = 1
num_generations = 2
max_completion_length = 512

# Safety
execute_with_time_limit(2)  # 2 second timeout
MAX_STEPS = 500  # Hard limit in game execution
# Block: input(), open(), exec(), eval(), __import__
# Block: while True:, while 1:, while(True)
```

Expected training time: ~3-4 hours for 1000 steps.
