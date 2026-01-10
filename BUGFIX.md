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

## 11. Model Collapsed to Local Optimum (return 'W')

**Problem:** Model learned to output only simple `return 'W'` (or 'A', 'S', 'D') strategies and stopped improving. GRPO had no gradient to learn from.

**Symptoms:**
- All completions exactly 14 tokens: `return 'W'`
- `reward: 4.0` constant for all generations
- `reward_std: 0.0` - NO variance in rewards
- `frac_reward_zero_std: 1.0` - 100% batches have zero variance
- `kl: 0.41` - model diverged far from base
- Training steps very fast (~1.1s) but no learning

**Root Cause:** The `strategy_succeeds` reward function gave a flat +2.0 for any function that runs, regardless of game score:
```python
# Old code - no variance!
if game_state == "success":
    scores.append(20.0)
else:
    scores.append(2.0)  # Same reward for score=0 and score=1000
```

All simple strategies got identical rewards:
- `function_works`: +1.0
- `no_cheating`: +1.0
- `strategy_succeeds`: +2.0
- **Total: +4.0** (always the same)

GRPO requires reward variance to compute advantages. With `reward_std = 0`, the algorithm cannot determine which strategies are better.

**Solution:** Replace flat reward with score-based reward that creates variance:
```python
if game_state == "success":
    scores.append(15.0)  # Reaching 2048
else:
    # Normalized linear: score/1000, capped at 5.0
    # Creates variance: score 0 -> 0.0, score 1000 -> 1.0, score 5000+ -> 5.0
    score_reward = min(5.0, game.score() / 1000)
    scores.append(score_reward)
```

**Reward Balance:**
The three reward functions are now balanced:

| Scenario | function_works | no_cheating | strategy_succeeds | **Total** |
|----------|----------------|-------------|-------------------|-----------|
| Syntax error | -2.0 | - | - | **-2.0** |
| Cheating code | +1.0 | -20.0 | - | **-19.0** |
| `return 'W'` (score 0) | +1.0 | +1.0 | 0.0 | **+2.0** |
| Simple (score 500) | +1.0 | +1.0 | 0.5 | **+2.5** |
| Decent (score 2000) | +1.0 | +1.0 | 2.0 | **+4.0** |
| Good (score 5000+) | +1.0 | +1.0 | 5.0 | **+7.0** |
| Wins (2048) | +1.0 | +1.0 | 15.0 | **+17.0** |

**Key insight:** Game score in 2048 is the sum of all merged tile values (not sum of grid). A strategy that doesn't create merges scores 0. This naturally creates variance between strategies.

**Files Changed:** `gpt_2048_rl.py` lines ~458-467

---

## 12. Model Collapsed to Garbage Output (return "{", return "<", etc.)

**Problem:** After 1000 steps of training, model outputs garbage like `return "{"`, `return "?"`, `return "<"` instead of valid moves (W/A/S/D). Model cycles through ASCII symbols.

**Symptoms:**
- Model outputs sequences of return statements with random symbols
- `clipped_ratio: 1.0` - all completions truncated at 512 tokens
- `reward: 2.0` constant - same as valid `return "W"`
- Game scores very low (0-96) because moves are invalid
- Valid moves briefly appeared at steps 100-200, then collapsed

**Root Cause:** The `strategy_succeeds` function gave **0 reward** for invalid moves instead of penalizing them:
```python
# Bug: Invalid moves get 0, total reward still +2.0
if test_move not in ["W", "A", "S", "D"]:
    scores.append(0)  # No penalty!
```

With `function_works` (+1.0) and `no_cheating` (+1.0), the total reward was +2.0 for both:
- Valid `return "W"` with score 0: +1.0 + 1.0 + 0.0 = **+2.0**
- Garbage `return "{"`: +1.0 + 1.0 + 0.0 = **+2.0**

The model had no gradient to distinguish between valid and invalid moves!

**Solution:** Penalize invalid outputs heavily in `strategy_succeeds`:
```python
# Fix: Invalid moves get -3.0, making total reward -1.0
if test_move not in ["W", "A", "S", "D"]:
    scores.append(-3.0)  # Penalize invalid moves HEAVILY

# Also increased score rewards with lower divisor for early progress
score_reward = min(10.0, game.score() / 200)  # Rewards early progress more
```

**New Reward Balance:**

| Scenario | function_works | no_cheating | strategy_succeeds | **Total** |
|----------|----------------|-------------|-------------------|-----------|
| Garbage `return "{"` | +1.0 | +1.0 | -3.0 | **-1.0** |
| `return "W"` (score 0) | +1.0 | +1.0 | 0.0 | **+2.0** |
| Early (score 100) | +1.0 | +1.0 | 0.5 | **+2.5** |
| Simple (score 500) | +1.0 | +1.0 | 2.5 | **+4.5** |
| Decent (score 1000) | +1.0 | +1.0 | 5.0 | **+7.0** |
| Good (score 2000+) | +1.0 | +1.0 | 10.0 | **+12.0** |
| Wins (2048) | +1.0 | +1.0 | 20.0 | **+22.0** |

Now garbage outputs get -1.0 while valid outputs get +2.0 to +22.0 - a clear gradient for learning.

**Files Changed:** `gpt_2048_rl.py` lines ~397-480

---

## 13. Model Stuck in "return W" Local Minimum

**Problem:** Model converged to simple single-direction strategies like `return "W"` and stopped improving. KL divergence spiked to 0.6-1.5+ (should be <0.1).

**Symptoms:**
- All strategies are simple `return "W"`, `return "A"`, `return "S"`, `return "D"`
- Score-based rewards: `return "W"` scores 340 points (1.7 reward)
- No incentive to try more complex strategies
- KL divergence very high (0.6-1.5+) indicating model diverged too far from base
- Model oscillates between W/A/S/D but never learns better strategies

**Root Cause:** Score-based rewards (`score / 200`) rewarded quantity of merges, not quality. Simple `return "W"` strategy works surprisingly well because:
1. On 6x6 board, pressing W 500 times creates many merges as tiles stack at top
2. Gets score 340+ which translates to 1.7 reward
3. More complex strategies often fail and get negative rewards
4. Model learns "simple = safe = good reward"

**Solution:** Replace score-based rewards with **tile-based rewards** that reward the highest tile achieved:
```python
# Old: score-based (allows "return W" to score well)
score_reward = min(10.0, game.score() / 200)

# New: tile-based (caps "return W" at max tile 16-32)
max_tile = max(max(row) for row in game.board())
if max_tile >= 8:
    level = int(math.log2(max_tile)) - 2  # 8->1, 16->2, 32->3, 64->4...
    tile_reward = float(level)
```

**Why this works:**
- `return "W"` can only reach max tile 16-32 (board fills from bottom)
- Smarter strategies that combine moves can reach 64, 128, 256+
- Clear gradient: reaching 64 (reward 4) is 2x better than reaching 16 (reward 2)

**New Reward Balance:**

| Strategy | Max Tile | Reward | Total |
|----------|----------|--------|-------|
| `return "W"` | 16 | 2.0 | +4.0 |
| `return "W"` | 32 | 3.0 | +5.0 |
| Smart strategy | 64 | 4.0 | +6.0 |
| Better strategy | 128 | 5.0 | +7.0 |
| Good strategy | 256 | 6.0 | +8.0 |
| Win (2048) | 2048 | 20.0 | +22.0 |

**Files Changed:** `gpt_2048_rl.py` lines ~466-482

---

## 14. Model Collapse Due to Low Variance (frac_reward_zero_std: 1.0)

**Problem:** Even with tile-based rewards, model collapsed to outputting identical `return "W"` for all generations. GRPO had zero variance to learn from.

**Symptoms:**
- `frac_reward_zero_std: 1.0` - 100% of batches have zero reward variance
- `reward_std: 0.0` - No variance at all
- All completions identical: `return "W"` repeated
- Max tile dropped from 512 (early training) to just 8
- Model outputs same strategy verbatim across all generations

**Root Cause:** Hyperparameters caused rapid collapse:
1. `num_generations=2` - Only 2 samples per batch, not enough for variance
2. `temperature=1.0` - Not enough randomness to explore different strategies
3. `learning_rate=5e-5` - Too aggressive, model converges too fast to local minimum

**Solution:** Adjust hyperparameters to encourage exploration and prevent collapse:
```python
training_args = GRPOConfig(
    temperature=1.3,      # Was 1.0 - higher temp forces diverse outputs
    learning_rate=2e-5,   # Was 5e-5 - slower learning prevents collapse
    num_generations=4,    # Was 2 - more samples = more variance
    # ... rest unchanged
)
```

## 15. Garbage output

**Problem:** The model outputs gabage like "he advisingotwork_s(ipSource уperson composition حق接受的答案 ignoret	forcei.Reverse clear精细化 reader	Car minute葭waiting_constעבורcreator nom exemptions statusטר้ cerциoğlu	border ghost营이고vrՀ consul layer_man deteriorate advance...)"

**Symptoms:**
- Incoherent output

**Root Cause:** Hyperparameters caused rapid collapse:
1. `temperature=1.3` - Temperature is too high


**Solution:** Adjust temperaturerparameters to encourage exploration and prevent collapse:
```python
training_args = GRPOConfig(
    temperature=1.1,      # Was 1.3 - higher temp forces diverse outputs
   
    # ... rest unchanged
)
```

**Why these values:**
- **temperature=1.3**: Forces the model to generate more diverse outputs, breaking the "all identical" pattern
- **learning_rate=2e-5**: Slower learning rate (2.5x slower) gives the model more time to explore before converging
- **num_generations=4**: With 4 samples per prompt instead of 2, much higher probability of getting different outputs and reward variance

**Trade-offs:**
- Training is ~2x slower (4 generations vs 2)
- But much less likely to collapse to a local minimum
- Early training showed max tile 512 is achievable - we need to preserve that exploration

**Files Changed:** `gpt_2048_rl.py` lines ~573-583

---

## 16. Persistent Collapse to Trivial Strategies Despite All Fixes

**Problem:** Despite all previous fixes (tile-based rewards, invalid move penalties, hyperparameter tuning), the model still collapses to trivial single-direction strategies like `return "A"` or `return "W"`.

**Symptoms:**
- `frac_reward_zero_std: 1.0` - 100% of batches have zero variance
- `completions/mean_length: 14.0` - all outputs exactly 14 tokens
- All completions identical: `return "A"` repeated verbatim
- Max tile stuck at 8-16 (never improves beyond simple strategy results)
- Step ~543 and still collapsed despite hyperparameter tuning

**Root Cause:** The trivial strategy `return "A"` achieves a positive total reward:
- `function_works`: +1.0 (valid syntax)
- `no_cheating`: +1.0 (no forbidden imports)
- `strategy_succeeds`: +1.0 to +2.0 (valid move, achieves tile 8-16)
- **Total: +3.0 to +4.0**

This is a stable local optimum. Once the model converges to it:
1. All 4 generations produce identical output
2. `reward_std = 0` (zero variance)
3. GRPO cannot compute gradients without variance
4. Model stays stuck indefinitely

Previous solutions (reward tweaks, hyperparameters) failed because they didn't address the core issue: **trivial strategies are valid and receive positive rewards**.

**Solution:** Add a 4th reward function `complexity_reward()` that explicitly penalizes trivial strategies and rewards complexity:

```python
def complexity_reward(completions, **kwargs):
    """Prevents collapse to trivially simple strategies"""
    scores = []
    for completion in completions:
        function = extract_function(completion[0]["content"])
        if function is None:
            scores.append(0.0)
            continue

        reward = 0.0

        # 1. LENGTH PENALTY/BONUS
        func_len = len(function)
        if func_len < 30:      # Very short (e.g., `return "A"`)
            reward -= 3.0      # Strong penalty
        elif func_len < 80:    # Short but has some logic
            reward -= 1.0
        elif func_len < 300:   # Good length with logic
            reward += 1.0
        else:
            reward += 0.5

        # 2. COMPLEXITY BONUS
        has_if = "if " in function and ":" in function
        has_for = "for " in function and " in " in function
        has_while = "while " in function and "True" not in function
        complexity_count = sum([has_if, has_for, has_while])
        if complexity_count >= 2:
            reward += 2.0
        elif complexity_count == 1:
            reward += 1.0

        # 3. DIVERSITY BONUS
        moves_present = sum(['"W"' in function or "'W'" in function,
                            '"A"' in function or "'A'" in function,
                            '"S"' in function or "'S'" in function,
                            '"D"' in function or "'D'" in function])
        if moves_present >= 3:
            reward += 2.0
        elif moves_present >= 2:
            reward += 1.0
        elif moves_present == 1:
            reward -= 1.0      # Only one direction - trivial

        scores.append(reward)
    return scores
```

**New Reward Balance:**

| Scenario | function_works | no_cheating | strategy_succeeds | complexity_reward | **Total** |
|----------|----------------|-------------|-------------------|-------------------|-----------|
| `return "A"` (trivial) | +1.0 | +1.0 | +1.0 (tile 8) | -3.0 -1.0 = -4.0 | **-1.0** |
| Simple if-else (2 moves) | +1.0 | +1.0 | +2.0 (tile 16) | -1.0 +1.0 +1.0 = +1.0 | **+5.0** |
| Complex strategy (3+ moves) | +1.0 | +1.0 | +4.0 (tile 64) | +1.0 +2.0 +2.0 = +5.0 | **+11.0** |

Now trivial strategies get **negative total reward** (-1.0) while complex strategies get much higher rewards (+11.0). This creates a strong gradient away from the local optimum.

**Key insight:** Previous solutions tried to make the local optimum less attractive by adjusting game rewards. This solution makes trivial strategies **explicitly undesirable** regardless of game performance.

**Files Changed:** `gpt_2048_rl.py`:
- Added `complexity_reward()` function (lines ~493-559)
- Added to `reward_funcs` list in GRPOTrainer (line ~669)

---

## 17. Deterministic Strategy Collapse Despite Complexity Reward

**Problem:** Despite the complexity_reward preventing trivial `return "A"` strategies, the model still collapsed to identical outputs across all generations, resulting in zero variance.

**Symptoms:**
- Model collapsed at step ~265 (epoch 0.53)
- `frac_reward_zero_std: 1.0` from step 265 to 1000
- `completions/mean_length: 64.0` (all identical)
- All 4 generations produce the exact same strategy
- `complexity_reward/mean: 5.0` (maximum score)

**Final collapsed strategy:**
```python
def strategy(board):
    count = sum(row.count(2) for row in board)
    if count == 6:
        return 'W'
    elif count == 4:
        return 'A'
    elif count == 2:
        return 'S'
    return 'D'
```

**Training results (improved vs previous runs):**
- Max tile 512: 3 times
- Max tile 256: 16 times
- Max tile 128: 216 times
- Max tile 64: 929 times
- Training time: 1h 54m (3.93s/step)

**Root Cause:** The complexity_reward successfully prevented trivial single-line strategies, but the model found a **new local optimum**:

| Check | Score | Why it passes |
|-------|-------|---------------|
| Length (~160 chars) | +1.0 | Has if/elif/else structure |
| Complexity (has `if`) | +1.0 | Multiple conditionals |
| Diversity (4 moves) | +2.0 | Uses W, A, S, D |
| **Total** | **+5.0** | Maximum complexity reward |

The strategy is **deterministic**: given the same board state, it always returns the same move. Since all 4 generations in GRPO see the same prompt (and evaluate on the same game seed), they all produce identical outputs → `reward_std = 0` → no learning.

**Key insight:** The complexity_reward ensures the model generates **structurally complex code**, but it doesn't prevent **output collapse** when:
1. All generations see the same input
2. Model learns a deterministic mapping from input to output
3. All outputs become identical

This is a fundamental limitation of GRPO with a single repeated prompt. The model needs **input diversity** (different prompts or game states per generation) to maintain output diversity.

**Partial success:** Despite the collapse, this run showed significant improvement:
- Model learned to generate multi-directional strategies (not just `return "A"`)
- Achieved higher max tiles (512) than previous runs
- The complexity reward successfully shaped code structure

**Potential solutions (not yet implemented):**
1. **Different game seeds per generation** - Each generation sees a different board state
2. **Prompt variations** - Include random examples or hints in prompt
3. **Explicit output diversity reward** - Penalize generations that are too similar to each other
4. **Higher temperature with nucleus sampling** - Force more randomness in generation
5. **Rejection sampling** - Filter out duplicate strategies before computing rewards

**Files Changed:** None (documentation only - analyzing results of Bug #16 fix)

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
| Model collapse (local opt) | Flat reward regardless of score | No learning | Score-based reward with variance |
| Garbage output collapse | Invalid moves get 0 not penalty | Model outputs `return "{"` | Penalize invalid moves (-3.0) |
| "return W" local minimum | Score rewards quantity not quality | Model stuck at simple strategies | Tile-based rewards (max tile level) |
| Zero variance collapse | Low temp + few generations + high LR | frac_reward_zero_std: 1.0 | temp=1.3, LR=2e-5, num_gen=4 |
| Persistent trivial collapse | Trivial strategies get positive reward | Model stuck at `return "A"` | complexity_reward() function |
| Deterministic strategy collapse | All generations see same input | Identical outputs, zero variance | **Unsolved** - need input diversity |

---

## Recommended Configuration

For NVIDIA GB10 (Blackwell) with 128GB unified memory:

```python
# Model
model_name = "unsloth/Qwen2.5-1.5B-Instruct"
max_seq_length = 768
load_in_4bit = False
dtype = torch.bfloat16

# Training - tuned to prevent collapse
temperature = 1.1              # Balanced (1.3 causes garbage, 1.0 causes collapse)
learning_rate = 2e-5           # Lower to prevent rapid collapse
per_device_train_batch_size = 8
gradient_accumulation_steps = 1
num_generations = 4            # More samples for variance
max_completion_length = 512

# 4 Reward Functions (order matters)
reward_funcs = [
    function_works,      # +1.0 valid / -2.0 invalid
    no_cheating,         # +1.0 clean / -20.0 cheating
    strategy_succeeds,   # Tile-based: 0 to +20.0
    complexity_reward,   # Penalize trivial, reward complex (-4.0 to +5.0)
]

# Safety
execute_with_time_limit(2)  # 2 second timeout
MAX_STEPS = 500  # Hard limit in game execution
# Block: input(), open(), exec(), eval(), __import__
# Block: while True:, while 1:, while(True)
```

Expected training time: ~3-4 hours for 1000 steps.
