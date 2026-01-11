# Training Collapse Analysis: The Variance Problem

## Executive Summary

The model repeatedly collapses to identical outputs across all 4 generations, resulting in `reward_std = 0.0` and `frac_reward_zero_std = 1.0`, preventing GRPO from learning. This occurs regardless of complexity_reward, frozen state detection, or other fixes.

**Root Cause**: GRPO requires reward variance to compute gradients, but training uses 1000 identical prompts. Once the model learns a deterministic response to this prompt, output diversity disappears and learning stops.

**Key Insight**: This is not a bug in GRPO or the reward functions - it's a fundamental dataset design issue. GRPO is working exactly as intended, but the single-prompt dataset cannot sustain the variance needed for continued learning.

---

## The Core Problem: Identical Prompts → Deterministic Outputs → Zero Variance

### How GRPO Works

GRPO (Group Relative Policy Optimization) computes advantages by comparing rewards **within a batch**:

```python
# For each prompt, generate 4 completions
completions = model.generate(prompt, num_generations=4)

# Compute rewards
rewards = [reward_func(c) for c in completions]

# Compute advantages (relative to batch mean)
mean_reward = sum(rewards) / len(rewards)
advantages = [r - mean_reward for r in rewards]

# Update policy based on advantages
# If all rewards are identical: advantages = [0, 0, 0, 0] → NO GRADIENT!
```

### Current Dataset Structure

From `gpt_2048_rl.py:883-888`:

```python
dataset = Dataset.from_list([
    {
        "prompt": [{...same prompt...}],
    }
] * 1000)  # ← 1000 copies of IDENTICAL prompt
```

**Every training step uses the exact same prompt.** The model sees:
- Step 1: "Write a Python strategy function for 2048..." (×4 generations)
- Step 2: "Write a Python strategy function for 2048..." (×4 generations)
- Step 3: "Write a Python strategy function for 2048..." (×4 generations)
- ...
- Step 1000: "Write a Python strategy function for 2048..." (×4 generations)

### The Convergence Trap

Once the model learns a response that achieves reasonable reward:

```
Input: "Write a Python strategy function for 2048..." (identical)
   ↓
Model: (deterministic mapping learned)
   ↓
Output: def strategy(board): ... (identical across all 4 generations)
   ↓
Rewards: [0.5, 0.5, 0.5, 0.5]
   ↓
Variance: 0.0 → Advantages: [0, 0, 0, 0] → NO LEARNING
```

The model has successfully optimized for the training objective but has nowhere left to go.

---

## Why Previous Fixes Didn't Work

### Fix 1: Different Seeds Per Generation (INEFFECTIVE for crashes)

**Implementation**: Lines 564, 626, 708 in `gpt_2048_rl.py`

```python
seeds = [np.random.randint(10000) for _ in range(len(completions))]
game = GameBoard(size=4, seed=seeds[idx], target=2048)
```

**Why it doesn't help when strategies crash**:

```python
# Execution flow in strategy_succeeds():

# Step 1: Create function
strategy = create_locked_down_function(function)

# Step 2: Test on empty board
test_move = call_with_timeout(strategy, (test_board,), timeout=1.0)

# Step 3: If test crashes → score -1.5, skip game execution
if crash:
    scores.append(-1.5)
    continue  # ← Game with different seed NEVER CREATED

# Step 4: Only reached if test passed
game = GameBoard(size=4, seed=seeds[idx], ...)  # ← Never reached!
```

When all 4 generations crash identically (e.g., `NameError: 'simulate_move' not defined`), they all get -1.5 before any game is created. Different seeds are irrelevant because game execution never happens.

**Different seeds only create variance if strategies execute the game successfully.**

### Fix 2: Temperature Adjustment (LIMITED EFFECTIVENESS)

Temperature controls randomness in token selection:
- **Low (1.0-1.1)**: Model converges quickly to deterministic outputs
- **Medium (1.15-1.25)**: Sweet spot - some diversity without garbage
- **High (1.3+)**: Generates syntactically invalid code or nonsense

**The problem**: Even at optimal temperature, GRPO will eventually converge to a consistent response pattern for an identical prompt. Temperature only delays convergence, it doesn't prevent it.

From user feedback: "if i go to a temperature of 1.3 the model outputs garbage"

This indicates the sweet spot is very narrow (1.15-1.25 range), and even there, convergence is inevitable with identical prompts.

### Fix 3: complexity_reward (ADDRESSES WRONG PROBLEM)

**Implementation**: Lines 749-815 in `gpt_2048_rl.py`

`complexity_reward` prevents collapse to trivial strategies like `return "A"` by penalizing short, single-direction code. It successfully prevents that specific local optimum.

**Why it's insufficient**:

The model just finds a *different* local optimum that satisfies complexity requirements:
- Previous collapse: `return "A"` (reward +4.0, blocked by complexity_reward)
- New collapse: Helper function pattern (reward +0.5, passes complexity checks)

```python
def strategy(board):
    # Has loops ✓
    # Has conditionals ✓
    # Uses multiple moves ✓
    # Passes complexity_reward ✓
    for move in ["W", "A", "S", "D"]:
        score = simulate_move(board, move)  # ← Crashes, but after passing complexity checks
        if score > best_score:
            best_move = move
    return best_move
```

`complexity_reward` shapes *what* the model outputs, but doesn't prevent *all outputs being identical*.

### Fix 4: Frozen State Detection (ADDRESSES WRONG PROBLEM)

**Implementation**: Lines 307-347 in `gpt_2048_rl.py`

Frozen state detection catches strategies that make non-changing moves. It successfully prevents 500-step timeouts.

**Why it's insufficient**:

This fix addresses strategies that execute but get stuck. The current collapse (helper function pattern) crashes before any game state is created, so frozen state detection never triggers.

---

## This is Bug #17: "Unsolved - Need Input Diversity"

From `BUGFIX.md:586-648`:

> **17. Deterministic Strategy Collapse Despite Complexity Reward**
>
> Despite the complexity_reward preventing trivial strategies, the model still collapsed to identical outputs across all generations, resulting in zero variance.
>
> **Root Cause:** The strategy is deterministic: given the same board state, it always returns the same move. Since all 4 generations in GRPO see the same prompt (and evaluate on the same game seed), they all produce identical outputs → `reward_std = 0` → no learning.
>
> **Key insight:** The complexity_reward ensures the model generates structurally complex code, but it doesn't prevent output collapse when:
> 1. All generations see the same input
> 2. Model learns a deterministic mapping from input to output
> 3. All outputs become identical
>
> This is a fundamental limitation of GRPO with a single repeated prompt. The model needs **input diversity** (different prompts or game states per generation) to maintain output diversity.
>
> **Potential solutions (not yet implemented):**
> 1. Different game seeds per generation
> 2. Prompt variations
> 3. Explicit output diversity reward
> 4. Higher temperature with nucleus sampling
> 5. Rejection sampling

**Solutions 1, 3, 4 have been attempted with limited success. Solution 2 (Prompt Variations) has NOT been implemented.**

---

## Why This Keeps Happening: Timeline of Collapses

| Training Run | Collapse Point | Collapsed Pattern | Reward | Why Previous Fixes Failed |
|--------------|----------------|-------------------|--------|---------------------------|
| **Bug #19 (original)** | Never converged | Frozen states (board unchanged) | 2.0 | No complexity_reward, no frozen detection |
| **After Bug #19 fixes** | Step 120 | Adjacent tile checker (structured code) | 5.5 | complexity_reward maxed out, but all outputs identical |
| **After convergence fixes** | Step ~10 | Helper function calls (simulate_move, etc.) | 0.5 | User disabled complexity_reward, temp reverted to 1.1 |
| **Current (complexity_reward re-enabled)** | Step ~10 | Same helper function pattern | 0.5 | Identical prompts still cause deterministic outputs |

**Pattern**: The model finds *different* local optima depending on which penalties are active, but always converges because the underlying problem (identical prompts) remains unsolved.

---

## Proposed Solutions

### Solution 1: Prompt Variations (RECOMMENDED - PRIMARY FIX)

**Rationale**: Directly addresses the root cause by ensuring different inputs even within the same batch.

**Implementation**:

Modify `create_dataset()` in `gpt_2048_rl.py` around line 865:

```python
def create_dataset():
    """Create training dataset with prompt variations"""
    import random

    base_prompt = """Write a Python strategy function for the 2048 game.

RULES:
- Board: 4x4 grid, board[row][col], 0=empty, 2/4/8/16...=tiles
- Return EXACTLY one of: "W" (up), "A" (left), "S" (down), "D" (right)
- ONLY write the strategy function. NO helper functions, NO imports, NO explanations.

```python
def strategy(board):
    # Your code here - analyze board and return "W", "A", "S", or "D"
""".strip()

    # Create variations with different hints/constraints
    # These don't change the task, but provide input diversity
    variations = [
        base_prompt,  # Original

        base_prompt + "\n\nHint: Consider which direction creates the most merges.",

        base_prompt + "\n\nHint: Try to keep high-value tiles in one area of the board.",

        base_prompt + "\n\nRemember: You can only return W, A, S, or D. No helper functions allowed.",

        base_prompt + "\n\nNote: The function will be called many times per game. Keep it efficient.",

        base_prompt + "\n\nTip: Empty cells (0 values) indicate space for new tiles.",

        base_prompt + "\n\nImportant: The function must return a string, not call other functions.",

        base_prompt + "\n\nExample moves: 'W' pushes tiles up, 'A' pushes left, 'S' down, 'D' right.",
    ]

    # Create dataset with random variations
    dataset_list = []
    random.seed(42)  # Reproducible shuffling
    for i in range(1000):
        prompt_text = random.choice(variations)
        dataset_list.append({
            "prompt": [{"role": "user", "content": prompt_text}]
        })

    # Shuffle to ensure variations are distributed across training
    random.shuffle(dataset_list)

    dataset = Dataset.from_list(dataset_list)
    return dataset, base_prompt  # Return base_prompt for testing
```

**Expected Impact**:

- Within a single batch of 4 generations, model likely sees 2-4 different prompt variations
- Different inputs → model cannot learn single deterministic mapping
- Even if model learns per-variation responses, batch variance is maintained
- `frac_reward_zero_std` should drop from 1.0 to < 0.5

**Trade-offs**:

- ✅ Directly addresses root cause
- ✅ Minimal code changes
- ✅ No performance overhead
- ⚠️ Model may take longer to converge (not necessarily bad - more exploration)
- ⚠️ May need to balance variation diversity (too different could confuse training)

---

### Solution 2: Explicit Diversity Reward (SAFETY NET)

**Rationale**: Directly penalizes identical outputs, creating a gradient that encourages exploration.

**Implementation**:

Add new reward function in `gpt_2048_rl.py` after `complexity_reward()`:

```python
def diversity_reward(completions, **kwargs):
    """
    Penalize when all generations produce identical or very similar code.

    This prevents GRPO from converging to a single deterministic response
    by explicitly rewarding variety within a batch.
    """
    scores = []

    # Extract all functions
    functions = [extract_function(c[0]["content"]) for c in completions]
    functions_valid = [f for f in functions if f is not None]

    # If most extractions failed, don't apply diversity reward
    if len(functions_valid) < 2:
        return [0.0] * len(completions)

    # Count unique functions (exact string match)
    unique_funcs = set(functions_valid)
    num_unique = len(unique_funcs)
    total_valid = len(functions_valid)

    # Calculate diversity score
    if num_unique == 1:
        # All identical - heavily penalize
        diversity_score = -3.0
    elif num_unique == 2 and total_valid == 4:
        # Low diversity (2 unique out of 4)
        diversity_score = -1.0
    elif num_unique == 2 and total_valid == 3:
        # Medium diversity (2 unique out of 3)
        diversity_score = 0.0
    elif num_unique >= 3:
        # Good diversity (3+ unique patterns)
        diversity_score = +1.0
    else:
        diversity_score = 0.0

    # Apply same score to all completions in batch
    # (This creates pressure at batch level, not individual level)
    return [diversity_score] * len(completions)
```

Add to `reward_funcs` in `train_model()` around line 930:

```python
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        function_works,
        no_cheating,
        strategy_succeeds,
        complexity_reward,
        diversity_reward,  # ← ADD THIS
    ],
    args=training_args,
    train_dataset=dataset,
)
```

**Expected Impact**:

- When all 4 generations are identical: each gets -3.0 penalty → total reward reduced
- Model learns that variety is rewarded → maintains output diversity
- Works in combination with other reward functions

**Trade-offs**:

- ✅ Direct penalty for the problematic behavior
- ✅ Works regardless of prompt structure
- ⚠️ Batch-level penalty may create noisy gradients
- ⚠️ Could prevent convergence to a genuinely good strategy
- ⚠️ May reward "diversity for diversity's sake" rather than exploration

**Recommendation**: Use as a safety net alongside Solution 1, not as primary fix.

---

### Solution 3: Heavier Crash Penalties (GRADIENT SHAPING)

**Rationale**: Create stronger gradient away from crashing patterns. Currently, crashes get -1.5, which leads to total reward of +0.5. Make crashes more costly.

**Implementation**:

Modify `strategy_succeeds()` around lines 410-413, 701-705:

```python
# In test execution phase (around line 407):
try:
    test_move = call_with_timeout(new_strategy, (test_board,), timeout=1.0)
    if not isinstance(test_move, str):
        scores.append(-2.0)  # Was -1.5, more harsh
        continue
    if test_move not in ["W", "A", "S", "D"]:
        scores.append(-2.0)  # Was -1.5, more harsh
        continue
except TimeoutError:
    scores.append(-2.5)  # Was -2.0, more harsh
    continue
except Exception as e:
    # NEW: Distinguish undefined function errors from other crashes
    error_str = str(e)
    if "NameError" in error_str or "not defined" in error_str:
        # Calling undefined functions is particularly bad
        scores.append(-5.0)  # Very heavy penalty
    elif "AttributeError" in error_str:
        # Calling methods on wrong types
        scores.append(-3.0)
    else:
        # Generic crashes
        scores.append(-2.5)  # Was -1.5
    continue
```

**New Reward Balance**:

| Scenario | function_works | no_cheating | strategy_succeeds | complexity_reward | **Total** |
|----------|----------------|-------------|-------------------|-------------------|-----------|
| Helper function pattern | +1.0 | +1.0 | -5.0 (NameError) | +1.0 | **-2.0** |
| Invalid return type | +1.0 | +1.0 | -2.0 | +1.0 | **+1.0** |
| Frozen state | +1.0 | +1.0 | -2.0 (frozen) | +1.0 | **+1.0** |
| Reaches tile 8 | +1.0 | +1.0 | +1.0 | +1.0 | **+4.0** |
| Reaches tile 32 | +1.0 | +1.0 | +3.0 | +1.0 | **+6.0** |

Now the helper function pattern is clearly unprofitable (-2.0), creating stronger incentive to actually execute the game.

**Trade-offs**:

- ✅ Creates clearer gradient away from non-functional code
- ✅ Distinguishes different types of failures
- ⚠️ May discourage exploration if penalties are too harsh
- ⚠️ Doesn't address the variance problem directly

**Recommendation**: Implement alongside Solution 1 to shape the reward landscape.

---

### Solution 4: Pre-Execution Validation (CRASH PREVENTION)

**Rationale**: Catch common crash patterns before execution and penalize them explicitly.

**Implementation**:

Add validation in `strategy_succeeds()` before line 606:

```python
# After extracting function
function = extract_function(response)
if function is None:
    scores.append(-1.0)
    continue

# NEW: Check for undefined function calls BEFORE execution
# Common patterns that will crash with NameError
undefined_patterns = [
    "simulate_move", "evaluate_move", "get_possible", "calculate_",
    "apply_move", "check_move", "valid_move", "best_move",
    "score_move", "test_move", "try_move", "make_move",
]

if any(pattern in function for pattern in undefined_patterns):
    if PRINTER % 5 == 1:
        print(f"❌ Undefined function pattern detected: likely to crash with NameError")
    scores.append(-4.0)  # Heavy penalty without running
    continue

# Check for other problematic patterns
if function.count("def ") > 1:
    # Multiple function definitions (helper functions not allowed)
    scores.append(-3.0)
    continue

if "import " in function:
    # Trying to import (should be caught by no_cheating but double-check)
    scores.append(-3.0)
    continue

# ... continue with existing execution code
```

**Expected Impact**:

- Catches helper function pattern immediately without execution overhead
- Provides specific feedback about why strategy is rejected
- Reduces wasted compute on strategies that will definitely crash

**Trade-offs**:

- ✅ Fast rejection of known-bad patterns
- ✅ No execution overhead
- ✅ Can provide specific error messages
- ⚠️ Hardcoded pattern list may miss new failure modes
- ⚠️ May reject creative solutions that happen to use similar variable names
- ⚠️ Maintenance burden as new patterns emerge

**Recommendation**: Implement as optimization after Solution 1 is working.

---

## Recommended Implementation Order

### Phase 1: Address Root Cause (CRITICAL)

1. **Implement Solution 1 (Prompt Variations)** - Lines 865-889 in `gpt_2048_rl.py`
   - Create 8 prompt variations
   - Randomly sample for each training example
   - This is the PRIMARY FIX that addresses the fundamental issue

### Phase 2: Gradient Shaping (HIGH PRIORITY)

2. **Implement Solution 3 (Heavier Crash Penalties)** - Lines 410-413, 701-705
   - Distinguish NameError (-5.0) from other crashes (-2.5)
   - Create stronger gradient away from non-functional code

### Phase 3: Safety Nets (MEDIUM PRIORITY)

3. **Implement Solution 2 (Diversity Reward)** - Add after `complexity_reward()`
   - Penalize identical outputs across batch
   - Acts as failsafe if prompt variations aren't diverse enough

4. **Implement Solution 4 (Pre-Execution Validation)** - Before line 606
   - Fast rejection of known-bad patterns
   - Reduces wasted compute

### Phase 4: Monitoring (ONGOING)

Track these metrics in W&B to verify fixes are working:

**Success indicators**:
- `frac_reward_zero_std` < 0.5 (was 1.0)
- `reward_std` > 0.5 (was 0.0)
- `completions/mean_length` varies by 20+ tokens between steps
- Average reward increases over time (even slowly)
- `diversity_reward/mean` > -1.0 (if implemented)

**Warning signs**:
- `frac_reward_zero_std` → 1.0 (collapse recurring)
- `completions/mean_length` very consistent (±5 tokens)
- Reward constant for 50+ consecutive steps
- Same error message repeated in logs

---

## Expected Results After Fixes

### Early Training (Steps 1-200)

**Before (current)**:
- All outputs identical by step 10
- `reward_std: 0.0` throughout
- No exploration

**After (with prompt variations)**:
- Diverse outputs due to different prompts
- `reward_std: 0.5-2.0` (healthy variance)
- Multiple collapse attempts, each caught by diversity_reward
- Some strategies crash (-5.0), others play (+1.0 to +4.0)
- Model explores different approaches

### Mid Training (Steps 200-600)

**Before**:
- Stuck at reward +0.5 for entire period
- No learning occurs

**After**:
- Gradual convergence toward working strategies
- `reward_std: 0.3-1.0` (decreasing but non-zero)
- Average reward climbs from +0.5 to +4.0
- Max tiles increase: 8 → 16 → 32
- Diversity maintained even as quality improves

### Late Training (Steps 600-1000)

**Before**:
- Still stuck at +0.5
- Model never learned to play

**After**:
- Fine-tuning of good strategies
- `reward_std: 0.2-0.8` (small but present)
- Average reward: +4.0 to +7.0
- Some strategies reach tile 64-128
- Model outputs show variety in approach (different orderings, conditionals, priorities)

---

## Why This Is Fundamentally Different From Previous Fixes

| Previous Fixes | What They Did | Why Insufficient |
|----------------|---------------|------------------|
| Different seeds per generation | Changed game boards | Only helps if strategies execute the game |
| Temperature adjustment | Increased token randomness | Model still converges to identical outputs for identical inputs |
| complexity_reward | Penalized trivial code | Model found different (non-trivial) local optimum |
| Frozen state detection | Caught stuck strategies | Doesn't help with strategies that crash before playing |
| Heavier penalties | Made crashes worse | Creates gradient but doesn't prevent identical outputs |

| **Prompt Variations** | **What It Does** | **Why It Works** |
|----------------------|------------------|------------------|
| Provides input diversity | Changes model input within batch | Model CANNOT learn deterministic mapping from identical input |

**Key insight**: All previous fixes tried to shape the reward landscape (what the model outputs) or catch specific failure modes. Prompt variations addresses the fundamental variance problem by ensuring the model cannot simply memorize a single response.

---

## Alternative Approaches (Not Recommended)

### Approach A: Increase num_generations

```python
num_generations = 8  # or 16
```

**Why not recommended**: This doesn't solve the variance problem. With identical prompts, all 8/16 generations will still eventually converge to identical outputs. It just delays the collapse and increases compute cost.

### Approach B: Reduce learning rate further

```python
learning_rate = 5e-6  # or lower
```

**Why not recommended**: Slower learning delays convergence but doesn't prevent it. With identical prompts, the model will eventually converge to a deterministic response, just more slowly. Training time increases with no guarantee of success.

### Approach C: Curriculum learning (easy → hard prompts)

**Why not recommended**: The task itself (write a strategy function) is already simple. Making it "easier" (smaller board, lower target) doesn't address the variance problem. You'd still need prompt variations for each curriculum stage.

### Approach D: Switch to different RL algorithm (PPO, DPO, etc.)

**Why not recommended**: GRPO's within-batch comparison is not the problem - it's an efficient design. Other algorithms like PPO also require exploration, and would face the same variance issues with identical prompts.

---

## Conclusion

The repeated training collapses are not caused by bugs in GRPO, insufficient penalties, or wrong hyperparameters. **The training dataset has 1000 identical prompts**, and once the model learns a consistent response to that prompt (any response, good or bad), variance disappears and GRPO stops learning.

The fix is straightforward: **provide input diversity through prompt variations**. This is a standard practice in RL training but was overlooked in the initial implementation.

All other fixes (complexity_reward, frozen detection, crash penalties, etc.) are valuable for shaping the reward landscape, but they cannot solve the fundamental variance problem. They should be viewed as complementary to prompt variations, not alternatives.

---

## References

- `BUGFIX.md` Bug #17 (lines 586-648): First documentation of this issue as "fundamental limitation"
- `TRAINING_POSTMORTEM.md` lines 172-196: Suggested diversity reward as potential solution
- `PREMATURE_CONVERGENCE_FIX.md` lines 136-138: Attempted fixes that didn't address root cause
- `gpt_2048_rl.py` lines 883-888: Current dataset creation with identical prompts
- GRPO paper: "Group Relative Policy Optimization" - requires within-batch variance

---

## Status

**Documented**: 2026-01-11
**Implemented**: NO - awaiting user decision
**Priority**: CRITICAL - blocks all training progress until addressed
