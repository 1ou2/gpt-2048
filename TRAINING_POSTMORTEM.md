# Training Post-Mortem: Premature Convergence at Step ~120

## Summary

Training collapsed at step ~120 (epoch 0.12, ~12% through training). The model converged to outputting identical strategies across all 4 generations, causing `reward_std = 0.0` and preventing GRPO from learning further.

## Timeline

### Early Training (Steps 1-115): Healthy Exploration
- `reward_std`: 0.25 to 12.59 (high variance ✓)
- `frac_reward_zero_std`: 0.0 (variance present ✓)
- Diverse strategies with different approaches
- Many frozen states (-2.0) but also some successes

### The Collapse (Step ~120, Epoch 0.12)
- `reward_std`: drops to **0.0** and STAYS there
- `frac_reward_zero_std`: jumps to **1.0** (100% zero variance)
- All 4 generations produce identical code
- Reward locks at **5.5** for remaining 880 steps

### Post-Collapse (Steps 120-1000): Zero Learning
- Model outputs same strategy verbatim for 880 steps
- `completions/mean_length`: 93-98 tokens (very consistent)
- No exploration, no improvement, no learning

## The Collapsed Strategy

```python
def strategy(board):
    for i in range(4):
        for j in range(3):
            if board[i][j] and board[i][j] == board[i][j+1]:
                return "W" if j % 2 else "S"
            if board[j][i] and board[j][i] == board[j+1][i]:
                return "A" if j % 2 else "D"
    # Implicit: returns None if no match found
```

### Why This Strategy is "Good Enough"

**Reward breakdown:**
- `function_works`: +1.0 (valid syntax)
- `no_cheating`: +1.0 (no forbidden imports)
- `complexity_reward`: +5.0 (has loops, conditionals, 4 directions - **MAX SCORE**)
- `strategy_succeeds`: -1.5 (returns None → invalid move)
- **Total: +5.5**

This is a **local optimum**. The strategy:
1. ✓ Passes all complexity checks
2. ✓ Has loops and conditionals
3. ✓ Uses all 4 directions
4. ✓ More complex than "return W"
5. ✗ Returns `None` when no adjacent tiles match
6. ✗ Gets invalid move penalty
7. ✗ Never actually plays the game successfully

**But it scores higher than most exploratory strategies** that might crash (-1.5 to -2.0) or timeout (-1.0).

## Root Cause: GRPO Requires Reward Variance

GRPO (Group Relative Policy Optimization) works by comparing rewards WITHIN a batch:
1. Generate 4 completions per prompt
2. Compute rewards for each
3. Calculate advantages: `advantage[i] = reward[i] - mean(rewards)`
4. Update policy based on advantages

**When all 4 generations are identical:**
- `reward[0] = reward[1] = reward[2] = reward[3] = 5.5`
- `mean(rewards) = 5.5`
- `advantage[i] = 5.5 - 5.5 = 0` for all i
- **No gradient, no learning!**

The metric `frac_reward_zero_std: 1.0` means "100% of batches have zero variance" → GRPO is completely stuck.

## Why Did This Happen?

### 1. Model Learned a Deterministic Mapping

The model discovered:
- Input: (any 2048 board state)
- Output: (check adjacent tiles, return direction)
- Result: +5.5 reward

Since all 4 generations see the **same prompt** and evaluate on the **same game seed**, a deterministic policy produces identical outputs.

### 2. Temperature Not High Enough

Current setting: `temperature=1.1`

This encourages SOME exploration but not enough to prevent convergence once a "good enough" strategy is found. The model found a strategy that:
- Reliably gets +5.5
- Never crashes (safe)
- Passes complexity checks
- Is "good enough" compared to risky exploration

### 3. Learning Rate Too High for Stable Convergence

Current: `learning_rate=2e-5`

At step 120, the model had strong gradients favoring this strategy pattern. High LR caused rapid convergence before exploring better alternatives.

### 4. The Complexity Reward Has a Ceiling

`complexity_reward` maxes out at +5.0, which this strategy achieves. There's no incentive to explore BEYOND this pattern once found.

## This is Bug #17 From BUGFIX.md

This is the exact issue documented as **"Deterministic Strategy Collapse Despite Complexity Reward"** in BUGFIX.md (lines 586-648):

> Despite the complexity_reward preventing trivial `return "A"` strategies, the model still collapsed to identical outputs across all generations, resulting in zero variance.

The previous training run (Bug #17) collapsed to:
```python
def strategy(board):
    count = sum(row.count(2) for row in board)
    if count == 6: return 'W'
    elif count == 4: return 'A'
    elif count == 2: return 'S'
    return 'D'
```

This training run collapsed to a different but equally stuck pattern.

---

## Remediations

### Solution 1: Increase Temperature (HIGH PRIORITY)

**Current:** `temperature=1.1`
**Recommended:** `temperature=1.4`

```python
# Line 845 in gpt_2048_rl.py
training_args = GRPOConfig(
    temperature=1.4,  # Was 1.1 - force MORE diversity
    # ... rest unchanged
)
```

**Why this helps:**
- Higher temperature = more randomness in token selection
- Even with same input, outputs will vary more
- Prevents premature convergence to single pattern
- Bug #15 showed 1.3 caused garbage, but that was WITHOUT the bug fixes
- With current complexity+frozen checks, 1.4 should be safe

**Trade-off:** May produce lower quality initially, but prevents collapse

### Solution 2: Different Game Seeds Per Generation (MEDIUM PRIORITY)

Currently all 4 generations evaluate on the SAME game seed. Change to:

```python
# In strategy_succeeds() around line 522
# OLD: seed = np.random.randint(10000)  # Single seed for all

# NEW: Different seed per completion
seeds = [np.random.randint(10000) for _ in range(len(completions))]

# Then in parallel execution (line 582):
game_args.append((function, seeds[idx], 4, 2048, PRINTER % 5 == 1))
```

**Why this helps:**
- Even deterministic strategies produce different results on different boards
- Creates variance even when code is identical
- GRPO can compare "strategy A on board X" vs "strategy A on board Y"

**Trade-off:** Less direct comparison of strategies, but prevents zero variance

### Solution 3: Explicit Diversity Reward (LOW PRIORITY)

Add a 5th reward function that penalizes identical outputs:

```python
def diversity_reward(completions, **kwargs):
    """Penalize when all generations are too similar"""
    functions = [extract_function(c[0]["content"]) for c in completions]
    functions = [f for f in functions if f is not None]

    if len(functions) < 2:
        return [0.0] * len(completions)

    # Check if all functions are identical (or very similar)
    unique_funcs = set(functions)
    if len(unique_funcs) == 1:
        # All identical - penalize everyone
        return [-2.0] * len(completions)
    elif len(unique_funcs) == 2:
        # Low diversity - mild penalty
        return [-0.5] * len(completions)
    else:
        # Good diversity - small bonus
        return [+0.5] * len(completions)
```

**Why this helps:**
- Directly penalizes the problematic behavior
- Forces model to maintain variety

**Trade-off:** Can be noisy, might prevent convergence to good solutions

### Solution 4: Lower Learning Rate (MEDIUM PRIORITY)

**Current:** `learning_rate=2e-5`
**Recommended:** `learning_rate=1e-5`

```python
# Line 846 in gpt_2048_rl.py
training_args = GRPOConfig(
    learning_rate=1e-5,  # Was 2e-5 - slower convergence
    # ... rest unchanged
)
```

**Why this helps:**
- Slower convergence gives more time to explore
- Prevents rapid lock-in to local optima
- Model has 1000 steps, so can afford slower learning

**Trade-off:** May not reach optimal policy in 1000 steps

### Solution 5: Curriculum Learning - Increase Frozen Threshold Over Time (LOW PRIORITY)

Currently frozen threshold is fixed at 10 moves. Make it stricter over training:

```python
# In _execute_strategy_worker
# OLD: FROZEN_THRESHOLD = 10

# NEW: Start lenient, get stricter
current_step = kwargs.get('step', 0)  # Would need to pass this
FROZEN_THRESHOLD = max(5, 20 - (current_step // 100))  # 20 → 15 → 10 → 5
```

**Why this helps:**
- Early training: allow "somewhat stuck" strategies to get partial credit
- Late training: require truly dynamic strategies

**Trade-off:** Requires passing step number through call chain

---

## Recommended Action Plan

**Priority 1 - Minimum Viable Fix:**
1. ✅ Increase temperature to 1.4
2. ✅ Lower learning rate to 1e-5

**Priority 2 - If Still Collapses:**
3. ✅ Implement different seeds per generation

**Priority 3 - If Desperate:**
4. ✅ Add diversity reward function

## Expected Results

With temperature=1.4 and LR=1e-5:
- First 200 steps: More varied outputs, some garbage, but variance maintained
- Steps 200-600: Convergence toward better strategies, but diversity preserved
- Steps 600-1000: Fine-tuning without premature collapse

Watch for:
- `frac_reward_zero_std` should stay < 0.5 (ideally < 0.2)
- `reward_std` should stay > 0.5
- `completion_length` variance should stay high
- Reward should gradually increase (not jump and flatline)

## Conclusion

The model didn't fail because the strategy was bad - it succeeded too well at finding a "good enough" local optimum that satisfies all constraints. The issue is that GRPO needs VARIANCE to learn, and once the model converges to outputting identical code, learning stops completely.

The fixes target the convergence speed (lower LR) and output diversity (higher temp + different seeds) to keep the model exploring longer.
