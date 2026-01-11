# Fix for Premature Convergence (Step ~120 Collapse)

## Changes Implemented

### 1. ✅ Increased Temperature (Line 904)
```python
# Before:
temperature=1.1

# After:
temperature=1.4
```

**Why:** Forces more randomness in token selection, preventing all 4 generations from producing identical outputs even when they've learned a "good enough" strategy.

### 2. ✅ Reduced Learning Rate (Line 905)
```python
# Before:
learning_rate=2e-5

# After:
learning_rate=1e-5
```

**Why:** Slows convergence rate, giving the model more steps to explore alternatives before locking into a local optimum.

### 3. ✅ Different Game Seeds Per Generation (Lines 564, 626, 708)
```python
# Before:
seed = np.random.randint(10000)  # All 4 generations use SAME seed

# After:
seeds = [np.random.randint(10000) for _ in range(len(completions))]
# Each generation gets DIFFERENT seed
```

**Why:** Even when the model outputs identical code, different board states produce different rewards, maintaining variance for GRPO.

## Problem Summary

- **Collapse occurred at:** Step ~120 (epoch 0.12, ~12% of training)
- **Symptom:** `reward_std = 0.0` and `frac_reward_zero_std = 1.0` from step 120 to 1000
- **Cause:** All 4 generations produced identical (or near-identical) strategies
- **Impact:** GRPO had zero variance to compute advantages → no learning for 880 steps

### The Collapsed Strategy

```python
def strategy(board):
    for i in range(4):
        for j in range(3):
            if board[i][j] and board[i][j] == board[i][j+1]:
                return "W" if j % 2 else "S"
            if board[j][i] and board[j][i] == board[j+1][i]:
                return "A" if j % 2 else "D"
    # Returns None if no match → invalid move
```

**Reward breakdown:**
- `function_works`: +1.0 ✓
- `no_cheating`: +1.0 ✓
- `complexity_reward`: +5.0 ✓ (max score!)
- `strategy_succeeds`: -1.5 (invalid move)
- **Total: +5.5** (local optimum)

## How The Fixes Work Together

### Before Fixes:
1. Model finds a strategy that gets +5.5 reward
2. High temperature (1.1) + high LR (2e-5) → rapid convergence
3. All 4 generations produce identical code
4. Same game seed → identical rewards
5. `reward_std = 0.0` → GRPO stuck

### After Fixes:
1. Model finds a strategy that gets +5.5 reward
2. **Higher temperature (1.4)** → more diverse outputs even when converging
3. **Lower LR (1e-5)** → slower convergence, more time to explore
4. **Different seeds** → even identical code produces different rewards
5. `reward_std > 0.0` → GRPO can continue learning

## Expected Results

### Early Training (Steps 1-200)
- `reward_std`: 0.5 to 2.0 (healthy variance)
- `frac_reward_zero_std`: < 0.3
- More "garbage" outputs due to higher temp, but diversity maintained
- Some completions get negative rewards, others positive → variance

### Mid Training (Steps 200-600)
- `reward_std`: 0.3 to 1.0
- `frac_reward_zero_std`: < 0.5
- Gradual convergence toward better strategies
- Diversity preserved even as average reward increases

### Late Training (Steps 600-1000)
- `reward_std`: 0.2 to 0.8
- `frac_reward_zero_std`: < 0.6
- Fine-tuning without collapse
- Rewards continue improving slowly

## Monitoring During Training

Watch these metrics in W&B or logs:

### 🚨 RED FLAGS (indicates collapse):
- `frac_reward_zero_std` → 1.0 (all batches have zero variance)
- `reward_std` → 0.0 (no variance at all)
- `completions/mean_length` very consistent (±2 tokens)
- `reward` constant for 50+ steps

### ✅ HEALTHY SIGNS:
- `frac_reward_zero_std` < 0.5 (most batches have variance)
- `reward_std` > 0.3 (decent variance)
- `completions/mean_length` varies by 10+ tokens
- `reward` gradually increasing with fluctuations

### 📊 KEY METRICS TO TRACK:
```
rewards/strategy_succeeds/std  # Should be > 0.5
rewards/complexity_reward/std  # Should be > 0.0
reward_std                     # Should be > 0.3
frac_reward_zero_std          # Should be < 0.5
completion_length variance    # Should be high
```

## If Collapse Still Happens

If `frac_reward_zero_std` reaches 1.0 again:

### Option A: Increase Temperature Further
```python
temperature=1.6  # Even more diversity
```

### Option B: Add Diversity Reward (Not implemented yet)
See `TRAINING_POSTMORTEM.md` for implementation details. This would directly penalize identical outputs.

### Option C: Curriculum Learning
Start with high temperature (1.6), gradually reduce to 1.2 over training.

## Trade-offs of These Fixes

### Higher Temperature (1.4)
- ✅ Prevents premature convergence
- ✅ Maintains output diversity
- ⚠️ More "garbage" outputs in early training
- ⚠️ Slower convergence to optimal policy

### Lower Learning Rate (1e-5)
- ✅ More exploration time
- ✅ Less likely to overshoot good solutions
- ⚠️ May not reach optimal policy in 1000 steps
- ⚠️ Training takes longer to show improvement

### Different Seeds Per Generation
- ✅ Guarantees variance even with identical code
- ✅ No downside
- ⚠️ Less direct comparison between strategies (but still useful)

## Files Changed

- `gpt_2048_rl.py` line 904: `temperature=1.4` (was 1.1)
- `gpt_2048_rl.py` line 905: `learning_rate=1e-5` (was 2e-5)
- `gpt_2048_rl.py` line 564: Generate different seeds per completion
- `gpt_2048_rl.py` line 626: Use different seed in parallel execution
- `gpt_2048_rl.py` line 708: Use different seed in sequential execution
- `TRAINING_POSTMORTEM.md`: Full analysis of collapse (new file)

## Testing Checklist

Before starting new training run:

✅ Code compiles without errors
✅ Temperature increased to 1.4
✅ Learning rate reduced to 1e-5
✅ Different seeds per generation implemented
✅ Parallel execution uses different seeds
✅ Sequential execution uses different seeds

## Success Criteria

The fix is successful if:
1. Training runs for 500+ steps without collapse
2. `frac_reward_zero_std` stays below 0.6 throughout
3. `reward_std` stays above 0.2 after step 100
4. Average reward increases over time (even if slowly)
5. Model outputs show variety in structure and approaches

## References

- `TRAINING_POSTMORTEM.md`: Detailed analysis of the collapse
- `BUGFIX.md` Bug #17: Previous occurrence of deterministic collapse
- GRPO paper: Group Relative Policy Optimization requires variance
