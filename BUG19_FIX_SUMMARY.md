# Bug #19 Fix Summary: Degenerate Strategy Collapse

## Problem Recap

After 6+ hours of training, the model failed to learn 2048:
- **7,735 games** hit 500-step timeout with 0 score
- Strategies generated moves that don't change the board (frozen state)
- No progress beyond tile 4
- Reward stuck at +2.0 (function_works + no_cheating)

## Solutions Implemented

### 1. ✅ Enabled `complexity_reward` (Prevents "return W" Collapse)

**Change:** Uncommented `complexity_reward` in trainer (line 874)

**Why this is critical:** This creates a reward structure where trivial strategies are UNPROFITABLE even with early tile rewards.

### 2. ✅ Added Reward Shaping for Early Tiles (2-8)

**Changes:**
- Tile 2: +0.2 reward (very small)
- Tile 4: +0.5 reward (small)
- Tile 8+: Standard tile-based rewards (1.0, 2.0, 3.0...)

**Why these specific values:** They provide a gradient for learning but are SMALLER than the complexity penalty.

### 3. ✅ Implemented Frozen Board State Detection

**Changes in `_execute_strategy_worker` (lines 307-347):**
- Tracks board state before/after each move
- Counts consecutive non-changing moves
- After 10 frozen moves → returns "frozen" state
- Penalized with -2.0 reward

**Impact:** Strategies that output invalid moves (don't change board) are detected and penalized within 10 steps instead of waiting 500 steps.

### 4. ✅ Rebalanced Penalties (Less Harsh, Encourages Exploration)

**Before → After:**
- Invalid moves: -3.0 → -1.5
- Test timeout: -3.0 → -2.0
- Test crashes: -2.0 → -1.5
- Game crashes: -3.0 → -1.5
- Game timeout: -1.0 (unchanged)
- **NEW** Frozen state: -2.0

**Why:** Harsh penalties (-3.0) discouraged exploration. The model preferred safe +2.0 over risky attempts that might crash. Less harsh penalties make exploration more attractive.

---

## The Math: Why This Prevents Local Minima

This is the KEY insight that prevents "return W" collapse:

### Trivial Strategy (return "W"):
```
function_works:      +1.0  ✓ Valid syntax
no_cheating:         +1.0  ✓ No forbidden imports
complexity_reward:   -4.0  ❌ Short code + single direction
strategy_succeeds:   +0.5  (reaches tile 4)
───────────────────────────
TOTAL:               -1.5  ❌ UNPROFITABLE!
```

### Simple If/Else (2 directions):
```
function_works:      +1.0  ✓
no_cheating:         +1.0  ✓
complexity_reward:   +1.0  ✓ Has conditionals + 2 moves
strategy_succeeds:   +1.0  (reaches tile 8)
───────────────────────────
TOTAL:               +4.0  ✓ Profitable, encourages exploration
```

### Complex Strategy (3+ directions):
```
function_works:      +1.0  ✓
no_cheating:         +1.0  ✓
complexity_reward:   +5.0  ✓ Complex logic + multiple moves
strategy_succeeds:   +4.0  (reaches tile 64)
───────────────────────────
TOTAL:              +11.0  ✓✓ Best reward!
```

## Complexity Reward Breakdown

The `complexity_reward` function (lines 730-808) scores based on:

1. **Length Penalty/Bonus:**
   - < 30 chars: -3.0 (very short like "return W")
   - 30-80 chars: -1.0 (short but has some logic)
   - 80-300 chars: +1.0 (good length)
   - > 300 chars: +0.5 (avoid rewarding verbosity)

2. **Complexity Bonus:**
   - Has `if`: +1.0
   - Has `for`: +1.0
   - Has `while` (not infinite): +1.0
   - Max: +2.0 for multiple control structures

3. **Diversity Bonus:**
   - Uses 1 direction: -1.0 (trivial)
   - Uses 2 directions: +1.0
   - Uses 3-4 directions: +2.0

**Example scores:**
- `return "W"`: -3 (length) -1 (single dir) = **-4.0**
- Simple if/else: -1 (short) +1 (if) +1 (2 dirs) = **+1.0**
- Complex multi-dir: +1 (length) +2 (complexity) +2 (4 dirs) = **+5.0**

## Why Early Tile Rewards Don't Create Local Minima

**Key principle:** Early tile rewards (+0.2, +0.5) are SMALLER than complexity penalty (-4.0).

**With complexity_reward enabled:**
- "return W" + tile 4: +1 +1 -4 +0.5 = **-1.5** (unprofitable)
- Smart strategy + tile 8: +1 +1 +1 +1.0 = **+4.0** (profitable)

**The gradient:**
- No progress (tile 0): -2.0
- Frozen state: -2.0
- Trivial + tile 4: -1.5
- Smart + tile 8: +4.0
- Smart + tile 16: +5.0
- Complex + tile 64: +11.0

The model learns: "I need to write COMPLEX code that ACTUALLY PLAYS to get positive rewards."

---

## Expected Training Behavior

### Early Training (Steps 1-100):
- Model explores different strategies
- Some hit frozen state penalty (-2.0)
- Some get small rewards for tiles 2-4
- Reward variance increases (good for GRPO)

### Mid Training (Steps 100-500):
- Model learns to avoid frozen states
- Starts writing conditional logic (complexity bonus)
- Reaches tiles 8-16 consistently
- Rewards climb to +3 to +5 range

### Late Training (Steps 500-1000):
- Model writes multi-directional strategies
- Reaches tiles 32-128
- Rewards climb to +7 to +11 range
- Occasional tile 256+ (reward +8 to +14)

## Monitoring

Watch these metrics during training:

1. **Frozen state frequency**: Should decrease over time
2. **Max tile progression**: Should see 8 → 16 → 32 → 64 progression
3. **Reward variance**: Should stay positive (not collapse to 0.0)
4. **Complexity scores**: Should increase (more complex strategies)
5. **Average reward**: Should climb from -2 to +5+ range

## Files Changed

- `gpt_2048_rl.py` line 874: Enabled `complexity_reward`
- `gpt_2048_rl.py` lines 307-347: Added frozen state detection
- `gpt_2048_rl.py` lines 434-451: Added early tile rewards (0.2, 0.5)
- `gpt_2048_rl.py` lines 409-413: Rebalanced penalties (-1.5, -2.0 instead of -3.0)
- `gpt_2048_rl.py` lines 419-425: Added frozen state penalty (-2.0)

## Testing Checklist

Before running training:

1. ✅ Code compiles without errors
2. ✅ Complexity reward is enabled
3. ✅ Early tile rewards are small (< 1.0)
4. ✅ Frozen state detection is active
5. ✅ Penalties are rebalanced

## Next Steps

Run training and monitor:
- First 10 steps: Should see variety of rewards (not just +2.0)
- First 50 steps: Should see some tile 8+ achievements
- First 200 steps: Should see tile 16-32 regularly
- Step 1000: Should see tile 64-128 regularly, possibly 256+

If the model still collapses:
- Check `frac_reward_zero_std` in logs (should be < 0.5)
- Check `complexity_reward/mean` (should be positive)
- Check for "frozen state" messages (should decrease over time)
