# Robust Timeout Mechanism for Strategy Execution

## Problem

The original `@execute_with_time_limit(2)` decorator from `unsloth` used signal-based timeouts (`signal.alarm()`), which have several limitations:

1. **Signals can be blocked or delayed** by certain system calls
2. **Pure Python infinite loops** may not check for signals frequently enough, causing the timeout to fail
3. **Not cross-platform** - doesn't work on Windows
4. **Not thread-safe** - signals only work in the main thread

This meant that generated strategy code with infinite loops could freeze the training process indefinitely.

## Solution

Replaced the signal-based timeout with a **multiprocessing-based timeout** that:

1. Runs strategy execution in a separate process
2. Forcefully terminates the process if it exceeds the timeout
3. Uses `process.terminate()` and `process.kill()` for guaranteed termination
4. Uses **'fork' on Linux** (fast, inherits memory) and **'spawn' elsewhere** (safer but slower)

## Key Changes

### 1. Import Changes (gpt_2048_rl.py:9-24)

```python
import sys
import multiprocessing as mp
from multiprocessing import Process, Queue
import signal

# Removed: execute_with_time_limit from unsloth import
```

### 2. New Timeout Implementation (gpt_2048_rl.py:261-321)

```python
def _execute_strategy_worker(strategy: Callable, game: GameBoard, result_queue: Queue):
    """Worker function that runs in a separate process"""
    # Runs the strategy with error handling

def execute_strategy(strategy: Callable, game: GameBoard, timeout: float = 2.0):
    """Execute strategy with a hard timeout using multiprocessing"""
    # Creates process, waits with timeout, forcefully terminates if needed
```

### 3. Multiprocessing Setup (gpt_2048_rl.py:816-826)

```python
if __name__ == "__main__":
    # Use 'fork' on Linux (faster) and 'spawn' elsewhere
    try:
        if sys.platform.startswith('linux'):
            mp.set_start_method('fork')
        else:
            mp.set_start_method('spawn')
    except RuntimeError:
        pass
```

## How It Works

1. **Process Creation**: Each strategy execution runs in a separate process
2. **Timeout Enforcement**: `process.join(timeout=2.0)` waits for completion
3. **Forceful Termination**: If process is still alive after timeout:
   - `process.terminate()` sends SIGTERM (graceful)
   - Wait 1 second
   - `process.kill()` sends SIGKILL (forceful) if still alive
4. **Result Retrieval**: Uses a `Queue` to pass results from worker to main process

## Why 'fork' vs 'spawn'?

- **'fork'** (Linux): Copies the parent process memory, so all modules are already loaded
  - Pros: Very fast (< 0.01s overhead)
  - Cons: Can have issues with CUDA contexts, not available on Windows

- **'spawn'** (Windows/macOS): Starts a fresh Python process
  - Pros: Clean slate, no CUDA issues, cross-platform
  - Cons: Must re-import all modules (~2-3s overhead with torch/unsloth)

For this project, 'fork' is safe because:
- Strategy execution doesn't use CUDA directly
- It only manipulates board state (pure Python)
- Runs on Linux/DGX systems

## Testing

Run the test suite:

```bash
source .venv/bin/activate
python test_timeout.py
```

Tests verify:
1. ✓ Normal strategies execute successfully
2. ✓ Infinite loops are caught and terminated
3. ✓ Slow computations timeout correctly
4. ✓ Nested loops timeout correctly
5. ✓ Exceptions are properly propagated
6. ✓ Fast strategies complete within timeout

## Performance Impact

- **Fork method (Linux)**: ~0.01s overhead per execution → negligible
- **Spawn method (Others)**: ~2-3s overhead → only use if fork unavailable
- **Timeout checking**: Sub-millisecond precision

## Limitations

1. **CUDA contexts**: If strategy execution required CUDA operations, 'fork' could cause issues. Consider 'spawn' or 'forkserver' in that case.
2. **Shared memory**: Child processes don't share memory with parent after fork (copy-on-write)
3. **Queue size**: Very large results could slow queue operations (not an issue for this use case)

## Alternatives Considered

1. **Threading + timeout**: Python threads can't be forcefully killed, infinite loops would still freeze
2. **signal.alarm()**: Original approach, unreliable with pure Python loops
3. **asyncio with timeout**: Requires cooperative cancellation, doesn't work with blocking code
4. **Process pool**: Overhead of maintaining pool, overkill for this use case

## Migration Notes

If you need to change the timeout value:

```python
# In reward functions or testing code:
steps, result = execute_strategy(strategy, game, timeout=5.0)  # 5 seconds
```

No other code changes needed - the function signature is backward compatible.
