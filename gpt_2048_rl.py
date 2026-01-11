"""
GPT-OSS Reinforcement Learning for 2048 Game
Goal: Make GPT-OSS play games with Reinforcement Learning (GRPO)
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import random
import copy
import math
import numpy as np
from unsloth import FastLanguageModel, check_python_modules, create_locked_down_function
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import TextStreamer
import multiprocessing as mp
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor
import signal

# Load environment variables from .env file
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

load_env()

# Set up Weights & Biases
if "WANDB" in os.environ:
    os.environ["WANDB_API_KEY"] = os.environ["WANDB"]
    os.environ["WANDB_PROJECT"] = "gpt-2048-rl"

# ============================================================================
# CONFIGURATION
# ============================================================================

# Enable/disable parallel reward computation (uses ProcessPoolExecutor)
# When enabled, game evaluations run in parallel (faster but uses more CPU cores)
# When disabled, games run sequentially (slower but easier to debug)
ENABLE_PARALLEL_REWARDS = True
PARALLEL_WORKERS = 8  # Number of parallel workers for reward computation


# ============================================================================
# 2048 GAME IMPLEMENTATION
# ============================================================================

def _compress_and_merge_row_left(row: List[int]) -> Tuple[List[int], int, bool]:
    n = len(row)
    tiles = [x for x in row if x != 0]
    gained = 0
    i = 0
    merged = []
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            v = tiles[i] * 2
            gained += v
            merged.append(v)
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged += [0] * (n - len(merged))
    changed = merged != row
    return merged, gained, changed

def _move_left(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
    changed_any = False
    total_gain = 0
    new_board = []
    for row in board:
        new_row, gained, changed = _compress_and_merge_row_left(row)
        new_board.append(new_row)
        total_gain += gained
        changed_any = changed_any or changed
    return new_board, total_gain, changed_any

def _move_right(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
    changed_any = False
    total_gain = 0
    new_board = []
    for row in board:
        rev = list(reversed(row))
        new_rev, gained, changed = _compress_and_merge_row_left(rev)
        new_row = list(reversed(new_rev))
        new_board.append(new_row)
        total_gain += gained
        changed_any = changed_any or changed
    return new_board, total_gain, changed_any

def _transpose(board: List[List[int]]) -> List[List[int]]:
    return [list(row) for row in zip(*board)]

def _move_up(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
    t = _transpose(board)
    moved, gain, changed = _move_left(t)
    return _transpose(moved), gain, changed

def _move_down(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
    t = _transpose(board)
    moved, gain, changed = _move_right(t)
    return _transpose(moved), gain, changed

def _empty_cells(board: List[List[int]]) -> List[Tuple[int, int]]:
    size = len(board)
    return [(r, c) for r in range(size) for c in range(size) if board[r][c] == 0]

def _can_move(board: List[List[int]]) -> bool:
    if _empty_cells(board):
        return True
    size = len(board)
    for r in range(size):
        for c in range(size - 1):
            if board[r][c] == board[r][c + 1]:
                return True
    for r in range(size - 1):
        for c in range(size):
            if board[r][c] == board[r + 1][c]:
                return True
    return False

@dataclass
class GameBoard:
    size: int
    seed: Optional[int] = None
    target: int = 2048
    probability_fours: float = 0.10  # originally spawns (4) 10% of the time!
    _rng: random.Random = field(init=False, repr=False)
    _board: List[List[int]] = field(init=False, repr=False)
    _score: int = field(default=0, init=False, repr=False)
    _state: str = field(default="ongoing", init=False, repr=False)

    def __post_init__(self):
        if self.size < 2:
            raise ValueError("Board size must be at least 2.")
        self._rng = random.Random(self.seed)
        self._board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self._add_random_tile()
        self._add_random_tile()
        self._update_state_after_change()

    class _BoardView:
        def __init__(self, game: "GameBoard"):
            self._game = game
        def __iter__(self):
            return iter(self._game._board)
        def __len__(self):
            return len(self._game._board)
        def __getitem__(self, idx):
            return self._game._board[idx]
        def __repr__(self) -> str:
            return repr(self._game._board)
        __str__ = __repr__
        def do_action(self, key: str) -> None:
            self._game.do_action(key)
        def state(self) -> str:
            return self._game.state()
        def pretty(self, colors: bool = True, border: bool = True, dot_for_zero: bool = True) -> str:
            return self._game._render_pretty(colors=colors, border=border, dot_for_zero=dot_for_zero)

    def board(self) -> "_BoardView":
        return GameBoard._BoardView(self)
    def state(self) -> str:
        return self._state
    def score(self) -> int:
        return self._score
    def do_action(self, key: str) -> None:
        if self._state != "ongoing":
            return
        if not isinstance(key, str) or len(key) == 0:
            self._state = "failed"
            return
        k = key.strip().lower()
        if k == "q":
            self._state = "failed"
            return
        move_map = {"w": _move_up, "a": _move_left, "s": _move_down, "d": _move_right}
        if k not in move_map:
            self._state = "failed"
            return
        mover = move_map[k]
        new_board, gain, changed = mover(self._board)
        if changed:
            self._board = new_board
            self._score += gain
            self._add_random_tile()
        self._update_state_after_change()
    def _add_random_tile(self) -> bool:
        empties = _empty_cells(self._board)
        if not empties:
            return False
        r, c = self._rng.choice(empties)
        self._board[r][c] = 4 if self._rng.random() < self.probability_fours else 2
        return True
    def _update_state_after_change(self) -> None:
        if any(self.target in row for row in self._board):
            self._state = "success"
            return
        if not _can_move(self._board):
            self._state = "failed"
            return
        self._state = "ongoing"
    def _render_pretty(self, colors: bool = True, border: bool = True, dot_for_zero: bool = True) -> str:
        """
        Pretty-print the board with colors that scale from 0 up to self.target.
        Uses ANSI 256-color codes (works in most terminals). Set colors=False to disable.
        """
        import math

        b = self._board
        mx = max((max(row) for row in b), default=0)
        cell_w = max(3, len(str(mx)))

        RESET = "\x1b[0m"

        # A smooth-ish gradient from cool → warm
        # (blue/cyan/green → yellow/orange/red). Tweak or expand as you like.
        GRAD = [33, 39, 45, 51, 50, 49, 48, 47, 46, 82, 118, 154, 190, 226, 220, 214, 208, 202, 196]
        ZERO_FG = 239  # dim gray

        def color_code(v: int) -> str:
            if not colors:
                return ""
            if v == 0:
                return f"\x1b[38;5;{ZERO_FG}m"
            # Normalize by exponent relative to target: r in [0,1]
            t = max(2, self.target)  # safety; avoid log2(1)
            # Guard: if v is not a power of two or is <1, handle gracefully
            try:
                r = max(0.0, min(1.0, math.log2(v) / math.log2(t)))
            except ValueError:
                r = 0.0
            idx = int(round(r * (len(GRAD) - 1)))
            return f"\x1b[38;5;{GRAD[idx]}m"

        def fmt(v: int) -> str:
            s = "." if (v == 0 and dot_for_zero) else str(v)
            s = s.rjust(cell_w)
            return color_code(v) + s + (RESET if colors else "")

        def hline(left: str, mid: str, right: str) -> str:
            return left + mid.join("─" * cell_w for _ in range(self.size)) + right

        rows = []
        if border:
            rows.append(hline("┌", "┬", "┐"))
        for r in range(self.size):
            content = "│".join(fmt(v) for v in b[r])
            rows.append(("│" + content + "│") if border else content)
            if border:
                rows.append(hline("└" if r == self.size - 1 else "├",
                                "┴" if r == self.size - 1 else "┼",
                                "┘" if r == self.size - 1 else "┤"))
        return "\n".join(rows)


# ============================================================================
# RL ENVIRONMENT SETUP
# ============================================================================

def _call_with_timeout_worker(func, args, result_queue: Queue):
    """Worker function for single function calls with timeout"""
    try:
        result = func(*args)
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", str(e)))

def call_with_timeout(func, args, timeout: float = 1.0):
    """
    Call a function with a hard timeout using multiprocessing.
    Used to test strategy functions before running full game.
    """
    result_queue = Queue()
    process = Process(target=_call_with_timeout_worker, args=(func, args, result_queue))
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
        if process.is_alive():
            process.kill()
            process.join()
        raise TimeoutError(f"Function call exceeded {timeout}s timeout")

    if not result_queue.empty():
        status, result = result_queue.get()
        if status == "error":
            raise RuntimeError(result)
        return result

    raise RuntimeError("Function call failed without returning a result")

def _execute_strategy_worker(strategy: Callable, game: GameBoard, result_queue: Queue):
    """Worker function that runs in a separate process"""
    try:
        assert callable(strategy)

        MAX_STEPS = 500  # Hard limit to prevent infinite loops
        FROZEN_THRESHOLD = 10  # Number of consecutive non-changing moves before declaring frozen

        steps = 0
        consecutive_frozen = 0
        last_board_state = None

        while game.state() == "ongoing":
            if steps >= MAX_STEPS:
                result_queue.put((steps, "failed"))
                return

            # Capture board state before move
            current_board_state = tuple(tuple(row) for row in game.board())

            action = strategy(list(game.board()))
            steps += 1
            if type(action) is not str:
                result_queue.put((steps, "failed"))
                return
            game.do_action(action)

            # Check if board changed after move
            new_board_state = tuple(tuple(row) for row in game.board())
            if current_board_state == new_board_state:
                consecutive_frozen += 1
                if consecutive_frozen >= FROZEN_THRESHOLD:
                    # Board frozen - strategy is stuck
                    result_queue.put((steps, "frozen"))
                    return
            else:
                consecutive_frozen = 0

        result_queue.put((steps, game.state()))
    except Exception as e:
        result_queue.put(("error", str(e)))

def execute_strategy(strategy: Callable, game: GameBoard, timeout: float = 1.0):
    """
    Execute strategy with a hard timeout using multiprocessing.
    This is more robust than signal-based timeouts and will forcefully
    terminate infinite loops or frozen code.
    Reduced from 2.0s to 1.0s to cut wasted time on bad strategies.
    """
    # Create a queue to get results from the worker process
    result_queue = Queue()

    # Create and start the worker process
    process = Process(target=_execute_strategy_worker, args=(strategy, game, result_queue))
    process.start()

    # Wait for the process to complete or timeout
    process.join(timeout=timeout)

    # Check if process is still alive (timed out)
    if process.is_alive():
        # Forcefully terminate the process
        process.terminate()
        process.join(timeout=1.0)  # Give it 1 second to terminate gracefully

        # If still alive, kill it forcefully
        if process.is_alive():
            process.kill()
            process.join()

        # Return timeout error
        raise TimeoutError(f"Strategy execution exceeded {timeout}s timeout")

    # Get the result from the queue
    if not result_queue.empty():
        result = result_queue.get()
        if isinstance(result, tuple) and len(result) == 2:
            if result[0] == "error":
                raise RuntimeError(f"Strategy execution failed: {result[1]}")
            return result

    # If we got here, something went wrong
    raise RuntimeError("Strategy execution failed without returning a result")


def _run_game_worker(args):
    """
    Worker function for parallel game evaluation.
    Takes (function_code, seed, board_size, target, print_debug) tuple.
    Returns (score, debug_info_dict).
    """
    function_code, seed, board_size, target, print_debug = args

    try:
        # Create the strategy function
        strategy = create_locked_down_function(function_code)

        # Test with empty board first (with timeout)
        test_board = [[0]*board_size for _ in range(board_size)]
        try:
            test_move = call_with_timeout(strategy, (test_board,), timeout=1.0)
            if not isinstance(test_move, str) or test_move not in ["W", "A", "S", "D"]:
                return (-1.5, {"error": "invalid_move", "move": str(test_move)})  # Less harsh
        except TimeoutError:
            return (-2.0, {"error": "test_timeout"})  # Still bad but not -3
        except Exception as e:
            return (-1.5, {"error": "test_crash", "exception": str(e)})  # Less harsh

        # Run the actual game
        game = GameBoard(size=board_size, seed=seed, target=target, probability_fours=0.10)
        steps, game_state = execute_strategy(strategy, game)

        # Penalize frozen board states (stuck strategies)
        if game_state == "frozen":
            return (-2.0, {
                "error": "frozen_state",
                "steps": steps,
                "max_tile": max(max(row) for row in game.board()),
            })

        # Calculate reward based on max tile
        if game_state == "success":
            score = 20.0
            reward_info = "success"
        else:
            max_tile = max(max(row) for row in game.board())

            # Small rewards for early tiles (2-8) to provide gradient
            # BUT kept small enough that complexity penalty dominates
            # This prevents "return W" from being profitable
            if max_tile == 2:
                score = 0.2  # Very small reward
                reward_info = "tile_2"
            elif max_tile == 4:
                score = 0.5  # Small reward
                reward_info = "tile_4"
            elif max_tile >= 8:
                # Standard tile-based rewards for tiles 8+
                # 8->1, 16->2, 32->3, 64->4, 128->5, 256->6, 512->7
                level = int(math.log2(max_tile)) - 2
                score = float(level)
                reward_info = f"tile_{max_tile}"
            else:
                score = 0.0
                reward_info = f"tile_0"

        return (score, {
            "steps": steps,
            "state": game_state,
            "game_score": game.score(),
            "max_tile": max(max(row) for row in game.board()),
            "reward_info": reward_info,
            "board": [list(row) for row in game.board()]
        })

    except TimeoutError:
        return (-1.0, {"error": "game_timeout"})
    except Exception as e:
        return (-1.5, {"error": "game_crash", "exception": str(e)})  # Rebalanced: less harsh


# ============================================================================
# CODE EXECUTION AND SAFETY
# ============================================================================

def extract_function(text):
    # Try to extract from backticks first
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first : second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def strategy(board):"): return fx

    # Fallback: find "def strategy(board):" anywhere in text
    if "def strategy(board):" in text:
        start = text.find("def strategy(board):")
        # Find the end of the function (next def, class, or significant dedent)
        lines = text[start:].split("\n")
        func_lines = [lines[0]]
        for line in lines[1:]:
            # Stop at empty line followed by non-indented content, or at next def/class
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                if line.startswith("def ") or line.startswith("class ") or line.startswith("```"):
                    break
            func_lines.append(line)
        fx = "\n".join(func_lines).rstrip()
        if fx.startswith("def strategy(board):"): return fx

    return None


# ============================================================================
# REWARD FUNCTIONS
# ============================================================================

def function_works(completions, **kwargs):
    scores = []
    dangerous_calls = ["input(", "open(", "exec(", "eval(", "__import__"]
    
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)

        # If function extraction failed, return -2.0
        if function is None:
            scores.append(-2.0)
            continue

        # Block dangerous functions that can hang
        if any(call in function for call in dangerous_calls):
            scores.append(-2.0)
            continue

        # Check if function uses forbidden modules
        ok, info = check_python_modules(function)
        if "error" in info:
            scores.append(-2.0)
            continue

        # Try to create the function
        try:
            new_strategy = create_locked_down_function(function)
            scores.append(1.0)
        except:
            scores.append(-0.5)

    return scores

def no_cheating(completions, **kwargs):
    scores = []
    dangerous_calls = ["input(", "open(", "exec(", "eval(", "__import__"]
    
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            # Block dangerous functions
            if any(call in function for call in dangerous_calls):
                scores.append(-20.0)
                continue
            
            ok, info = check_python_modules(function)
            scores.append(1.0 if ok else -20.0)  # Penalize heavily!
        else:
            scores.append(-1.0)  # Failed creating function
    return scores

global PRINTER
PRINTER = 0

def strategy_succeeds(completions, **kwargs):
    global PRINTER
    scores = []
    # Generate DIFFERENT random seeds for each completion to ensure variance
    # Even deterministic strategies will produce different results on different boards
    seeds = [np.random.randint(10000) for _ in range(len(completions))]

    # Phase 1: Validate all functions (fast, sequential)
    validated_functions = []
    for idx, completion in enumerate(completions):
        response = completion[0]["content"]

        # Print full completion every 5 steps (for monitoring)
        if PRINTER % 5 == 0:
            print("\n" + "="*80)
            print(f"COMPLETION {PRINTER} (Generation {idx+1}/{len(completions)}):")
            print("="*80)
            print(response)
            print("="*80 + "\n")

        PRINTER += 1

        # Extract function
        function = extract_function(response)
        if function is None:
            if PRINTER % 5 == 1:
                print("❌ Failed to extract function from response")
            validated_functions.append((-1.0, None))  # (score, function_code)
            continue

        # Block dangerous functions that can hang (input, open, etc.)
        dangerous_calls = ["input(", "open(", "exec(", "eval(", "__import__"]
        if any(call in function for call in dangerous_calls):
            if PRINTER % 5 == 1:
                print(f"❌ Dangerous function call detected")
            validated_functions.append((-2.0, None))
            continue

        # Check for forbidden modules
        ok, info = check_python_modules(function)
        if "error" in info:
            if PRINTER % 5 == 1:
                print(f"❌ Forbidden modules detected: {info}")
            validated_functions.append((-2.0, None))
            continue

        # Try to create executable function (test it works)
        try:
            new_strategy = create_locked_down_function(function)
        except Exception as e:
            if PRINTER % 5 == 1:
                print(f"❌ Failed to create function: {str(e)}")
            validated_functions.append((-1.0, None))
            continue

        # Passed validation - store for game execution
        validated_functions.append((None, function))

    # Phase 2: Execute games (expensive, optionally parallel)
    if ENABLE_PARALLEL_REWARDS:
        # Parallel execution
        game_args = []
        func_idx = 0  # Track which function we're on for seed assignment
        for idx, (score_or_none, function) in enumerate(validated_functions):
            if function is not None:
                # (function_code, seed, board_size, target, print_debug)
                # Use different seed per completion to ensure variance
                game_args.append((function, seeds[idx], 4, 2048, PRINTER % 5 == 1))
                func_idx += 1

        if game_args:
            with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                game_results = list(executor.map(_run_game_worker, game_args))
        else:
            game_results = []

        # Merge results back
        result_idx = 0
        for score_or_none, function in validated_functions:
            if function is None:
                # Already failed validation
                scores.append(score_or_none)
            else:
                # Get result from parallel execution
                score, debug_info = game_results[result_idx]
                result_idx += 1

                # Print debug info
                if "error" not in debug_info:
                    print(f"Steps = {debug_info['steps']} | State = {debug_info['state']} | Score = {debug_info['game_score']}")
                    print(function)
                    # Print board (simplified)
                    max_tile = debug_info['max_tile']
                    if debug_info['state'] == "success":
                        print("🎉 SUCCESS! Reached 2048!")
                    elif max_tile >= 8:
                        print(f"🎯 Max tile: {max_tile} -> reward: {score}")
                else:
                    error_type = debug_info.get('error', 'unknown')
                    if error_type == "test_timeout":
                        print("⏱️  Test call timed out (infinite loop?)")
                    elif error_type == "game_timeout":
                        print("⏱️  Timeout (1s exceeded)")
                    elif error_type == "frozen_state":
                        print(f"❄️  Frozen state (board unchanged for 10+ moves)")
                        print(function)
                    elif error_type == "invalid_move":
                        print(f"❌ Invalid move: '{debug_info.get('move', '?')}' (must be W/A/S/D)")
                    else:
                        print(f"💥 Exception: {debug_info.get('exception', 'unknown')}")

                scores.append(score)

    else:
        # Sequential execution (original behavior)
        for idx, (score_or_none, function) in enumerate(validated_functions):
            if function is None:
                # Already failed validation
                scores.append(score_or_none)
                continue

            # TEST: Function must return valid move (W/A/S/D)
            try:
                new_strategy = create_locked_down_function(function)
                test_board = [[0]*4 for _ in range(4)]
                try:
                    test_move = call_with_timeout(new_strategy, (test_board,), timeout=1.0)
                    if not isinstance(test_move, str):
                        if PRINTER % 5 == 1:
                            print(f"❌ Invalid return type: {type(test_move)}")
                        scores.append(-1.5)  # Rebalanced: less harsh
                        continue
                    if test_move not in ["W", "A", "S", "D"]:
                        if PRINTER % 5 == 1:
                            print(f"❌ Invalid move: '{test_move}' (must be W/A/S/D)")
                        scores.append(-1.5)  # Rebalanced: less harsh
                        continue
                except TimeoutError:
                    if PRINTER % 5 == 1:
                        print(f"⏱️  Test call timed out (infinite loop?)")
                    scores.append(-2.0)  # Rebalanced: less harsh but still bad
                    continue
                except Exception as e:
                    if PRINTER % 5 == 1:
                        print(f"❌ Function crashed on test: {str(e)}")
                    scores.append(-1.5)  # Rebalanced: less harsh
                    continue

                # Execute strategy on game with DIFFERENT seed per completion
                game = GameBoard(size=4, seed=seeds[idx], target=2048, probability_fours=0.10)
                steps, game_state = execute_strategy(new_strategy, game)

                # Check for frozen state
                if game_state == "frozen":
                    print(f"❄️  Frozen state (board unchanged for 10+ moves)")
                    print(function)
                    scores.append(-2.0)
                    continue

                print(f"Steps = {steps} | State = {game_state} | Score = {game.score()}")
                print(function)
                print(game.board().pretty())

                if game_state == "success":
                    print("🎉 SUCCESS! Reached 2048!")
                    scores.append(20.0)
                else:
                    max_tile = max(max(row) for row in game.board())
                    # Small rewards for early tiles to provide gradient
                    if max_tile == 2:
                        scores.append(0.2)
                    elif max_tile == 4:
                        scores.append(0.5)
                    elif max_tile >= 8:
                        level = int(math.log2(max_tile)) - 2
                        tile_reward = float(level)
                        print(f"🎯 Max tile: {max_tile} -> reward: {tile_reward}")
                        scores.append(tile_reward)
                    else:
                        scores.append(0.0)
            except TimeoutError:
                print("⏱️  Timeout (1s exceeded)")
                scores.append(-1.0)  # Timeout is expected, mild penalty
            except Exception as e:
                print(f"💥 Exception: {str(e)}")
                scores.append(-1.5)  # Rebalanced: less harsh

    return scores


def complexity_reward(completions, **kwargs):
    """
    Reward function to prevent collapse to trivially simple strategies.

    This addresses the recurring issue where GRPO collapses to `return "A"` or similar
    single-line strategies. Without this, simple strategies achieve +4.0 reward
    (function_works + no_cheating + valid move) which is a local optimum.

    Rewards:
    - Length: Penalize < 50 tokens, bonus for 50-200 tokens
    - Complexity: Bonus for conditionals (if/for/while with logic)
    - Diversity: Bonus for using multiple move directions
    """
    scores = []

    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)

        if function is None:
            scores.append(0.0)  # Neutral - other functions handle this
            continue

        reward = 0.0

        # 1. LENGTH PENALTY/BONUS
        # `return "A"` is ~14 tokens, we want to discourage this
        func_len = len(function)
        if func_len < 30:  # Very short (e.g., `return "A"`)
            reward -= 3.0  # Strong penalty
        elif func_len < 80:  # Short but has some logic
            reward -= 1.0  # Mild penalty
        elif func_len < 300:  # Good length with logic
            reward += 1.0  # Bonus
        else:  # Long function
            reward += 0.5  # Smaller bonus (avoid rewarding verbosity)

        # 2. COMPLEXITY BONUS
        # Reward actual conditional logic (not just keywords in strings)
        has_if = "if " in function and ":" in function.split("if ", 1)[-1][:50]
        has_for = "for " in function and " in " in function
        has_while = "while " in function and "True" not in function  # Exclude infinite loops

        complexity_count = sum([has_if, has_for, has_while])
        if complexity_count >= 2:
            reward += 2.0  # Multiple control structures
        elif complexity_count == 1:
            reward += 1.0  # At least one control structure

        # 3. DIVERSITY BONUS
        # Reward strategies that can return different moves
        moves_present = sum([
            '"W"' in function or "'W'" in function,
            '"A"' in function or "'A'" in function,
            '"S"' in function or "'S'" in function,
            '"D"' in function or "'D'" in function,
        ])
        if moves_present >= 3:
            reward += 2.0  # Uses 3-4 different directions
        elif moves_present >= 2:
            reward += 1.0  # Uses 2 different directions
        elif moves_present == 1:
            reward -= 1.0  # Only one direction - likely trivial

        scores.append(reward)

    return scores


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load model with LoRA for RL training"""
    max_seq_length = 1024  # Room for prompt + complete strategy function
    lora_rank = 16  # Higher rank for smaller models

    # Model options (use Instruct versions for chat template support):
    # - "unsloth/Qwen2.5-0.5B-Instruct"  # Fastest: ~30-45 min total (great for learning!)
    # - "unsloth/Qwen2.5-1.5B-Instruct"  # Fast: ~1.5-2 hours total
    # - "unsloth/Qwen2.5-3B-Instruct"    # Medium: ~3-4 hours total
    # - "unsloth/gpt-oss-20b"            # Slow: ~25 hours total (reasoning model)
    model_name = "unsloth/Qwen2.5-3B-Instruct"  # Good balance for learning

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # Full precision - faster compute, you have 128GB memory
        dtype=torch.bfloat16,  # BF16 for Blackwell
        attn_implementation="sdpa",  # Force PyTorch SDPA for Blackwell GPU compatibility
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,  # *2 speeds up training
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
        random_state=3407,
    )

    # Optional: torch.compile for speedup (may add initial overhead)
    # Uncomment below for potential 1.5-2x training speedup after warmup
    # model = torch.compile(model, mode="reduce-overhead")

    return model, tokenizer, max_seq_length


# ============================================================================
# TRAINING DATA SETUP
# ============================================================================

def create_dataset():
    """Create the training dataset"""
    # Prompt optimized based on training error analysis:
    # - 38% errors were "multi-function" → explicit "ONLY ONE FUNCTION"
    # - 26% errors were truncated code → ask for concise code, no explanations
    # - Model writes verbose explanations → "CODE ONLY, no explanations"
    prompt = """Write a Python strategy function for the 2048 game.

RULES:
- Board: 4x4 grid, board[row][col], 0=empty, 2/4/8/16...=tiles
- Return EXACTLY one of: "W" (up), "A" (left), "S" (down), "D" (right)
- ONLY write the strategy function. NO helper functions, NO imports, NO explanations.

```python
def strategy(board):
    # Your code here - analyze board and return "W", "A", "S", or "D"
""".strip()

    dataset = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": prompt}],
        }
    ] * 1000)

    return dataset, prompt


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, tokenizer, dataset, max_seq_length):
    """Set up and run GRPO training"""
    # Calculate prompt length
    prompt_sample = dataset[0]["prompt"]
    maximum_length = len(tokenizer.apply_chat_template(prompt_sample, add_generation_prompt=True))
    max_prompt_length = maximum_length + 1  # + 1 just in case!

    # Allow enough tokens for explanation + complete code
    # 512 tokens should handle most strategy implementations
    max_completion_length = min(768, max_seq_length - max_prompt_length)

    training_args = GRPOConfig(
        temperature=1.1,  # Increased from 1.0 to prevent premature convergence
        learning_rate=1e-5,  # Reduced from 2e-5 to slow convergence, allow more exploration
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=4,  
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        # num_train_epochs=1,  # Set to 1 for a full training run
        max_steps=1000,  # 100 steps for learning (~10-15 min with 1.5B), increase to 500-1000 for better results
        save_steps=100,
        report_to="wandb",  # Weights & Biases logging enabled
        output_dir="outputs",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            function_works,
            no_cheating,
            strategy_succeeds,
            #complexity_reward,  # ENABLED: Prevents collapse to trivial strategies
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    return trainer


# ============================================================================
# INFERENCE
# ============================================================================

def test_model(model, tokenizer, prompt):
    """Test the trained model"""
    # Enable fast inference mode (2x speedup)
    FastLanguageModel.for_inference(model)

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )

    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        temperature=1.0,
        #max_new_tokens=300,  # Limit tokens for strategy functions
        do_sample=True,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )


# ============================================================================
# SAVING MODEL
# ============================================================================

def save_model(model, tokenizer, method="mxfp4"):
    """
    Save the trained model
    method: 'mxfp4', 'merged_16bit', or 'lora'
    """
    # Merge and save locally in mxfp4 4bit format
    if method == "mxfp4":
        model.save_pretrained_merged("finetuned_model", tokenizer, save_method="mxfp4")
    # Merge and save in 16bit
    elif method == "merged_16bit":
        model.save_pretrained_merged("finetuned_model", tokenizer, save_method="merged_16bit")
    # Save LoRA adapters
    elif method == "lora":
        model.save_pretrained("finetuned_model")
        tokenizer.save_pretrained("finetuned_model")

def push_to_hub(model, tokenizer, repo_name, token, method="mxfp4"):
    """
    Push model to Hugging Face Hub
    repo_name: format "username/repo_name"
    token: HF token from https://huggingface.co/settings/tokens
    method: 'mxfp4', 'merged_16bit', or 'lora'
    """
    if method == "mxfp4":
        model.push_to_hub_merged(repo_name, tokenizer, token=token, save_method="mxfp4")
    elif method == "merged_16bit":
        model.push_to_hub_merged(repo_name, tokenizer, token=token, save_method="merged_16bit")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    print("Loading model...")
    model, tokenizer, max_seq_length = load_model()

    print("Creating dataset...")
    dataset, prompt = create_dataset()

    print("Starting training...")
    trainer = train_model(model, tokenizer, dataset, max_seq_length)

    print("\nTraining complete! Testing model...")
    test_model(model, tokenizer, prompt)

    # Optionally save the model
    save_model(model, tokenizer, method="mxfp4")
    # push_to_hub(model, tokenizer, "your_username/gpt-oss-2048-rl", "your_token", method="mxfp4")

def evaluate():
    """assess base model capabilities"""
    print("Loading model...")
    model, tokenizer, max_seq_length = load_model()

    print("Creating dataset...")
    dataset, prompt = create_dataset()

    print(f"Prompt:\n{prompt}")
    test_model(model, tokenizer,prompt)

if __name__ == "__main__":
    # Set multiprocessing start method for robust timeout handling
    # Use 'fork' on Linux (faster, inherits memory) and 'spawn' elsewhere
    try:
        if sys.platform.startswith('linux'):
            mp.set_start_method('fork')
        else:
            mp.set_start_method('spawn')
    except RuntimeError:
        # Already set, ignore
        pass

    os.environ["TRITON_PTXAS_PATH"]="/usr/local/cuda/bin/ptxas"
    main()
    #evaluate()