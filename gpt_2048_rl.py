"""
GPT-OSS Reinforcement Learning for 2048 Game
Goal: Make GPT-OSS play games with Reinforcement Learning (GRPO)
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import random
import copy
import math
import numpy as np
from unsloth import FastLanguageModel, execute_with_time_limit, check_python_modules, create_locked_down_function
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import TextStreamer

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

def _execute_strategy(strategy: Callable, game: GameBoard):
    assert callable(strategy)

    MAX_STEPS = 500  # Hard limit to prevent infinite loops
    steps = 0
    while game.state() == "ongoing":
        if steps >= MAX_STEPS:
            return steps, "failed"  # Too many steps = bad strategy
        action = strategy(list(game.board()))
        steps += 1
        if type(action) is not str:
            return steps, "failed"
        game.do_action(action)
    return steps, game.state()

@execute_with_time_limit(2)  # Reduced from 5s - bad code should fail fast
def execute_strategy(strategy: Callable, game: GameBoard):
    return _execute_strategy(strategy, game)


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
    infinite_loop_patterns = ["while True:", "while 1:", "while True :", "while(True)"]
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

        # Block potential infinite loops
        if any(pattern in function for pattern in infinite_loop_patterns):
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
    infinite_loop_patterns = ["while True:", "while 1:", "while True :", "while(True)"]
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            # Block dangerous functions
            if any(call in function for call in dangerous_calls):
                scores.append(-20.0)
                continue
            # Block infinite loops
            if any(pattern in function for pattern in infinite_loop_patterns):
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
    # Generate a random game board with seed
    seed = np.random.randint(10000)

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
            scores.append(-1.0)  # Penalize extraction failure
            continue

        # Block dangerous functions that can hang (input, open, etc.)
        dangerous_calls = ["input(", "open(", "exec(", "eval(", "__import__"]
        if any(call in function for call in dangerous_calls):
            if PRINTER % 5 == 1:
                print(f"❌ Dangerous function call detected")
            scores.append(-2.0)  # Penalize dangerous calls
            continue

        # Block potential infinite loops
        infinite_loop_patterns = ["while True:", "while 1:", "while True :", "while(True)"]
        if any(pattern in function for pattern in infinite_loop_patterns):
            if PRINTER % 5 == 1:
                print(f"❌ Potential infinite loop detected")
            scores.append(-2.0)  # Penalize infinite loops
            continue

        # Check for forbidden modules
        ok, info = check_python_modules(function)
        if "error" in info:
            if PRINTER % 5 == 1:
                print(f"❌ Forbidden modules detected: {info}")
            scores.append(-2.0)  # Penalize forbidden modules
            continue

        # Try to create executable function
        try:
            new_strategy = create_locked_down_function(function)
        except Exception as e:
            if PRINTER % 5 == 1:
                print(f"❌ Failed to create function: {str(e)}")
            scores.append(-1.0)  # Penalize function creation failure
            continue

        # TEST: Function must return valid move (W/A/S/D)
        test_board = [[0]*6 for _ in range(6)]
        try:
            test_move = new_strategy(test_board)
            if not isinstance(test_move, str):
                if PRINTER % 5 == 1:
                    print(f"❌ Invalid return type: {type(test_move)}")
                scores.append(-3.0)  # Penalize non-string return
                continue
            if test_move not in ["W", "A", "S", "D"]:
                if PRINTER % 5 == 1:
                    print(f"❌ Invalid move: '{test_move}' (must be W/A/S/D)")
                scores.append(-3.0)  # Penalize invalid moves HEAVILY
                continue
        except Exception as e:
            if PRINTER % 5 == 1:
                print(f"❌ Function crashed on test: {str(e)}")
            scores.append(-2.0)  # Penalize crashes
            continue

        # Execute strategy on game
        try:
            game = GameBoard(size=6, seed=seed, target=2048, probability_fours=0.10)
            steps, game_state = execute_strategy(new_strategy, game)

            print(f"Steps = {steps} | State = {game_state} | Score = {game.score()}")
            print(function)
            print(game.board().pretty())

            if game_state == "success":
                print("🎉 SUCCESS! Reached 2048!")
                scores.append(20.0)  # Success - massive reward!
            else:
                # Reward based on highest tile achieved (not score)
                # This breaks the "return W" local minimum because:
                # - Simple single-direction strategies cap at ~16-32
                # - Smarter strategies can reach 64, 128, 256+
                # Level: 8->1, 16->2, 32->3, 64->4, 128->5, 256->6, 512->7
                max_tile = max(max(row) for row in game.board())
                if max_tile >= 8:
                    level = int(math.log2(max_tile)) - 2
                    tile_reward = float(level)
                    print(f"🎯 Max tile: {max_tile} -> reward: {tile_reward}")
                    scores.append(tile_reward)
                else:
                    scores.append(0.0)
        except TimeoutError as e:
            print("⏱️  Timeout (2s exceeded)")
            scores.append(-1.0)  # Timeout penalty
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
            scores.append(-3.0)  # Crash penalty

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
    max_seq_length = 768  # Room for prompt + complete strategy function
    lora_rank = 16  # Higher rank for smaller models

    # Model options (use Instruct versions for chat template support):
    # - "unsloth/Qwen2.5-0.5B-Instruct"  # Fastest: ~30-45 min total (great for learning!)
    # - "unsloth/Qwen2.5-1.5B-Instruct"  # Fast: ~1.5-2 hours total
    # - "unsloth/Qwen2.5-3B-Instruct"    # Medium: ~3-4 hours total
    # - "unsloth/gpt-oss-20b"            # Slow: ~25 hours total (reasoning model)
    model_name = "unsloth/Qwen2.5-1.5B-Instruct"  # Good balance for learning

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
    prompt = """```python
def strategy(board):
    # 2048 game: board is 6x6 list of lists (0=empty, 2/4/8/...=tiles)
    # Return EXACTLY one of: "W" "A" "S" "D" (up/left/down/right)
    # DO NOT use external functions. Keep it simple.
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
    max_completion_length = min(512, max_seq_length - max_prompt_length)

    training_args = GRPOConfig(
        temperature=1.1,  # Higher temp for exploration (was 1.0)
        learning_rate=2e-5,  # Lower LR to prevent collapse (was 5e-5)
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        num_generations=4,  # More generations for variance (was 2)
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
            complexity_reward,  # Prevents collapse to trivial strategies
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
    os.environ["TRITON_PTXAS_PATH"]="/usr/local/cuda/bin/ptxas"
    main()
    #evaluate()