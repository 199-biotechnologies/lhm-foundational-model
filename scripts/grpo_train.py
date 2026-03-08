"""
GRPO Training for Improbability — Medical Reasoning via Reinforcement Learning

Implements Group Relative Policy Optimization (GRPO) for medical MCQ reasoning,
following the NVIDIA NV-Reason-CXR pipeline: SFT warm-start → GRPO with
verifiable rewards.

Key design decisions (from NV-Reason-CXR + AlphaMed papers):
- KL coefficient β=0 (avoid anchoring to noisy SFT)
- 3-component reward: correctness + format + length
- Temperature 1.0 for diverse generation
- Learning rate 1e-6 (10x lower than SFT)
- MedQA included fully (high informativeness), PubMedQA excluded (degrades perf)

Usage:
    # After SFT, run GRPO on raw MCQ data
    python3 scripts/grpo_train.py --model experiments/exp6_medical_qa/outputs/sft_model
    python3 scripts/grpo_train.py --model Qwen/Qwen3.5-2B --dry-run
"""

import json
import re
import argparse
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# REWARD FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def reward_correctness(completion: str, gt_letter: str) -> float:
    """Check if the model's answer matches ground truth. Continuous 0-1.

    Extracts the answer letter from <answer> tags or from the text after </think>.
    Returns 1.0 for correct, 0.0 for incorrect.
    """
    # Try <answer> tags first
    answer_match = re.search(r'<answer>\s*([A-J])', completion)
    if answer_match:
        return 1.0 if answer_match.group(1) == gt_letter else 0.0

    # Try text after </think>
    think_end = completion.find('</think>')
    if think_end >= 0:
        after_think = completion[think_end + 8:].strip()
        for char in after_think:
            if char in "ABCDEFGHIJ":
                return 1.0 if char == gt_letter else 0.0

    # Try any standalone letter near the end
    last_100 = completion[-100:]
    letters = re.findall(r'\b([A-J])\b', last_100)
    if letters:
        return 1.0 if letters[-1] == gt_letter else 0.0

    return 0.0


def reward_format(completion: str) -> float:
    """Check if <think> tags are present exactly once. Binary 0/1.

    Following NV-Reason-CXR: format acts as hard threshold in the reward formula.
    """
    think_opens = completion.count('<think>')
    think_closes = completion.count('</think>')

    if think_opens == 1 and think_closes == 1:
        # Verify ordering
        open_idx = completion.index('<think>')
        close_idx = completion.index('</think>')
        if open_idx < close_idx:
            return 1.0

    return 0.0


def reward_length(completion: str, min_tokens: int = 100) -> float:
    """Soft penalty for truncated reasoning. Continuous -1 to 0.

    Following NV-Reason-CXR: penalize responses shorter than threshold
    to discourage collapsed reasoning.
    """
    # Extract reasoning content
    think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
    if not think_match:
        return -1.0

    reasoning = think_match.group(1).strip()
    word_count = len(reasoning.split())

    if word_count >= min_tokens:
        return 0.0

    # Linear penalty
    return -1.0 * (1.0 - word_count / min_tokens)


def compute_reward(completion: str, gt_letter: str) -> float:
    """Combined reward: r = r_correctness * r_format + r_length

    Following NV-Reason-CXR formula where format acts as hard threshold.
    """
    r_cor = reward_correctness(completion, gt_letter)
    r_fmt = reward_format(completion)
    r_len = reward_length(completion)

    return r_cor * r_fmt + r_len


# ──────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────

def load_mcq_data(
    input_file: Path,
    datasets: Optional[list] = None,
    max_examples: int = 0,
) -> list:
    """Load raw MCQ data for GRPO training.

    GRPO only needs questions + ground truth letters — no reasoning traces.
    Following AlphaMed: include all MedQA, sample MedMCQA, exclude PubMedQA.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "upgrade_batch", Path(__file__).parent / "upgrade_batch.py"
    )
    ub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ub)
    extract_gt_letter = ub.extract_gt_letter

    examples = []
    with open(input_file) as f:
        for line in f:
            ex = json.loads(line)
            ds = ex["dataset_name"]

            # Filter datasets
            if datasets and ds not in datasets:
                continue

            # Skip PubMedQA variants (AlphaMed: "limited informativeness")
            if "pubmedqa" in ds:
                continue

            # Extract GT letter
            gt = extract_gt_letter(ex)
            if not gt:
                continue

            examples.append({
                "question": ex["question"],
                "options": ex.get("options", ""),
                "gt_letter": gt,
                "dataset": ds,
            })

            if max_examples and len(examples) >= max_examples:
                break

    return examples


def format_prompt(example: dict) -> str:
    """Format a question as a prompt for GRPO generation."""
    return (
        f"{example['question']}\n\n"
        f"{example['options']}\n\n"
        f"Answer:"
    )


# ──────────────────────────────────────────────────────────────────────
# GRPO TRAINING (TRL)
# ──────────────────────────────────────────────────────────────────────

def build_grpo_dataset(examples: list) -> list:
    """Convert MCQ examples to GRPO prompt format."""
    dataset = []
    for ex in examples:
        dataset.append({
            "prompt": format_prompt(ex),
            "gt_letter": ex["gt_letter"],
        })
    return dataset


def train_grpo_trl(
    model_path: str,
    examples: list,
    output_dir: str,
    num_generations: int = 8,
    learning_rate: float = 1e-6,
    num_epochs: int = 10,
    max_new_tokens: int = 512,
):
    """Train with GRPO using HuggingFace TRL library.

    Requires: pip install trl>=0.22
    Hardware: works on MPS (Apple Silicon) with smaller batch sizes.
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig
    except ImportError:
        print("ERROR: Install trl>=0.22, transformers, peft")
        print("  pip install trl transformers peft")
        return

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    # LoRA config for efficient training
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # Build dataset
    dataset = build_grpo_dataset(examples)

    # GRPO reward function — closure over dataset for GT access
    gt_map = {ex["prompt"]: ex["gt_letter"] for ex in dataset}

    def reward_fn(completions, prompts=None, **kwargs):
        """Reward function for GRPOTrainer."""
        rewards = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if prompts else ""
            gt = gt_map.get(prompt, "")
            r = compute_reward(completion, gt)
            rewards.append(r)
        return rewards

    # GRPO config — following NV-Reason-CXR hyperparameters
    config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=num_generations,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        beta=0.0,  # KL coefficient = 0 (NV-Reason-CXR: avoid anchoring to noisy SFT)
        max_grad_norm=0.2,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        seed=42,
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        config=config,
        reward_funcs=[reward_fn],
        tokenizer=tokenizer,
        peft_config=lora_config,
        train_dataset=dataset,
    )

    print(f"Starting GRPO training: {len(dataset)} prompts × {num_generations} generations")
    print(f"LR={learning_rate}, epochs={num_epochs}, β={config.beta}")
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"GRPO model saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────
# MLX-GRPO (Apple Silicon native)
# ──────────────────────────────────────────────────────────────────────

def prepare_mlx_grpo_data(examples: list, output_path: Path):
    """Prepare GRPO training data in MLX-GRPO format.

    MLX-GRPO expects JSONL with prompt + solution fields.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in examples:
            entry = {
                "prompt": format_prompt(ex),
                "solution": ex["gt_letter"],
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(examples)} GRPO training examples to {output_path}")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Improbability")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B",
                        help="Path to SFT model or HuggingFace model ID")
    parser.add_argument("--input", type=str,
                        default="docs/datasets/medreason_32k.jsonl",
                        help="Input MCQ dataset")
    parser.add_argument("--output", type=str,
                        default="experiments/exp6_medical_qa/outputs/grpo_model",
                        help="Output directory for GRPO model")
    parser.add_argument("--datasets", nargs="+",
                        default=["medqa", "medmcqa", "MMLU", "MedXpertQA"],
                        help="Dataset types to include")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Max examples (0 = all)")
    parser.add_argument("--num-generations", type=int, default=8,
                        help="Generations per prompt for GRPO")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--backend", choices=["trl", "mlx"], default="trl",
                        help="Training backend")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show data stats, don't train")
    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return

    print("=" * 70)
    print("Improbability GRPO Training")
    print(f"Model: {args.model}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Backend: {args.backend}")
    print("=" * 70)

    # Load data
    examples = load_mcq_data(input_file, args.datasets, args.max_examples)
    print(f"\nLoaded {len(examples)} MCQ examples")

    # Dataset breakdown
    from collections import Counter
    ds_counts = Counter(ex["dataset"] for ex in examples)
    for ds, count in ds_counts.most_common():
        print(f"  {ds:25s}: {count}")

    if args.dry_run:
        print("\n[DRY RUN] — no training performed")

        # Test reward functions on sample
        print("\nReward function tests:")
        good = "<think>\nThe patient presents with acute chest pain radiating to the left arm. ST elevation in leads II, III, and aVF indicates inferior wall involvement. Troponin elevation confirms myocardial injury. The most likely diagnosis is acute inferior STEMI.\n</think>\nA. Acute myocardial infarction"
        bad_format = "The answer is A because of chest pain."
        bad_answer = "<think>\nSome reasoning here about the case.\n</think>\nB. Wrong answer"

        print(f"  Good completion (GT=A): {compute_reward(good, 'A'):.2f}")
        print(f"  Bad format (GT=A):      {compute_reward(bad_format, 'A'):.2f}")
        print(f"  Wrong answer (GT=A):    {compute_reward(bad_answer, 'A'):.2f}")
        return

    if args.backend == "mlx":
        # Prepare data for MLX-GRPO
        output_path = Path(args.output) / "grpo_data.jsonl"
        prepare_mlx_grpo_data(examples, output_path)
        print(f"\nMLX-GRPO data prepared. Run with:")
        print(f"  cd MLX-GRPO && uv run mlx-grpo.py \\")
        print(f"    --config configs/production.toml \\")
        print(f"    --set model={args.model} \\")
        print(f"    --set dataset_path={output_path}")
    else:
        train_grpo_trl(
            model_path=args.model,
            examples=examples,
            output_dir=args.output,
            num_generations=args.num_generations,
            learning_rate=args.lr,
            num_epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
