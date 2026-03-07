"""
MLX-native MedQA log-likelihood benchmark.

Scores P(answer_token | prompt + option_text) for each MCQ option,
picks highest log-likelihood as the model's answer.

Usage:
    python3 scripts/mlx_medqa_benchmark.py --model <path> --samples 200
"""

import json
import random
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load


def load_medqa_samples(n=200):
    """Load n random MedQA test questions."""
    test_file = Path("data/raw/medical_qa/medqa_test.json")
    if not test_file.exists():
        # Fall back to train split
        test_file = Path("data/raw/medical_qa/medqa_train.json")
    with open(test_file) as f:
        data = json.load(f)
    random.seed(42)
    samples = random.sample(data, min(n, len(data)))
    return samples


def score_option(model, tokenizer, prompt, option_text):
    """Compute log-likelihood of option_text given prompt."""
    full_text = prompt + " " + option_text
    tokens = tokenizer.encode(full_text)
    prompt_tokens = tokenizer.encode(prompt)

    input_ids = mx.array([tokens[:-1]])
    logits = model(input_ids)
    log_probs = nn.log_softmax(logits, axis=-1)

    # Sum log probs only over the answer tokens (after prompt)
    start = len(prompt_tokens) - 1
    target_tokens = tokens[len(prompt_tokens):]
    if not target_tokens:
        # If tokenization alignment is off, score last token
        target_tokens = [tokens[-1]]
        start = len(tokens) - 2

    total_ll = 0.0
    for i, tok in enumerate(target_tokens):
        idx = start + i
        if idx < log_probs.shape[1]:
            total_ll += log_probs[0, idx, tok].item()

    # Normalize by length
    return total_ll / max(len(target_tokens), 1)


def benchmark_model(model_path, n_samples=200):
    """Run MedQA benchmark on a model."""
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    mx.eval(model.parameters())

    samples = load_medqa_samples(n_samples)
    correct = 0
    total = 0

    for i, item in enumerate(samples):
        question = item["question"]
        options = item["options"]
        correct_key = item["answer_idx"]

        prompt = f"Question: {question}\n\n"
        prompt += "\n".join(f"{k}. {v}" for k, v in options.items())
        prompt += "\n\nAnswer:"

        best_score = float("-inf")
        best_key = None

        for key, value in options.items():
            option_text = f" {key}. {value}"
            score = score_option(model, tokenizer, prompt, option_text)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key == correct_key:
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}] Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    accuracy = 100 * correct / total
    print(f"  FINAL: {correct}/{total} ({accuracy:.1f}%)")
    return accuracy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to MLX model")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    benchmark_model(args.model, args.samples)


if __name__ == "__main__":
    main()
