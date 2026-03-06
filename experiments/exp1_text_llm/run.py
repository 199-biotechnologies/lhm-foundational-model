"""
Experiment 1: Text-LLM (DT-GPT style)

Fine-tune Qwen3.5-0.8B on EHR-as-text patient narratives to predict
future health trajectories.

Architecture thesis: Can a generic small LLM learn health patterns
from structured text representations of patient records?
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

from src.data.mimic_loader import build_patient_records
from src.data.ehr_to_text import build_training_pairs
from src.evaluation.metrics import ExperimentResults, print_comparison_table


OUTPUT_DIR = Path(__file__).parent / "outputs"
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MAX_SEQ_LENGTH = 1024


class EHRTextDataset(Dataset):
    """Dataset of EHR text pairs (history -> future) for causal LM training."""

    def __init__(self, pairs: list[dict], tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for pair in pairs:
            # Concatenate input and output with a separator
            full_text = pair["input"] + "\n" + pair["output"] + tokenizer.eos_token
            encoded = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            if len(encoded["input_ids"]) > 10:  # Skip very short sequences
                self.examples.append(encoded)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
        }


def setup_model_and_tokenizer():
    """Load Qwen3.5-0.8B with QLoRA config."""
    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model — use float32 on MPS (Apple Silicon), bfloat16 on CUDA
    device_map = "auto"

    if torch.backends.mps.is_available():
        # Apple Silicon: float32 required for stable training (float16 causes NaN on MPS)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            trust_remote_code=True,
            device_map=None,
        ).to("mps")
    elif torch.cuda.is_available():
        # CUDA: can use 4-bit quantization
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map=device_map,
        )
    else:
        # CPU fallback
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            trust_remote_code=True,
        )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, tokenizer, trainable_params


def evaluate_generation(model, tokenizer, test_pairs: list[dict], n_samples: int = 10):
    """
    Generate predictions for test patients and evaluate quality.

    For now: qualitative evaluation (print examples).
    Quantitative evaluation of generated trajectories TBD with full dataset.
    """
    model.eval()
    device = next(model.parameters()).device

    print("\n--- Sample Predictions ---")
    for pair in test_pairs[:n_samples]:
        input_text = pair["input"]
        actual = pair["output"]

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            except RuntimeError:
                generated = "[generation failed — model produced NaN]"

        print(f"\nInput (last 200 chars): ...{input_text[-200:]}")
        print(f"Predicted: {generated[:300]}")
        print(f"Actual:    {actual[:300]}")
        print("-" * 80)


def run_experiment():
    """Run the full Qwen3.5-0.8B text-LLM experiment."""
    print("=" * 60)
    print("EXPERIMENT 1: Text-LLM (Qwen3.5-0.8B + QLoRA)")
    print("=" * 60)

    start_time = time.time()

    # Load data
    print("\n1. Building training pairs from MIMIC-IV...")
    records = build_patient_records()
    pairs = build_training_pairs(records, min_visits=2)
    print(f"   Total pairs: {len(pairs)}")

    # Split (by subject_id)
    subject_ids = list(set(p["subject_id"] for p in pairs))
    np.random.seed(42)
    np.random.shuffle(subject_ids)
    n_test = max(1, int(len(subject_ids) * 0.15))
    n_val = max(1, int(len(subject_ids) * 0.15))
    test_subj = set(subject_ids[:n_test])
    val_subj = set(subject_ids[n_test:n_test + n_val])
    train_subj = set(subject_ids[n_test + n_val:])

    train_pairs = [p for p in pairs if p["subject_id"] in train_subj]
    val_pairs = [p for p in pairs if p["subject_id"] in val_subj]
    test_pairs = [p for p in pairs if p["subject_id"] in test_subj]
    print(f"   Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    # Setup model
    print("\n2. Setting up model...")
    model, tokenizer, trainable_params = setup_model_and_tokenizer()

    # Create datasets
    print("\n3. Tokenizing data...")
    train_dataset = EHRTextDataset(train_pairs, tokenizer)
    val_dataset = EHRTextDataset(val_pairs, tokenizer)
    print(f"   Train examples: {len(train_dataset)} | Val examples: {len(val_dataset)}")

    # Training arguments
    output_path = OUTPUT_DIR / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=10,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    # Data collator with padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked
        pad_to_multiple_of=8,
    )

    # Train
    print("\n4. Training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    train_time = time.time() - start_time

    # Evaluate
    print("\n5. Evaluating...")
    eval_results = trainer.evaluate()
    print(f"   Eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")

    # Generate sample predictions
    evaluate_generation(model, tokenizer, test_pairs, n_samples=5)

    # Results
    results = ExperimentResults(
        experiment_name="Text-LLM (Qwen3.5-0.8B)",
        training_time_seconds=train_time,
        model_params=trainable_params,
        notes=f"Eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}",
    )

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n6. Results saved to {results_path}")
    print(f"   Training time: {train_time:.1f}s")

    # Save model
    model_path = OUTPUT_DIR / "model"
    model.save_pretrained(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    print(f"   Model saved to {model_path}")

    print_comparison_table([results])

    return results


if __name__ == "__main__":
    run_experiment()
