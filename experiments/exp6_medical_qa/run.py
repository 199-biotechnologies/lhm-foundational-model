"""
Experiment 6: Medical QA Instruction Tuning

Fine-tune Improbability-0.8B (already EHR-trained) on 416K medical QA questions
to improve reasoning and medical knowledge.

Training data:
  - MedQA (USMLE): 10K questions — clinical reasoning
  - MedMCQA: 183K questions with expert explanations — chain-of-thought
  - PubMedQA: 212K questions — biomedical research reasoning

This stacks medical knowledge on top of EHR trajectory understanding.
"""

import json
import sys
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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.medical_qa_loader import build_medical_qa_dataset

OUTPUT_DIR = Path(__file__).parent / "outputs"
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
EHR_MODEL_PATH = Path(__file__).parent.parent / "exp1_text_llm" / "outputs" / "model"
MAX_SEQ_LENGTH = 768  # Shorter than EHR (1024) since QA examples are more concise


class MedicalQADataset(Dataset):
    """Instruction-tuning dataset for medical QA."""

    def __init__(self, examples, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.encoded = []

        for ex in examples:
            full_text = ex["input"] + " " + ex["output"] + tokenizer.eos_token
            tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            if len(tokens["input_ids"]) > 10:
                self.encoded.append(tokens)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        item = self.encoded[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
        }


def main():
    print("=" * 70)
    print(" Experiment 6: Medical QA Instruction Tuning")
    print(" Base: Improbability-0.8B (EHR-trained)")
    print(" Data: MedQA + MedMCQA + PubMedQA (416K questions)")
    print("=" * 70)

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"\nDevice: {device}")

    # Load data — curated subset for MPS training speed
    # All MedQA (10K, gold standard), MedMCQA capped at 30K (has CoT explanations),
    # PubMedQA labeled (1K), skip artificial (211K low-quality)
    print("\n1. Loading medical QA datasets...")
    all_examples = build_medical_qa_dataset(
        include_medqa=True,
        include_medmcqa=True,
        include_pubmedqa=True,
        include_pubmedqa_artificial=False,
        max_per_source=30000,
    )

    # Split: 95% train, 5% eval
    np.random.seed(42)
    indices = np.random.permutation(len(all_examples))
    n_eval = max(500, int(len(all_examples) * 0.05))
    eval_idx = indices[:n_eval]
    train_idx = indices[n_eval:]

    train_examples = [all_examples[i] for i in train_idx]
    eval_examples = [all_examples[i] for i in eval_idx]
    print(f"  Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    # Load model — start from EHR-trained Improbability if available, else base
    print("\n2. Loading model...")
    if EHR_MODEL_PATH.exists():
        print(f"  Loading Improbability-0.8B from {EHR_MODEL_PATH}")
        model_path = str(EHR_MODEL_PATH)
    else:
        print(f"  EHR model not found, using base {BASE_MODEL}")
        model_path = BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        trust_remote_code=True,
        device_map=None,
    )

    # Check if model has LoRA adapters (PeftModel) — merge before adding new LoRA
    if hasattr(model, "peft_config") and hasattr(model, "merge_and_unload"):
        print("  Merging existing LoRA adapters...")
        model = model.merge_and_unload()

    model = model.to(device)

    # Apply fresh LoRA for medical QA training
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Tokenize datasets
    print("\n3. Tokenizing...")
    train_dataset = MedicalQADataset(train_examples, tokenizer)
    eval_dataset = MedicalQADataset(eval_examples, tokenizer)
    print(f"  Train tokens: {len(train_dataset)}, Eval tokens: {len(eval_dataset)}")

    # Training
    print("\n4. Training...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=1,  # 1 epoch over 400K examples is substantial
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch size = 32
        learning_rate=2e-5,  # Lower LR for continued fine-tuning
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,
        fp16=False,  # MPS needs float32
        bf16=False,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    # Eval
    eval_result = trainer.evaluate()
    print(f"\n  Train loss: {train_result.training_loss:.4f}")
    print(f"  Eval loss:  {eval_result['eval_loss']:.4f}")
    print(f"  Time:       {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    # Save merged model
    print("\n5. Saving model...")
    save_path = OUTPUT_DIR / "model"
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"  Saved to {save_path}")

    # Also save a merged version (base + LoRA merged for easy loading)
    print("  Creating merged model...")
    merged = model.merge_and_unload()
    merged_path = OUTPUT_DIR / "model_merged"
    merged.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    print(f"  Merged model saved to {merged_path}")

    # Save results
    results = {
        "experiment": "exp6_medical_qa",
        "base_model": model_path,
        "training_data": {
            "medqa": sum(1 for e in train_examples if e["source"] == "medqa"),
            "medmcqa": sum(1 for e in train_examples if e["source"] == "medmcqa"),
            "pubmedqa_labeled": sum(1 for e in train_examples if e["source"] == "pubmedqa_labeled"),
            "pubmedqa_artificial": sum(1 for e in train_examples if e["source"] == "pubmedqa_artificial"),
            "total": len(train_examples),
            "with_explanations": sum(1 for e in train_examples if e["has_explanation"]),
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "grad_accum": 8,
            "effective_batch_size": 32,
            "learning_rate": 2e-5,
            "train_loss": round(train_result.training_loss, 4),
            "eval_loss": round(eval_result["eval_loss"], 4),
            "elapsed_seconds": round(elapsed, 1),
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "trainable_params": trainable,
            "total_params": total,
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f" COMPLETE")
    print(f" Train loss: {train_result.training_loss:.4f}")
    print(f" Eval loss:  {eval_result['eval_loss']:.4f}")
    print(f" Time:       {elapsed/3600:.1f} hours")
    print(f" Model:      {merged_path}")
    print(f"{'=' * 70}")

    # Cleanup
    del model, merged
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
