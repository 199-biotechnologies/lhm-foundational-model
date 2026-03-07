# Session Log: 2026-03-07

## What Was Done

### 1. MLX LoRA Training — Fixed & Working

**Problem**: Previous training on EHR-finetuned Qwen3.5-0.8B with broken LR schedule.

**Root causes found & fixed**:
- MLX `warmup` parameter interacts badly with `grad_accumulation_steps` — LR stays near zero for 60%+ of training
- EHR-finetuned model has catastrophic forgetting (always predicts "B" on MCQs)
- LR 2e-4 too high (destroys model), 1e-5 too low (no learning)

**Solution**: Clean Qwen3.5-0.8B base + flat LR 5e-5 + no warmup schedule

**Results**:
| Model | MedQA Accuracy | Notes |
|---|---|---|
| Vanilla Qwen3.5-0.8B | 20% | B-biased (83%) |
| EHR-finetuned | 25% | B-biased (95%) |
| **Distilled (5e-5, 3 epochs)** | **36%** | Balanced A-D distribution |

**Key files**:
- Config: `experiments/exp6_medical_qa/mlx_training_config_distilled.yaml`
- Model: `experiments/exp6_medical_qa/outputs/mlx_fused_distilled/`
- Benchmark: `scripts/mlx_medqa_benchmark.py`
- Base model: `experiments/exp6_medical_qa/outputs/mlx_qwen35_base/`

### 2. MedReason Dataset Downloaded & Analyzed

Downloaded 32,682 examples from UCSC-VLAA/MedReason (HuggingFace).

**How it was built**:
- GPT-4o (`gpt-4o-0806-nofilter-global`) + PrimeKG knowledge graph
- 4-stage pipeline: Entity extraction → KG path search → Path pruning (K=3) → CoT generation
- Self-verified: 45K generated → 32,682 kept (71.4% retention)
- Cost: ~$3,600

**Quality stats**:
- Mean reasoning: 2,659 chars, min 916 chars (zero junk)
- 100% have structured Reasoning Process + Conclusion
- Their 8B model hit 71.8% MedQA after training

**Files**:
- Data: `docs/datasets/medreason_32k.jsonl` (110 MB)
- Metadata: `docs/datasets/medreason_metadata.md`

### 3. GPT-5.4 Upgrade Framework

Designed pipeline to upgrade MedReason with GPT-5.4 reasoning:
- Re-reason with deeper pathophysiology and genuine differentials
- Self-verification pass (optional)
- Domain filtering for longevity/preventive/regenerative
- Output in `<think>` format for direct Qwen 3.5 training

**Files**:
- Script: `scripts/upgrade_medreason.py`
- Design doc: `docs/datasets/upgrade_framework.md`

### 4. Training Datasets Catalogue Updated

Expanded with longevity, wearable, precision medicine, and regenerative datasets:
- LongevityBench (30K prompts, Insilico Medicine)
- ComputAgeBench (66 epigenetic datasets)
- JETS/GluFormer/SSL-Wearables (wearable FMs)
- UK Biobank, All of Us, MIMIC-IV strategy
- Custom GPT-5.4 distillation for peptide/hormone/longevity

**File**: `docs/training-datasets-catalogue.md`
