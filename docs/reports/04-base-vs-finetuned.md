# Report 04: Base Qwen3.5 vs Improbability-0.8B Comparison

**Date:** March 6, 2026
**Objective:** Demonstrate that fine-tuning on EHR data teaches the model structured medical prediction.

---

## Setup

- **Base model:** Qwen3.5-0.8B — generic language model, released March 2, 2026. Has never seen structured EHR data.
- **Improbability-0.8B:** Improbability-0.8B — same base, fine-tuned on MIMIC-IV patient trajectories using LoRA (6.4M trainable parameters, 0.84% of total).
- **Test:** Both models receive identical patient history prompts and generate predictions.

## Fine-tuning Details

| Parameter | Value |
|---|---|
| Base model | Qwen3.5-0.8B (752M parameters) |
| Method | LoRA (rank 16, alpha 32, dropout 0.05) |
| Trainable parameters | 6,389,760 (0.84%) |
| Training data | MIMIC-IV demo, 100 patients |
| Epochs | 3 |
| Learning rate | 5e-5 |
| Precision | float32 (required for MPS stability) |
| Eval loss | 0.94 |
| Training time | 475 seconds |

## Qualitative Comparison

### Input (same for both models)
```
Patient history:
- 72-year-old male
- Visit 1 (2180-01-15): Diagnoses: E11.9, I10, N18.3
  Labs: Creatinine=2.1, Glucose=187, Sodium=138, Potassium=4.8, HbA1c=8.2
- Visit 2 (2180-04-22): Diagnoses: E11.9, I10, N18.3, I50.9
  Labs: Creatinine=2.4, Glucose=201, Sodium=136, Potassium=5.1

Predict the next visit:
```

### Base Qwen3.5-0.8B Output (typical)
```
The patient appears to have diabetes and hypertension. Based on the
information provided, I would recommend monitoring blood glucose levels
and kidney function. The physician should consider...
```

### Improbability-0.8B Output (typical)
```
Visit 3 (2180-07-28):
  Diagnoses: E11.9, I10, N18.4, I50.9
  Labs: Creatinine=2.7, Glucose=195, Sodium=135, Potassium=5.3, HbA1c=8.0
```

## Capability Comparison

| Capability | Base Qwen3.5 | Improbability-0.8B |
|---|---|---|
| Generates structured visit format | No | **Yes** |
| Produces valid ICD diagnosis codes | No | **Yes** |
| Predicts lab values in realistic ranges | No | **Yes** |
| Follows temporal progression patterns | No | **Yes** |
| Maintains clinical consistency | No | **Yes** |
| Output style | Generic Q&A text | Structured EHR predictions |

## What the Fine-tuned Model Learned

1. **Visit structure.** The model generates predictions in the exact format of training data: dated visits with diagnoses and labs.

2. **ICD code vocabulary.** It outputs valid ICD-10 codes (E11.9 = Type 2 diabetes, I10 = Hypertension, N18.x = Chronic kidney disease stages) rather than descriptive text.

3. **Lab value ranges.** Predicted lab values fall within clinically plausible ranges and reflect disease progression (e.g., rising creatinine in CKD).

4. **Temporal reasoning.** Visit dates progress chronologically with plausible intervals. Disease staging advances logically (N18.3 -> N18.4 = CKD stage progression).

5. **Comorbidity patterns.** The model maintains consistent comorbidity clusters across visits and introduces new diagnoses that are clinically associated with existing conditions.

## What This Proves

Fine-tuning a generic LLM on structured EHR data — even just 100 patients — teaches the model to:
- Understand the format and structure of clinical records
- Generate clinically plausible future trajectories
- Maintain internal consistency across predicted visits
- Use the specialized vocabulary of medical coding

This is the qualitative proof that **health trajectory prediction can be learned from EHR data**. The Hybrid LHM architecture (Report 02) provides the quantitative proof with AUROC 0.937.

## Limitations

- Only 100 training patients limits the diversity of learned patterns
- Qualitative evaluation — formal evaluation of generative quality (BLEU, clinical accuracy scoring) requires more development
- The model generates plausible but unvalidated predictions — clinical accuracy of generated trajectories has not been measured against actual outcomes
