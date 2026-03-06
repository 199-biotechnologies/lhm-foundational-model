# LHM Benchmark Archive

All Phase 1 benchmark results, raw data, and run environment metadata.

## Structure

```
docs/benchmarks/
  README.md              ← this file
  RESULTS.md             ← full narrative results with tables and interpretation
  raw/
    environment.json     ← Python/PyTorch/library versions and hardware
    exp0_xgboost.json    ← XGBoost baseline results
    exp1_text_llm.json   ← Qwen3.5-0.8B LoRA results
    exp1_text_llm_eval.json ← Qwen3.5 eval metrics (loss, runtime)
    exp2_mamba.json       ← EHRMamba (SSM) results
    exp3_continuous_time.json ← Continuous-time model results
    exp4_medical_tokens.json  ← Medical token decoder results
    exp5_hybrid.json      ← Hybrid LHM results
    medical_benchmarks.json ← MedQA, PubMedQA, MIMIC clinical scores
```

## Quick Reference

### Architecture Shootout (MIMIC-IV demo, 100 patients)

| Exp | Architecture | Readmission AUROC | Mortality AUROC |
|-----|-------------|-------------------|-----------------|
| 0 | XGBoost | **0.675** | 0.500 |
| 1 | Qwen3.5-0.8B + LoRA | generative (eval_loss=0.94) | generative |
| 2 | EHRMamba | 0.500 | 0.500 |
| 3 | Continuous-Time | 0.500 | 0.500 |
| 4 | Medical Token Decoder | 0.500 | 0.500 |
| 5 | Hybrid LHM | 0.500 | 0.500 |

### Medical Knowledge (Qwen3.5-0.8B)

| Benchmark | Base | Fine-tuned | Published Baseline |
|-----------|------|------------|-------------------|
| MedQA (USMLE) | 37.0% | **38.0%** | PubMedBERT 38.3% |
| MIMIC Readmission AUROC | — | **0.708** | LSTM 0.68 |

### Run Environment

- Date: 2026-03-06
- Hardware: Apple Silicon (MPS), macOS 26.2
- Python 3.12.12, PyTorch 2.10.0, Transformers 5.1.0
- Dataset: MIMIC-IV demo (100 patients, 275 admissions)

## Notes

- Model weights (*.pt) are in `experiments/*/outputs/` but excluded from git (too large)
- To reproduce: run each experiment script, results will populate `experiments/*/outputs/`
- The `raw/` JSON files are the authoritative benchmark data — always committed to git
