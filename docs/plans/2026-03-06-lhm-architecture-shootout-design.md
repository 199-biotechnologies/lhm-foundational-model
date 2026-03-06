# LHM: Health Trajectory Prediction Foundation Model — Architecture Shootout

**Date:** 2026-03-06
**Status:** Approved
**Type:** Proof of Principle / Architecture Exploration

---

## 1. Goal

Rapidly test which neural architecture best predicts health outcomes from patient data, to establish the architectural foundation for a larger health foundation model ("LHM"). This is a proof of principle — speed and learning are prioritized over production-readiness.

## 2. Thesis

Health data is fundamentally different from natural language:

- **Irregular in time** — lab visits, doctor appointments, wearable gaps are all non-uniform
- **Multimodal** — static (genome), streaming (wearables), episodic (clinical events)
- **Long-range** — a patient's history spans decades, not paragraphs
- **Continuous dynamics** — disease progression is continuous, not discrete tokens

Therefore, the optimal health prediction architecture should be:

1. **Linear-complexity** for long patient histories (SSM/Mamba over vanilla attention)
2. **Continuous-time aware** for irregular sampling (Neural ODE positioning)
3. **Latent-space predictive** for sensor data (JEPA over raw reconstruction)
4. **Sparse/MoE** for multi-domain health (different experts for different body systems)

No single existing model combines all four. We test each independently, then combine the winners.

## 3. Architectural Landscape (March 2026)

### What Exists

| Model | Architecture | Data | Key Result |
|-------|-------------|------|------------|
| DT-GPT | LLM fine-tune (BioMistral) | Flatiron NSCLC EHR | Outperforms ML baselines for trajectory prediction |
| EHRMamba | Mamba SSM | MIMIC-IV | SOTA on 6 clinical tasks, 300% longer sequences |
| HyMaTE | Hybrid Mamba+Transformer | EHR | Outperforms either alone |
| TrajGPT | Selective Recurrent Attention + ODE | Irregular health time series | Handles arbitrary future timesteps |
| ContiFormer | Continuous-time Transformer + Neural ODE | Irregular time series | Extends attention to continuous domain |
| CoMET | Decoder-only transformer, ETHOS tokenizer | 118M patients (Epic Cosmos) | 78 clinical tasks, scales with data |
| JETS | JEPA | 3M person-days Apple Watch | 87% AUC hypertension from wearables |
| GluFormer | Transformer, next-token prediction | 10M glucose measurements | Predicts diabetes/CVD death years out |
| SleepFM | Contrastive learning | 600K hours sleep data | 130+ diseases from one night |

### Key Architecture Innovations

- **DeepSeek MLA** — Compresses KV cache via low-rank latent vectors, massive inference savings
- **Qwen3.5** — Hybrid MoE, improved architecture, scaled RL (released March 2, 2026)
- **Mamba-3** — Improved selective SSM with better long-range recall
- **Forward-Forward** — 41% less energy, 34% faster training, but accuracy lags (monitor, don't adopt)

## 4. Design

### 4.1 Base Model

**Qwen3.5-0.8B-Base** (released March 2, 2026)

- Latest small model architecture from Alibaba
- Apache 2.0, base variant available for fine-tuning
- Unsloth support for 2x faster, 70% less VRAM fine-tuning
- Can run on Apple Silicon M-series

Comparison: Qwen3.5-2B if 0.8B lacks capacity.

### 4.2 Data

**Primary:** MIMIC-IV (PhysioNet)
- ~300K hospital admissions, ~65K ICU stays
- Labs, diagnoses (ICD-10), procedures, medications, vitals, clinical notes
- Longitudinal — patients with multiple admissions 2008-2019
- Free, credentialed access (1-3 days with CITI training)

**Development:** MIMIC-IV Demo (100 patients, instant access, no credentialing)

### 4.3 Experiments

| # | Name | Architecture | What It Tests | Base | Est. Time |
|---|------|-------------|--------------|------|-----------|
| 0 | Baseline | XGBoost on tabular features | Canonical ML comparison | scikit-learn | 1 day |
| 1 | Text-LLM | EHR-to-text, fine-tune (DT-GPT style) | Can a generic LLM learn health patterns from text? | Qwen3.5-0.8B | 2 days |
| 2 | Mamba-EHR | State space model on tokenized EHR | Does Mamba beat transformers on long patient histories? | EHRMamba / Odyssey | 2 days |
| 3 | Continuous-time | Neural ODE + attention for irregular timestamps | Does continuous-time modeling help with irregular clinical data? | TrajGPT-style | 2-3 days |
| 4 | Medical tokens | Custom ETHOS-style tokenizer + small decoder | Does purpose-built tokenization beat text? | From scratch | 2 days |
| 5 | Hybrid winner | Best of 2+3+4 combined | Does combining the best ideas win? | Custom | 3 days |

**Total: ~12-14 days**

### 4.4 Evaluation

Same evaluation protocol across all experiments for fair comparison:

**Tasks:**
1. **30-day readmission** — Binary classification (AUC-ROC)
2. **Next-diagnosis prediction** — Multi-label, top-k accuracy
3. **Lab value trajectory** — Regression, MAE vs. actual future values
4. **In-hospital mortality** — Binary classification (AUC-ROC, AUPRC)

**Split:** 70/15/15 train/val/test on MIMIC-IV, patient-level split (no data leakage)

**Baselines:** XGBoost (Experiment 0) + published results from EHRMamba paper

### 4.5 Deliverable

1. Comparison table: all architectures ranked by task performance
2. Working prototype of the winning architecture
3. Simple Gradio/Streamlit UI for interactive trajectory prediction
4. "What-if" intervention simulation (change meds in input, observe trajectory change)
5. Written analysis: which architecture and why, recommendations for full model

## 5. Stack

- **Language:** Python 3.11+
- **Training:** PyTorch, Unsloth (QLoRA), HuggingFace Transformers
- **Models:** Qwen3.5-0.8B-Base, EHRMamba (Odyssey toolkit)
- **Data:** MIMIC-IV via PhysioNet, pandas, polars for processing
- **Experiment tracking:** Weights & Biases or CSV-based logging
- **UI:** Gradio or Streamlit
- **Compute:** Apple Silicon M-series (dev/small runs), cloud A100 if needed

## 6. Project Structure

```
lhm-foundational-model/
  docs/
    plans/           # This design doc + implementation plan
  research/          # Background research reports
  data/
    raw/             # MIMIC-IV downloads (gitignored)
    processed/       # Processed patient records
  src/
    data/            # Data loading, EHR-to-text conversion, tokenizers
    models/          # Model definitions for each experiment
    training/        # Training loops, configs
    evaluation/      # Evaluation metrics, comparison framework
    ui/              # Gradio/Streamlit demo
  experiments/       # Experiment configs and results
    exp0_xgboost/
    exp1_text_llm/
    exp2_mamba/
    exp3_continuous_time/
    exp4_medical_tokens/
    exp5_hybrid/
  notebooks/         # Exploration and analysis
  tests/             # Unit tests for data pipeline
```

## 7. Phase 2 (After Shootout)

Once the winning architecture is identified:

1. Add wearable/sensor data (vitals summaries appended to patient records)
2. Add genomics via PRS proxy (family history, demographics)
3. Apply for UK Biobank (15-week timeline, 500K participants with genomics + EHR + wearables + 20yr outcomes)
4. Scale model (1B+ parameters)
5. Cross-modal fusion: frozen Evo 2 + GluFormer + SleepFM embeddings as input features

## 8. Phase 3 (Full Foundation Model)

- End-to-end multimodal architecture trained from scratch
- Custom medical tokenizer + modality-specific encoders + cross-attention fusion
- Training on UK Biobank + All of Us + clinical cohorts
- Prospective validation trials
- Consumer product: continuous personal health trajectory predictions

## 9. References

### Architecture Papers
- DT-GPT: [npj Digital Medicine](https://www.nature.com/articles/s41746-025-02004-3) | [GitHub](https://github.com/MendenLab/DT-GPT)
- EHRMamba: [arXiv](https://arxiv.org/abs/2405.14567) | [Odyssey toolkit](https://github.com/VectorInstitute/odyssey)
- HyMaTE: [arXiv](https://arxiv.org/html/2509.24118v1)
- TrajGPT: [arXiv](https://arxiv.org/html/2410.02133v1)
- ContiFormer: [arXiv](https://arxiv.org/abs/2402.10635)
- CoMET: [arXiv](https://arxiv.org/html/2508.12104v1)
- JETS/JEPA: [Empirical Health](https://www.empirical.health/blog/wearable-foundation-model-jets/)

### Health Foundation Models
- Evo 2: [Nature](https://www.nature.com/articles/s41586-026-10176-5)
- AlphaGenome: [Nature](https://www.nature.com/articles/s41586-025-10014-0)
- EDEN: [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.12.699009v1)
- GluFormer: [Nature](https://www.nature.com/articles/s41586-025-09925-9)
- SleepFM: [Nature Medicine](https://www.nature.com/articles/s41591-025-04133-4)

### Base Model
- Qwen3.5 Small Series: [HuggingFace](https://huggingface.co/Qwen) | [Announcement](https://x.com/Alibaba_Qwen)
