# Synthetic Data Generation Strategies for LHM

## Overview

Synthetic EHR data can augment limited real datasets, enable pre-training before credentialed access, and fill modality gaps. Three approaches are viable for LHM.

## 1. Frontier LLM Distillation

Use large medical LLMs to generate synthetic patient trajectories that follow clinically plausible patterns.

**Method:**
- Prompt GPT-5.4 / Gemini 3.1 Pro / Claude Opus with real MIMIC-IV patient summaries (de-identified)
- Ask for synthetic variations: different demographics, comorbidity combinations, treatment paths
- Validate statistical alignment against real population distributions
- Use generated trajectories for pre-training before fine-tuning on real data

**Key paper:** SynthEHR-Eviction (npj Digital Medicine, Feb 2026) — LLM-generated synthetic EHR for social determinant detection

**Caution:** Strong Model Collapse (Nature/ICLR 2025) warns against training on purely LLM-generated data. Use synthetic data for augmentation (20-30% of training mix), not replacement.

## 2. GAN / Diffusion-Based Synthesis

**Tools:**
- Mostly AI / Gretel.ai — high-fidelity structured EHR synthesis with differential privacy
- Syntegra — healthcare-validated synthetic control arms
- EHR-Safe (Google) — privacy-preserving EHR generation

**Method:**
- Train a generative model on real MIMIC-IV data
- Generate 10-100x synthetic patients preserving statistical properties
- Use differential privacy guarantees for regulatory compliance

## 3. DualAlign Framework (2025)

Uses frontier LLMs to maintain:
- **Statistical Alignment** — synthetic data matches real population distributions
- **Semantic Alignment** — clinical logic and temporal progression are preserved

Particularly useful for LHM because longitudinal consistency (visit ordering, disease progression, lab trends) is critical.

## Recommended Strategy for LHM

1. **Phase 2a:** Use Gemini/GPT to generate 1000 synthetic patient trajectories based on MIMIC-IV patterns. Pre-train Exp 2-5 architectures on synthetic + real mix.
2. **Phase 2b:** Once full MIMIC-IV is available, train GAN/diffusion model on real data for higher-fidelity augmentation.
3. **Phase 3:** Use frontier LLMs to generate multi-modal synthetic data (EHR + simulated wearable streams) for architecture testing before real wearable data arrives.
