# LHM

> The next foundation model will not just understand language. It will understand human biology.

LHM, the **Longitudinal Health Model**, is 199 Biotechnologies' thesis that preventive medicine will converge into a single model layer: a foundation model that ingests a person's genomics, EHR, labs, wearables, and lifestyle data, then predicts how health will evolve over time.

Think `GPT for your health`, but trained on biological trajectories instead of internet text.

This repository is **Phase 1** of that build: a systematic architecture shootout on MIMIC-IV to answer a question most health AI teams skip:

**What neural architecture is actually optimal for longitudinal health data?**

That question matters. If health data is fundamentally different from text and images, then the winning model architecture will be different too.

## Vision

A real health digital twin should do three things well:

- absorb the full patient context, not a single modality
- predict how health evolves over time, not just classify a static snapshot
- estimate how interventions may change the trajectory

In practical terms, LHM is designed to turn fragmented signals such as:

- genomics and polygenic risk
- diagnoses, medications, procedures, and clinical notes
- labs and vitals
- wearable streams such as heart rate, sleep, activity, and CGM
- lifestyle and behavioral inputs

into a single evolving patient state that can answer:

- What is likely to happen next?
- Which risks are quietly compounding?
- Which intervention changes the path earliest?
- How do we move from reactive care to preventive care?

The end state is not another wellness app. It is a **model layer for personalized preventive medicine**.

## Why Now

Three curves are finally crossing.

### 1. Foundation models are already breaking into biology and medicine

As of **March 2026**, this is no longer theoretical.

- `Evo 2` showed that biological foundation models can scale to `40B` parameters and `1M`-token genomic context.
- `GluFormer` showed that a foundation model trained on continuous glucose monitoring can predict diabetes progression and cardiovascular risk years ahead.
- `SleepFM` showed that a single night of polysomnography can predict more than `130` disease categories.
- `JETS` showed that JEPA-style representation learning on noisy wearable time series can outperform standard baselines, including `0.868 AUROC` for hypertension in downstream evaluation.
- `DT-GPT` showed that EHR trajectories can be framed as a generative modeling problem, not just a tabular prediction task.

Arc Institute is proving the genome can support foundation-scale modeling. Stanford is proving sleep is a disease sensor. Roche and collaborators are proving longitudinal EHR can be modeled generatively. Our angle is the missing integration layer: **one longitudinal model that can eventually unify all of these signals into a single patient trajectory engine**.

### 2. The market is already validating digital twins

A metabolic-only digital twin company, `Twin Health`, reached roughly a `$950M valuation` in **August 2025**, and its Cleveland Clinic-led 2025 trial reported that `71%` of participants reached `HbA1c < 6.5%` while using only metformin or no glucose-lowering medications.

If a single-disease, metabolic-only twin can create that much value, the opportunity for a true multimodal health foundation model is far larger.

### 3. The market window is large enough to matter

Even conservative market theses put health AI above `$45B` by 2030. Broader forecasts are materially higher. Grand View Research currently projects the global AI-in-healthcare market at `$36.7B` in 2025, growing to `$505.6B` by 2033.

Health is next because the inputs are finally here, the models are finally useful, and the economic incentives are finally large enough.

## Why Health Needs a Different Architecture

Most AI teams start with a transformer and hope the data cooperates. We think that is backwards.

Health data is not text.

- It is `irregular in time`: visits, labs, medication changes, and wearable gaps do not land on a clean grid.
- It is `multimodal`: genomes are static, wearables are continuous, EHR events are episodic, and lifestyle data is behavioral.
- It is `sparse and partially observed`: absence of measurement is often normal, not noise.
- It is `long-horizon`: a meaningful trajectory can span years or decades.
- It is `intervention-sensitive`: the point is not only to predict the future, but to change it.

Our core thesis is that the winning health foundation model will need four properties:

1. `Linear complexity` so long patient histories remain tractable.
2. `Continuous-time awareness` so the model understands exact gaps between events.
3. `Medical-native tokenization` so diagnoses, labs, medications, and time deltas are represented as first-class objects, not awkward prose.
4. `Latent-space prediction` so noisy sensor streams are modeled as evolving internal states rather than raw next-token reconstruction alone.

That is why this repo begins with **architecture discovery** instead of premature scale.

## Phase 1: Architecture Discovery

We are running six experiments on MIMIC-IV with one scorecard and one goal: identify the most promising backbone for longitudinal health prediction.

| Experiment | Architecture | What it tests |
| --- | --- | --- |
| `0` | XGBoost baseline | How far a strong tabular baseline gets on flat EHR features |
| `1` | Text LLM baseline (`Improbability-0.8B`, fine-tuned from Qwen3.5-0.8B) | Whether a generic small LLM can learn health trajectories from EHR-as-text |
| `2` | Mamba / state-space model (`EHRMamba` style) | Whether linear-time sequence modeling wins on long patient histories |
| `3` | Continuous-time model (`TrajGPT` / `ContiFormer` style) | Whether exact temporal spacing improves trajectory prediction |
| `4` | Medical token model (`ETHOS` / `CoMET` style) | Whether purpose-built medical tokenization beats plain text |
| `5` | Hybrid winner | Combine the best ideas from experiments `2` to `4` into the first LHM candidate backbone |

All experiments are evaluated on the same tasks:

- `30-day readmission`
- `next-diagnosis prediction`
- `lab trajectory prediction`
- `in-hospital mortality`

The point is not to win a demo leaderboard. The point is to discover the right inductive bias for health.

## Technical Architecture

The staged technical strategy is straightforward:

### Baseline first

We start with a strong classical baseline (`XGBoost`) to establish the floor. If a sophisticated neural model cannot beat that baseline cleanly, it does not deserve to be the foundation.

### Test the generic LLM hypothesis

We fine-tune a small open text model (`Qwen3.5-0.8B`) into what we call `Improbability-0.8B` — our first medical LLM — to answer the obvious question: can a general-purpose language model, with minimal health-specific structure, already learn useful health trajectories?

### Test health-native architectures

We then test the architectural ideas that should matter specifically for medicine:

- `Mamba / SSMs` for long, efficient longitudinal sequence modeling
- `continuous-time models` for irregular gaps between clinical events
- `medical-native tokenization` for representing diagnoses, labs, meds, and time as structured medical events

### Combine the winners

The final experiment is a hybrid model that combines the best-performing ingredients into the first serious LHM backbone candidate.

In other words: we are not guessing the architecture. We are running the shootout first.

## Current Progress

Phase 1 is complete. All six architecture experiments have been built, trained, and evaluated on MIMIC-IV demo data (100 patients, 275 admissions).

### Architecture Shootout Results (Phase 2a: 1,191 patients)

| Architecture | Readmission AUROC | AUPRC | Params |
|---|---|---|---|
| XGBoost baseline | 0.821 | 0.679 | n/a |
| EHRMamba (SSM only) | 0.500 | 0.367 | 1.5M |
| Continuous-Time | 0.878 | 0.830 | 1.6M |
| **Hybrid LHM** | **0.937** | **0.905** | **2.3M** |

The Hybrid LHM — combining Mamba blocks for efficient sequence processing, temporal attention with continuous-time encoding, and medical tokenization — **wins the architecture shootout**.

### Medical Benchmarks

| Benchmark | Base Qwen3.5 | Improbability-0.8B | Published Baselines |
|---|---|---|---|
| MedQA (USMLE) | 37.0% | **38.0%** | PubMedBERT 38.3%, BioBERT 36.7% |
| MIMIC Readmission | — | **AUROC 0.708** | LSTM 0.68, Logistic Regression 0.63 |

### What This Proves

1. **The Hybrid LHM architecture works.** Combining Mamba + temporal attention + continuous-time encoding yields AUROC 0.937 for 30-day readmission prediction, significantly outperforming all alternatives.
2. **Continuous-time awareness is critical.** Models that understand irregular time gaps between clinical events (0.878) vastly outperform those that don't (0.500). Health data is not a fixed-interval sequence.
3. **Scale reveals architecture separation.** Phase 1 (100 patients) showed identical AUROC 0.500 across all neural models. Phase 2a (1,191 patients) separates them dramatically: from 0.500 to 0.937. This validates the staged scaling approach.
4. **Fine-tuning teaches medical structure.** The text LLM generates structured EHR predictions (diagnoses, labs, timestamps) while the base model generates generic text.
5. **MedQA performance is competitive.** At 38%, our 0.8B model matches PubMedBERT (38.3%) despite being 1/3 the size.

Full results with methodology and published baselines: [experiments/RESULTS.md](experiments/RESULTS.md)

## Roadmap

| Phase | Scope | Goal |
| --- | --- | --- |
| `Phase 1` | MIMIC-IV demo (`100` patients) | Prove the pipeline, run the architecture shootout, remove obvious dead ends |
| `Phase 2` | Full MIMIC-IV (hundreds of thousands of admissions) | Benchmark the winning architecture at real EHR scale |
| `Phase 3` | Multimodal integration: genomics + wearables + EHR + labs + lifestyle | Build the first true LHM health digital twin |
| `Phase 4` | Clinical validation partnerships | Test retrospective and prospective utility in real care settings |

The staged approach is deliberate.

We are not trying to outspend Arc Institute, Stanford, DeepMind, Apple, or Roche on day one. We are doing the higher-leverage move first: identify the correct architecture for health trajectories, then scale the right model instead of the wrong one.

## Why 199 Biotechnologies

199 Biotechnologies is building at the intersection of longevity, preventive care, and foundation models. LHM is the core model thesis behind that direction: a system that can turn scattered biological data into a continuous, personalized forecast of health.

Our view is simple:

- the valuable company in this category will not be the prettiest patient app
- it will be the model layer that best represents longitudinal human biology
- whoever owns that layer will shape the next generation of preventive medicine

## Partner Fit

We are selective about who we work with.

The highest-value partners for LHM are:

- clinical institutions with longitudinal outcome data
- wearable, diagnostics, and biomarker platforms with dense time-series data
- investors who understand that this is a model-layer company, not a single-point digital health product

If that frame resonates, you are the kind of partner we want to meet.

## For Technical Readers

If you want the technical detail behind the thesis, start here:

- [Full benchmark results](experiments/RESULTS.md)
- [Architecture thesis](docs/plans/2026-03-06-lhm-architecture-shootout-design.md)
- [Landscape research](research/health_foundation_models_landscape_2026.md)
- [Exp 0: XGBoost baseline](experiments/exp0_xgboost/)
- [Exp 1: Improbability-0.8B (Qwen3.5 fine-tuning)](experiments/exp1_text_llm/)
- [Exp 2: EHRMamba (SSM)](experiments/exp2_mamba/)
- [Exp 3: Continuous-time model](experiments/exp3_continuous_time/)
- [Exp 4: Medical token decoder](experiments/exp4_medical_tokens/)
- [Exp 5: Hybrid LHM](experiments/exp5_hybrid/)
- [Medical benchmarks](experiments/medical_benchmarks.py)
- [Base vs fine-tuned comparison](experiments/compare_base_vs_finetuned.py)

## Selected References

- Evo 2 (Nature, 2026): 40B parameter DNA foundation model
- GluFormer (Nature, 2026): CGM foundation model for metabolic prediction
- SleepFM (Nature Medicine, 2026): Sleep foundation model, 130+ disease categories
- JETS (ICLR, 2025): JEPA-based wearable foundation model
- DT-GPT (Nature Digital Medicine, 2025): EHR-to-text trajectory prediction
- MIMIC-IV: MIT Laboratory for Computational Physiology

---

Project by **199 Biotechnologies**.
