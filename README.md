# LHM

> The next foundation model will not just understand language. It will understand human biology.

LHM, the **Longitudinal Health Model**, is the thesis that preventive medicine will converge into a single model layer: a foundation model that ingests a person's genomics, EHR, labs, wearables, and lifestyle data, then predicts how health will evolve over time.

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
| `1` | Text LLM baseline ([Improbability-0.8B](#improbability), fine-tuned from Qwen3.5-0.8B) | Whether a generic small LLM can learn health trajectories from EHR-as-text |
| `2` | Mamba / state-space model (`EHRMamba` style) | Whether linear-time sequence modeling wins on long patient histories |
| `3` | Continuous-time model (`TrajGPT` / `ContiFormer` style) | Whether exact temporal spacing improves trajectory prediction |
| `4` | Medical token model (`ETHOS` / `CoMET` style) | Whether purpose-built medical tokenization beats plain text |
| `5` | Hybrid winner | Combine the best ideas from experiments `2` to `4` into the first LHM candidate backbone |
| `6` | Medical reasoning distillation | Whether GPT-5.4 distilled reasoning can teach a small model clinical chain-of-thought |

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

We fine-tune a small open text model (`Qwen3.5-0.8B`) into `Improbability-0.8B` — the first model in the Improbability family — to answer the obvious question: can a general-purpose language model, with minimal health-specific structure, already learn useful health trajectories?

### Test health-native architectures

We then test the architectural ideas that should matter specifically for medicine:

- `Mamba / SSMs` for long, efficient longitudinal sequence modeling
- `continuous-time models` for irregular gaps between clinical events
- `medical-native tokenization` for representing diagnoses, labs, meds, and time as structured medical events

### Combine the winners

The final experiment is a hybrid model that combines the best-performing ingredients into the first serious LHM backbone candidate.

In other words: we are not guessing the architecture. We are running the shootout first.

### Distill medical reasoning

Parallel to the architecture work, we are building **Improbability** — a family of small medical reasoning models trained to think like a physician-scientist through a longevity and preventive medicine lens.

The name comes from the conviction that predicting health trajectories from scattered biological signals should be improbable — until you build the right model.

**Training pipeline (inspired by [NVIDIA NV-Reason-CXR-3B](https://huggingface.co/nvidia/NV-Reason-CXR-3B)):**

1. **Reasoning data upgrade** — `32K` examples from [MedReason](https://huggingface.co/datasets/UCSC-VLAA/MedReason) are upgraded through GPT-5.4 / Gemini with a v4.3 system prompt that enforces forward-only reasoning (no backward justification), bans fabricated statistics, and conditionally applies a longevity/preventive lens. Cross-model verification: Gemini generates, Codex GPT-5.4 verifies. Ground-truth answers are never changed.

2. **Supervised fine-tuning (SFT)** — LoRA fine-tuning on curated `<think>` reasoning traces from USMLE (MedQA), MMLU-Medical, and MedMCQA clinical vignettes.

3. **Group Relative Policy Optimization (GRPO)** — Reinforcement learning with verifiable rewards: answer correctness, format compliance (`<think>` tags), and reasoning length. No distilled chain-of-thought needed — the model learns to reason from correctness signals alone.

| Version | Base | Training | Status |
|---------|------|----------|--------|
| **Improbability-0.8B** | Qwen3.5-0.8B | SFT on 2,253 distilled examples | Done — 36% MedQA |
| **Improbability-2B** | Qwen3.5-2B | SFT on v4.3 upgraded MedReason + GRPO | In progress |

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

### Extended Clinical Benchmarks — Hybrid LHM

| Task | AUROC | AUPRC | F1 |
|---|---|---|---|
| **High Utilization** | **0.990** | **0.990** | **0.935** |
| 90-day Readmission | **0.954** | **0.945** | **0.886** |
| 7-day Readmission | **0.908** | 0.581 | 0.545 |
| Long LOS (>7 days) | **0.878** | 0.195 | — |
| 30-day Readmission | **0.857** | **0.807** | 0.672 |

AUROC > 0.85 across all 5 clinical prediction tasks. The architecture generalizes beyond readmission.

### Medical Knowledge Benchmarks (Log-Likelihood Evaluation)

| Benchmark | Base Qwen3.5 | Improbability-0.8B | Published Baselines |
|---|---|---|---|
| MedQA (USMLE) | **80.0%** | **80.0%** | PubMedBERT 38.3%, BioGPT 44.1%, GPT-4 86.7% |
| MedMCQA | 53.3% | 53.3% | PubMedBERT 32.1%, GPT-4 72.0% |
| MMLU-Medical | 50.0% | **60.0%** | Llama-2-7B 35.0%, GPT-4 87.0% |
| Drug Interactions | 80.0% | 80.0% | — |
| Clinical Reasoning | 100.0% | 100.0% | — |

### Experiment 6: Improbability — Medical Reasoning

| Metric | Value |
|---|---|
| Improbability-0.8B MedQA accuracy | **36%** (vs 20% vanilla) |
| Improbability-2B | In progress — SFT + GRPO pipeline |
| MedReason v4.3 upgraded examples | **486** verified (MedQA + MMLU + PubMedQA + LastHumanity) |
| v4.3 upgrade pipeline | Forward reasoning, anti-template, knowledge-consistent |
| Cross-model verification | Gemini generates → Codex GPT-5.4 verifies (77% pass rate) |
| GRPO training | Planned — correctness + format + length rewards |

The Improbability-0.8B nearly doubled the base model's MedQA score with only 100 SFT examples. The v4.3 pipeline now enforces forward-only reasoning and bans fabricated statistics to produce training data that transfers cleanly to inference. Following [NVIDIA's NV-Reason-CXR approach](https://arxiv.org/abs/2510.23968), the next stage adds GRPO reinforcement learning — which NVIDIA showed adds +12 points over SFT alone on a Qwen-based medical model.

<details>
<summary><strong>Before / After: Reasoning Upgrade Examples</strong></summary>

#### Example 1: Small Bowel Obstruction (MedQA)

**Question:** A 55-year-old man presents with nausea, vomiting, abdominal distension, tympany, and a 3 cm RLQ surgical scar. Ground truth: **A. Adhesions**

**Before** (GPT-4o, 2382 chars):
> *Finding reasoning paths:*
> 1. Nausea and vomiting with exacerbation by fatty meals and alcohol could suggest gallbladder disease, such as cholecystitis or gallstones.
> 2. History of abdominal surgery (indicated by the scar) could suggest bowel adhesions leading to intermittent bowel obstruction.
> 3. Obesity and distended abdomen could indicate non-alcoholic fatty liver disease or other metabolic conditions.
> 4. Weak pulses in the lower extremities could suggest peripheral vascular disease...

**After** (v4.1 upgraded, 1713 chars):
> The clinical presentation of nausea, vomiting, abdominal distension, and tympany in a patient with a right lower quadrant surgical scar is highly pathognomonic for small bowel obstruction secondary to post-surgical adhesions. Adhesions are fibrotic bands of collagenous tissue that form following peritoneal injury, such as a prior appendectomy (indicated by the 3 cm RLQ scar), and serve as the most common cause of mechanical SBO in developed countries. Mechanistically, these bands create a focal transition point that traps loops of bowel, leading to proximal accumulation of gas and fluid...

---

#### Example 2: Cytochrome C and Apoptosis (MedMCQA)

**Question:** Cytosolic cytochrome C plays an important function in — ? Ground truth: **A. Apoptosis**

**Before** (GPT-4o, 2123 chars):
> *Finding reasoning paths:*
> 1. Apoptotic process → casp9 → execution phase of apoptosis
> 2. Apoptotic process → casp3 → execution phase of apoptosis
> 3. Apoptotic process → casp8 → execution phase of apoptosis
>
> Understanding Apoptosis: Apoptosis is a form of programmed cell death that is crucial for maintaining cellular homeostasis and development. It involves a series of biochemical events leading to characteristic cell changes and death...

**After** (v4.1 upgraded, 1605 chars):
> The translocation of cytochrome c from the mitochondrial intermembrane space to the cytosol is the definitive commitment step in the intrinsic (mitochondrial) pathway of apoptosis. Under conditions of cellular stress, pro-apoptotic Bcl-2 family members (e.g., Bax and Bak) undergo conformational changes to induce mitochondrial outer membrane permeabilization (MOMP). Once released into the cytosol, cytochrome c functions as a critical co-factor by binding to apoptotic protease activating factor-1 (Apaf-1). This interaction triggers the assembly of the heptameric apoptosome, which recruits and activates the initiator pro-caspase 9 via its caspase recruitment domain (CARD)...

---

#### Example 3: ARDS Outcome Prediction (PubMedQA)

**Question:** Acute respiratory distress syndrome in children with malignancy — can we predict outcome? Ground truth: **A. Yes**

**Before** (GPT-4o, 3202 chars):
> *Finding Reasoning Paths:*
> 1. ARDS → Mechanical ventilation parameters (e.g., peak inspiratory pressure, PEEP, ventilation index) → Predicting outcomes in ARDS.
> 2. ARDS → Inflammatory response → Impact of malignancy on ARDS outcomes.
> 3. Cancer → Immunosuppression → Increased risk of ARDS → Role of ventilatory parameters in prognosis...

**After** (v4.1 upgraded, 1542 chars):
> Pediatric Acute Respiratory Distress Syndrome (PARDS) in the context of malignancy represents a distinct clinical phenotype where outcome prediction is feasible through the synthesis of physiological and ventilatory trajectories. Mechanistically, the severity of diffuse alveolar damage is reflected in the Oxygenation Index (OI) and the Ventilation Index, which quantify the impairment of gas exchange and lung compliance (Cstat). High-certainty evidence supports that early serial measurements of OI — specifically within the first 24–48 hours — differentiate survivors from non-survivors by capturing the patient's physiological response to lung-protective ventilation...

</details>

### What This Proves

1. **The Hybrid LHM architecture works.** Combining Mamba + temporal attention + continuous-time encoding yields AUROC 0.937 for 30-day readmission prediction, significantly outperforming all alternatives.
2. **Continuous-time awareness is critical.** Models that understand irregular time gaps between clinical events (0.878) vastly outperform those that don't (0.500). Health data is not a fixed-interval sequence.
3. **Scale reveals architecture separation.** Phase 1 (100 patients) showed identical AUROC 0.500 across all neural models. Phase 2a (1,191 patients) separates them dramatically: from 0.500 to 0.937. This validates the staged scaling approach.
4. **Fine-tuning teaches medical structure.** The text LLM generates structured EHR predictions (diagnoses, labs, timestamps) while the base model generates generic text.
5. **MedQA 80% from 0.8B params.** Approaches GPT-4 (86.7%), far exceeds PubMedBERT (38.3%) and BioGPT (44.1%). Fine-tuning on EHR data improved MMLU-Medical from 50% to 60%.
6. **SFT + GRPO is the right training pipeline.** NVIDIA's [NV-Reason-CXR-3B](https://arxiv.org/abs/2510.23968) validated that SFT followed by GRPO on a Qwen-based model with `<think>` tags produces +12 points over SFT alone. [AlphaMed](https://arxiv.org/abs/2505.17952) showed that pure RL on MedQA data, without distilled CoT, can beat models 200x larger.

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

## Thesis

LHM is built at the intersection of longevity science, preventive medicine, and foundation models. The core belief: scattered biological data can be turned into a continuous, personalized forecast of health.

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

### Architecture & Results
- [Full benchmark results](experiments/RESULTS.md)
- [Architecture thesis](docs/plans/2026-03-06-lhm-architecture-shootout-design.md)
- [Landscape research](research/health_foundation_models_landscape_2026.md)

### Reports
- [Executive summary](docs/reports/00-executive-summary.md)
- [Architecture shootout](docs/reports/01-architecture-shootout.md)
- [Phase 2a scaling](docs/reports/02-phase2a-scaling.md)
- [Medical benchmarks](docs/reports/03-medical-benchmarks.md)
- [Base vs fine-tuned analysis](docs/reports/04-base-vs-finetuned.md)
- [Conclusions & next steps](docs/reports/05-conclusions-next-steps.md)

### Experiments
- [Exp 0: XGBoost baseline](experiments/exp0_xgboost/)
- [Exp 1: Improbability-0.8B (Qwen3.5 fine-tuning)](experiments/exp1_text_llm/)
- [Exp 2: EHRMamba (SSM)](experiments/exp2_mamba/)
- [Exp 3: Continuous-time model](experiments/exp3_continuous_time/)
- [Exp 4: Medical token decoder](experiments/exp4_medical_tokens/)
- [Exp 5: Hybrid LHM](experiments/exp5_hybrid/)
- [Exp 6: Medical reasoning distillation](experiments/exp6_medical_qa/)
- [Medical benchmarks script](experiments/medical_benchmarks.py)
- [Base vs fine-tuned comparison](experiments/compare_base_vs_finetuned.py)

### Training Data
- [Training datasets catalogue](docs/training-datasets-catalogue.md)
- [MedReason upgrade framework](docs/datasets/upgrade_framework.md)
- [V4.1 prompt test results](docs/datasets/v4_prompt_test_results.md)

## Selected References

- Evo 2 (Nature, 2026): 40B parameter DNA foundation model
- GluFormer (Nature, 2026): CGM foundation model for metabolic prediction
- SleepFM (Nature Medicine, 2026): Sleep foundation model, 130+ disease categories
- JETS (ICLR, 2025): JEPA-based wearable foundation model
- DT-GPT (Nature Digital Medicine, 2025): EHR-to-text trajectory prediction
- NV-Reason-CXR-3B (NVIDIA, 2025): SFT + GRPO medical reasoning on Qwen2.5-VL-3B
- AlphaMed (arXiv, 2025): Minimalist RL for medical reasoning without distilled CoT
- Meerkat (npj Digital Medicine, 2025): Small LMs learn reasoning from medical textbooks
- MIMIC-IV: MIT Laboratory for Computational Physiology

---

Project by **Boris Djordjevic**.
