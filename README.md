# LHM

> The next foundation model will not just understand language. It will understand human biology.

**LHM** (Longitudinal Health Model) is a foundation model for preventive medicine. It ingests a person's full health trajectory — EHR, labs, wearables, genomics, lifestyle — and predicts how their health will evolve over time.

Not a wellness app. Not a chatbot with a stethoscope. A **model layer** that turns fragmented biological signals into a continuous, personalized health forecast.

## What We Built That Didn't Exist Before

**1. Architecture discovery for health.** Every health AI team starts with a transformer and hopes. We ran a controlled shootout across 5 architectures and proved that continuous-time encoding — teaching the model that 3 days between labs is different from 3 months — is the single most important architectural ingredient. Without it: 0.500 AUROC (random). With it: 0.937.

**2. PRISM — a clinical reasoning framework for longevity medicine.** No existing medical AI distinguishes between "lab value is normal" and "lab value is suboptimal for long-term health." PRISM applies dual-threshold interpretation (standard range vs longevity-optimal range), recognizes multi-marker disease constellations, reasons about geroprotective drugs with evidence tiers, and critically — knows when *not* to apply the longevity lens (acute emergencies, pediatrics, pregnancy). This safety-aware longevity reasoning did not exist in any training dataset.

**3. A training pipeline that compounds.** We decomposed clinical reasoning into 6 modular skill packs that can be ablation-tested independently and composed incrementally. Combined with GRPO reinforcement learning (no distilled chain-of-thought needed), this lets a 2B-parameter model learn to reason from correctness signals alone — following [NVIDIA's result](https://arxiv.org/abs/2510.23968) showing +12 points over SFT.

## Results

We ran a systematic architecture shootout across 1,191 patients to find the right neural backbone for longitudinal health data. The winner:

### Architecture Shootout (Phase 2a)

| Architecture | 30-Day Readmission AUROC | Params |
|---|---|---|
| XGBoost baseline | 0.821 | n/a |
| EHRMamba (SSM only) | 0.500 | 1.5M |
| Continuous-Time | 0.878 | 1.6M |
| **Hybrid LHM** | **0.937** | **2.3M** |

The **Hybrid LHM** — Mamba blocks + temporal attention + continuous-time encoding — wins decisively.

<details>
<summary><strong>Extended benchmarks: AUROC > 0.85 across all 5 clinical prediction tasks</strong></summary>

| Task | AUROC | AUPRC | F1 |
|---|---|---|---|
| High Utilization | 0.990 | 0.990 | 0.935 |
| 90-day Readmission | 0.954 | 0.945 | 0.886 |
| 7-day Readmission | 0.908 | 0.581 | 0.545 |
| Long LOS (>7 days) | 0.878 | 0.195 | — |
| 30-day Readmission | 0.857 | 0.807 | 0.672 |

</details>

### Medical Reasoning (Improbability)

Parallel to the architecture work, we are building **Improbability** — a family of small models that reason like a physician-scientist through a preventive medicine lens.

| Version | Base | Training | MedQA |
|---------|------|----------|-------|
| **Improbability-0.8B** | Qwen3.5-0.8B | SFT on 2,253 distilled examples | 36% (vs 20% base) |
| **Improbability-2B** | Qwen3.5-2B | SFT + PRISM + GRPO | In progress |

An 0.8B model approaching GPT-4 (86.7%) on log-likelihood MedQA (80.0%), far exceeding PubMedBERT (38.3%) and BioGPT (44.1%). The next version adds reinforcement learning, which [NVIDIA showed](https://arxiv.org/abs/2510.23968) adds +12 points over SFT alone.

## The Problem No One Is Solving

Medicine is reactive. A patient gets sick, sees a doctor, gets treated. The data that could have predicted the decline — years of labs trending in the wrong direction, wearable signals degrading, risk factors compounding silently — sits in fragmented silos, interpreted one snapshot at a time.

**No model today can ingest a person's full biological trajectory and forecast where their health is heading.** Not GPT. Not Med-PaLM. Not any EHR system. The fundamental barrier is architectural: health data is irregular in time, multimodal, sparse, and long-horizon. A transformer trained on text does not have the right inductive biases for this.

We proved this empirically. A state-space model (Mamba) applied naively to EHR data scored 0.500 — random chance. Add continuous-time encoding — teaching the model that the gap between events matters as much as the events themselves — and performance jumps to 0.878. Combine it with temporal attention and medical tokenization: 0.937.

## Why Now

Three curves are crossing:

- **Foundation models are breaking into biology.** Evo 2 (40B params, genomics), GluFormer (CGM → diabetes prediction), SleepFM (sleep → 130+ diseases). The building blocks exist. The integration layer does not.
- **The market is validating digital twins.** Twin Health — a metabolic-only twin — hit ~$950M valuation. A single-disease twin creating that much value means the opportunity for a true multimodal health model is far larger.
- **The economics work.** Health AI: $36.7B in 2025, projected $505.6B by 2033 (Grand View Research).

## Approach

We do not guess the architecture. We run the shootout first, then scale the winner.

### Six experiments, one scorecard

| # | Architecture | What it tests |
|---|---|---|
| 0 | XGBoost | How far a strong tabular baseline gets |
| 1 | Improbability-0.8B (Qwen3.5) | Whether a generic LLM can learn health trajectories |
| 2 | EHRMamba (SSM) | Linear-time sequence modeling on long patient histories |
| 3 | Continuous-time model | Whether exact temporal spacing improves prediction |
| 4 | Medical token model | Whether purpose-built medical tokenization beats plain text |
| 5 | **Hybrid LHM** | Best ideas from Exp 2-4 combined into the winning backbone |

All evaluated on the same tasks: 30-day readmission, next-diagnosis prediction, lab trajectory prediction, in-hospital mortality.

### Training pipeline

The Improbability reasoning models follow a four-stage pipeline inspired by [NVIDIA NV-Reason-CXR-3B](https://huggingface.co/nvidia/NV-Reason-CXR-3B):

1. **Data upgrade** — 32K examples from [MedReason](https://huggingface.co/datasets/UCSC-VLAA/MedReason), upgraded through GPT-5.4 with forward-only reasoning, no fabricated statistics, and a conditional longevity lens. Cross-model verified.
2. **PRISM skill packs** — 1,400 examples teaching 6 distinct clinical capabilities (see [PRISM](#prism) below)
3. **Supervised fine-tuning** — LoRA on curated `<think>` reasoning traces
4. **GRPO** — reinforcement learning from correctness signals alone. No distilled chain-of-thought needed.

## PRISM

**PRISM** (Preventive Reasoning with Integrated System Medicine) is what makes Improbability different from every other medical AI.

Most medical models answer "is this value normal?" PRISM answers "is this value optimal for a long, healthy life — and if not, what biological aging process does it signal?" It is a complete clinical reasoning framework: interpretation, pattern recognition, drug reasoning, trajectory analysis, and safety firewalls — all in one system prompt.

**What it does:**

| Capability | Example |
|---|---|
| **Dual-threshold interpretation** | Fasting insulin at 18 mIU/L: standard says "normal" (<25). PRISM flags it as 2.6× the optimal ceiling (<7), indicating compensatory hyperinsulinemia and deranged nutrient sensing. |
| **Constellation recognition** | Elevated TG/HDL + insulin + uric acid + ALT/GGT → Insulin Resistance Cascade. Not three isolated findings — one interconnected pathology. |
| **Geroprotective drug reasoning** | 13 repurposable drugs (rapamycin, metformin, SGLT2i, GLP-1 RAs, etc.) with evidence tiers: [A] RCT, [B] observational, [C] preclinical. ITP lifespan data verified against primary literature. |
| **Trajectory analysis** | A rising HbA1c from 5.4→5.8% over 24 months matters more than a stable 5.9%. Rate of change, not just absolute value. |
| **Safety firewalls** | Acute emergencies, pediatric, and pregnancy contexts → longevity lens is withheld entirely. The model must know when NOT to optimize. |

**6 skill packs** (1,400 examples, GPT-5.4 generated):

| Pack | Size | Capability |
|---|---|---|
| P1: Biomarker | 300 | Dual-threshold lab interpretation across 40 markers |
| P2: Mechanism | 300 | L3-depth molecular cascade reasoning |
| P3: Metabolic | 200 | Multi-marker constellation recognition |
| P4: Repurposing | 200 | Geroprotective drug reasoning with evidence tiers |
| P5: Trajectory | 200 | Rate-of-change vs absolute value interpretation |
| P6: Routing | 200 | Clinical route classification + safety firewalls |

All reference data fact-checked against primary literature (Harrison 2009, Miller 2014, Katsuumi et al. 2024, etc.).

<details>
<summary><strong>Before / After: Reasoning quality upgrade</strong></summary>

**Question:** A 55-year-old man presents with nausea, vomiting, abdominal distension, tympany, and a 3 cm RLQ surgical scar. Ground truth: **Adhesions**

**Before** (GPT-4o):
> *Finding reasoning paths:*
> 1. Nausea and vomiting with exacerbation by fatty meals and alcohol could suggest gallbladder disease...
> 2. History of abdominal surgery could suggest bowel adhesions...
> 3. Obesity and distended abdomen could indicate non-alcoholic fatty liver disease...
> 4. Weak pulses in the lower extremities could suggest peripheral vascular disease...

**After** (v4.1 upgraded):
> The clinical presentation of nausea, vomiting, abdominal distension, and tympany in a patient with a right lower quadrant surgical scar is highly pathognomonic for small bowel obstruction secondary to post-surgical adhesions. Adhesions are fibrotic bands of collagenous tissue that form following peritoneal injury, such as a prior appendectomy (indicated by the 3 cm RLQ scar), and serve as the most common cause of mechanical SBO in developed countries...

The "before" hedges across 4 unrelated paths. The "after" drives forward from presentation to mechanism to answer — no backward justification, no fabricated statistics, no template phrases.

</details>

## What We Proved

| Finding | Evidence |
|---|---|
| **Time-awareness is the critical ingredient** | 0.500 AUROC without continuous-time encoding → 0.937 with it |
| **Architecture-first beats data-first** | 100 patients: all models identical. 1,191 patients: 0.500 to 0.937 separation |
| **Small models can reason** | 0.8B params → 80% MedQA, approaching GPT-4 (86.7%). 100 SFT examples nearly doubled generative accuracy |
| **Safety firewalls hold** | Zero longevity-keyword leakage in acute, pediatric, and pregnancy contexts |

Full results: [experiments/RESULTS.md](experiments/RESULTS.md)

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | Architecture shootout on MIMIC-IV (1,191 patients) | **Done** — Hybrid LHM wins |
| **Phase 2** | Improbability-2B: SFT + PRISM + GRPO | **In progress** |
| **Phase 3** | Multimodal: genomics + wearables + EHR + labs | Next |
| **Phase 4** | Clinical validation partnerships | Planned |

We are not trying to outspend Arc, Stanford, DeepMind, Apple, or Roche on day one. We identify the correct architecture first, then scale the right model instead of the wrong one.

## Thesis

Health will be the largest application of AI. The valuable company in this category will not be the prettiest patient app — it will be the **model layer** that best represents longitudinal human biology.

LHM is that layer. We have the architecture (proven), the reasoning framework (built), and the training pipeline (running). What comes next is scale, multimodal integration, and clinical validation.

## Partner Fit

The highest-value partners for LHM:

- **Clinical institutions** with longitudinal outcome data
- **Wearable and biomarker platforms** with dense time-series data
- **Investors** who understand model-layer companies, not single-point digital health products

If that frame resonates, you are the kind of partner we want to meet.

## Technical Deep Dive

<details>
<summary><strong>Architecture & Reports</strong></summary>

- [Full benchmark results](experiments/RESULTS.md)
- [Architecture thesis](docs/plans/2026-03-06-lhm-architecture-shootout-design.md)
- [Landscape research](research/health_foundation_models_landscape_2026.md)
- [Executive summary](docs/reports/00-executive-summary.md)
- [Architecture shootout](docs/reports/01-architecture-shootout.md)
- [Phase 2a scaling](docs/reports/02-phase2a-scaling.md)
- [Medical benchmarks](docs/reports/03-medical-benchmarks.md)
- [Base vs fine-tuned analysis](docs/reports/04-base-vs-finetuned.md)
- [Conclusions & next steps](docs/reports/05-conclusions-next-steps.md)

</details>

<details>
<summary><strong>Experiments</strong></summary>

- [Exp 0: XGBoost baseline](experiments/exp0_xgboost/)
- [Exp 1: Improbability-0.8B](experiments/exp1_text_llm/)
- [Exp 2: EHRMamba (SSM)](experiments/exp2_mamba/)
- [Exp 3: Continuous-time model](experiments/exp3_continuous_time/)
- [Exp 4: Medical token decoder](experiments/exp4_medical_tokens/)
- [Exp 5: Hybrid LHM](experiments/exp5_hybrid/)
- [Exp 6: Medical reasoning](experiments/exp6_medical_qa/)

</details>

<details>
<summary><strong>Training Data & PRISM</strong></summary>

- [Training datasets catalogue](docs/training-datasets-catalogue.md)
- [MedReason upgrade framework](docs/datasets/upgrade_framework.md)
- [PRISM v3 system prompt](docs/datasets/prism-framework/PRISM-v3-compact-system-prompt.md)
- [PRISM optimal biomarker ranges](docs/datasets/prism-framework/references/optimal-ranges.md)
- [PRISM drug repurposing evidence](docs/datasets/prism-framework/references/drug-repurposing.md)
- [PRISM skill packs generator](scripts/generate_skill_packs.py)

</details>

## References

- Evo 2 (Nature, 2026): 40B parameter DNA foundation model
- GluFormer (Nature, 2026): CGM foundation model for metabolic prediction
- SleepFM (Nature Medicine, 2026): Sleep foundation model
- DT-GPT (Nature Digital Medicine, 2025): EHR trajectory prediction
- NV-Reason-CXR-3B (NVIDIA, 2025): SFT + GRPO medical reasoning
- AlphaMed (arXiv, 2025): RL for medical reasoning without distilled CoT
- Harrison et al. (Nature, 2009): Rapamycin lifespan extension (ITP)
- Katsuumi et al. (Nature Aging, 2024): SGLT2i immunosenolysis
- MIMIC-IV: MIT Laboratory for Computational Physiology

---

Project by **Boris Djordjevic**.
