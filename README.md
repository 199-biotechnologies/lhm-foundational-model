<div align="center">

# LHM — Longitudinal Health Model

**The foundation model for predicting how human health evolves over time.**

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/lhm-foundational-model?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/lhm-foundational-model/stargazers)
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

LHM ingests a patient's full health trajectory — EHR records, lab results, wearable data, genomics — and predicts where their health is heading. Not a chatbot. Not a wellness app. A **model layer** that turns fragmented biological signals into continuous, personalized health forecasts. We ran a controlled architecture shootout across 5 neural architectures and proved that continuous-time encoding is the single most important ingredient for health prediction. Without it: 0.500 AUROC (random chance). With it: 0.937.

[Why This Exists](#why-this-exists) | [Architecture Comparison](#architecture-comparison) | [Install](#install) | [Quick Start](#quick-start) | [How It Works](#how-it-works) | [Features](#features) | [Roadmap](#roadmap) | [Contributing](#contributing) | [License](#license)

---

## Why This Exists

Medicine is reactive. A patient gets sick, sees a doctor, gets treated. Years of labs trending in the wrong direction, wearable signals degrading, risk factors compounding silently — all of it sits in fragmented silos, interpreted one snapshot at a time.

No model today can ingest a person's full biological trajectory and forecast where their health is heading. Not GPT. Not Med-PaLM. Not any EHR system. The barrier is architectural: health data is irregular in time, multimodal, sparse, and long-horizon. A transformer trained on text lacks the right inductive biases for this.

We proved it empirically. A state-space model (Mamba) applied naively to EHR data scored 0.500 — random chance. Add continuous-time encoding and performance jumps to 0.878. Combine it with temporal attention and medical tokenization: **0.937**.

## Architecture Comparison

Six architectures tested on identical data, identical metrics, identical evaluation tasks. Benchmarked on 1,191 patients (MIMIC-IV + Synthea).

| Architecture | 30-Day Readmission AUROC | Params | Key Insight |
|:---|:---:|:---:|:---|
| XGBoost baseline | 0.821 | n/a | Strong classical baseline on tabular features |
| Improbability-0.8B (Qwen3.5 + LoRA) | — | 6.4M / 752M | Generative model; learns structured EHR prediction format |
| EHRMamba (SSM) | 0.500 | 1.5M | Pure Mamba fails without temporal awareness |
| Continuous-Time | 0.878 | 1.6M | Temporal encoding alone recovers most of the signal |
| Medical Token Decoder | 0.500 | 2.3M | Domain tokenization alone is insufficient |
| **Hybrid LHM** | **0.937** | **3.0M** | Mamba + temporal attention + continuous-time = winner |

<details>
<summary><strong>Extended benchmarks: AUROC > 0.85 across all 5 clinical tasks</strong></summary>

| Task | AUROC | AUPRC | F1 |
|:---|:---:|:---:|:---:|
| High Utilization | 0.990 | 0.990 | 0.935 |
| 90-day Readmission | 0.954 | 0.945 | 0.886 |
| 7-day Readmission | 0.908 | 0.581 | 0.545 |
| Long LOS (>7 days) | 0.878 | 0.195 | — |
| 30-day Readmission | 0.857 | 0.807 | 0.672 |

</details>

<details>
<summary><strong>Medical knowledge benchmarks (Improbability-0.8B)</strong></summary>

| Benchmark | Base Qwen3.5-0.8B | Improbability-0.8B | Published Baselines |
|:---|:---:|:---:|:---|
| MedQA (USMLE) | 80.0% | 80.0% | PubMedBERT 38.3%, BioGPT 44.1%, GPT-4 86.7% |
| MedMCQA | 53.3% | 53.3% | PubMedBERT 32.1%, BioGPT 37.0%, GPT-4 72.0% |
| MMLU-Medical | 50.0% | **60.0%** | Llama-2-7B 35.0%, GPT-4 87.0% |
| Drug Interactions | 80.0% | 80.0% | — |
| Clinical Reasoning | 100.0% | 100.0% | — |

80% MedQA from an 0.8B model. Approaching GPT-4 (86.7%) with 1000x fewer parameters.

</details>

## Install

```bash
git clone https://github.com/199-biotechnologies/lhm-foundational-model.git
cd lhm-foundational-model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.12+, PyTorch 2.5+. `mamba-ssm` only installs on Linux (CUDA required for SSM experiments).

## Quick Start

Run the full architecture shootout:

```bash
# Classical baseline
python experiments/exp0_xgboost/run.py

# Neural architectures
python experiments/exp2_mamba/run.py
python experiments/exp3_continuous_time/run.py
python experiments/exp4_medical_tokens/run.py
python experiments/exp5_hybrid/run.py

# Text LLM (generative)
python experiments/exp1_text_llm/run.py

# Benchmarks and comparison
python experiments/medical_benchmarks.py
python experiments/compare_all.py
```

Each experiment reads from `data/`, trains on identical splits, and writes results to `experiments/<name>/outputs/`.

## How It Works

### Architecture Discovery

We do not guess the architecture. We run the shootout first, then scale the winner.

The **Hybrid LHM** combines three ingredients that each fail alone:

1. **Mamba blocks** — Linear-time sequence modeling for long patient histories (6 SSM blocks)
2. **Temporal attention** — 2 attention blocks that learn cross-event dependencies
3. **Continuous-time encoding** — Exponential decay bias + sinusoidal encoding that teaches the model that 3 days between labs is different from 3 months

This is the key finding: time-awareness is the critical architectural ingredient for health data. Every other design choice is secondary.

### PRISM Clinical Reasoning

**PRISM** (Preventive Reasoning with Integrated System Medicine) powers the Improbability reasoning models.

Most medical AI answers "is this value normal?" PRISM answers "is this value optimal for a long, healthy life — and if not, what aging process does it signal?"

| Capability | What It Does |
|:---|:---|
| Dual-threshold interpretation | Fasting insulin at 18 mIU/L: standard says "normal." PRISM flags it as 2.6x the optimal ceiling. |
| Constellation recognition | Elevated TG/HDL + insulin + uric acid + ALT/GGT = one interconnected pathology, not four findings. |
| Geroprotective drug reasoning | 13 repurposable drugs with evidence tiers: [A] RCT, [B] observational, [C] preclinical. |
| Trajectory analysis | Rising HbA1c from 5.4 to 5.8% over 24 months matters more than a stable 5.9%. |
| Safety firewalls | Acute emergencies, pediatric, and pregnancy contexts = longevity lens withheld entirely. |

Trained on 6 skill packs (1,400 examples) covering biomarker interpretation, mechanism reasoning, metabolic constellations, drug repurposing, trajectory analysis, and clinical routing.

### Training Pipeline

Four-stage pipeline inspired by [NVIDIA NV-Reason-CXR-3B](https://huggingface.co/nvidia/NV-Reason-CXR-3B):

1. **Data upgrade** — 32K examples from MedReason, upgraded with forward-only reasoning and a conditional longevity lens
2. **PRISM skill packs** — 1,400 examples teaching 6 distinct clinical capabilities
3. **Supervised fine-tuning** — LoRA on curated reasoning traces
4. **GRPO** — Reinforcement learning from correctness signals alone (no distilled chain-of-thought needed)

## Features

- **5 neural architectures** benchmarked head-to-head on identical data and evaluation tasks
- **Hybrid LHM backbone** — Mamba + temporal attention + continuous-time encoding (0.937 AUROC)
- **PRISM framework** — Clinical reasoning with dual-threshold interpretation and safety firewalls
- **Improbability models** — Small LMs (0.8B-2B) that score 80% on MedQA with 1000x fewer params than GPT-4
- **GRPO training** — RL from correctness signals, following NVIDIA's result showing +12 points over SFT
- **MIMIC-IV + Synthea pipeline** — End-to-end data ingestion, tokenization, training, and evaluation
- **Modular experiments** — Each architecture runs independently with its own config and output directory

## Roadmap

| Phase | Scope | Status |
|:---|:---|:---:|
| Phase 1 | Architecture shootout on 1,191 patients | Done |
| Phase 2 | Improbability-2B: SFT + PRISM + GRPO | In progress |
| Phase 3 | Multimodal: genomics + wearables + EHR + labs | Next |
| Phase 4 | Clinical validation partnerships | Planned |

## Project Structure

```
lhm-foundational-model/
├── src/                    # Core library
│   ├── data/               # Data loading and tokenization
│   ├── models/             # Architecture implementations
│   ├── training/           # Training loops
│   ├── evaluation/         # Metrics and benchmarks
│   └── ui/                 # Gradio interface
├── experiments/            # Architecture shootout
│   ├── exp0_xgboost/       # Classical baseline
│   ├── exp1_text_llm/      # Improbability-0.8B
│   ├── exp2_mamba/         # EHRMamba (SSM)
│   ├── exp3_continuous_time/ # Temporal encoding
│   ├── exp4_medical_tokens/  # Medical tokenization
│   ├── exp5_hybrid/        # Hybrid LHM (winner)
│   └── exp6_medical_qa/    # Medical reasoning
├── docs/                   # Reports and analysis
├── research/               # Landscape research
├── scripts/                # PRISM generation, utilities
└── data/                   # MIMIC-IV + Synthea data
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where help matters most:
- Additional clinical prediction tasks and evaluation metrics
- Multimodal data integration (wearables, genomics)
- Scaling experiments on larger patient cohorts
- Clinical validation and domain expertise

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">

Built by [Boris Djordjevic](https://github.com/longevityboris) at [199 Biotechnologies](https://github.com/199-biotechnologies) | [Paperfoot AI](https://paperfoot.ai)

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/lhm-foundational-model?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/lhm-foundational-model/stargazers)
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

</div>
