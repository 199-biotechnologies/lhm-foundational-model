# Health Foundation Models: Landscape & Opportunities for Preventive Medicine

**Date:** March 6, 2026
**Purpose:** Research synthesis on the current state of foundation models for health, genomics, and preventive medicine — and what could be built.

---

## Executive Summary

We are in a **"transformer moment for biology"** (NVIDIA's words). In the past 6 months alone, multiple landmark foundation models have been published — trained on trillions of DNA base pairs, hundreds of thousands of hours of sleep data, and millions of glucose measurements. The convergence of genomics, wearables, EHR data, and foundation model architectures creates an unprecedented opportunity to build a **preventive health foundation model** — one that doesn't just read DNA, but integrates multi-modal health signals to predict and prevent disease years before symptoms appear.

This report maps the landscape, identifies the key players and models, and outlines a concrete path forward.

---

## 1. The DNA/Genomics Foundation Models (The "GPT Moment" for Biology)

### 1.1 Evo 2 — Arc Institute (Published in Nature, March 2026)

The model you likely saw on X. This is the biggest news right now.

- **Scale:** 40 billion parameters, trained on **9 trillion DNA base pairs** from 128,000+ species across all domains of life
- **Context:** 1 million token context window at single-nucleotide resolution
- **Dataset:** OpenGenome2 — 8.8T nucleotides from bacteria, archaea, eukaryotes, bacteriophages
- **Key capabilities:**
  - Predicts functional impact of genetic variants (including BRCA1 pathogenic mutations) **without task-specific fine-tuning**
  - Generates complete, realistic genomes (mitochondrial, prokaryotic, eukaryotic)
  - Zero-shot variant effect prediction across organisms
- **Fully open:** weights, training code, inference code, dataset
- **Architecture:** CNN trained on chunks of 8K and 1M bases (StripedHyena backbone, not transformer)

**Relevance to preventive medicine:** Can identify harmful mutations and predict disease-causing variants at population scale. Could be the genomic backbone of a health foundation model.

> Sources: [Nature](https://www.nature.com/articles/s41586-026-10176-5) | [Arc Institute](https://arcinstitute.org/tools/evo) | [Phys.org](https://phys.org/news/2026-03-evo-ai-genetic-code-domains.html)

---

### 1.2 EDEN — Basecamp Research + NVIDIA (January 2026)

- **Scale:** 28 billion parameters, trained on **9.7 trillion nucleotide tokens**
- **Architecture:** Llama3-style, next-token prediction, 8192 context length
- **Dataset:** BaseData — 10B+ novel genes from 1M+ newly discovered species, collected from 150+ locations across 28 countries over 5 years. Enriched for environmental metagenomes, phage, and mobile genetic elements.
- **Compute:** Trained on 1,008 NVIDIA H200 GPUs using BioNeMo
- **Key capabilities:**
  - **AI-Programmable Gene Insertion (aiPGI)** — designs novel Large Serine Recombinases
  - 100% success rate on tested disease-relevant target sites in human genome
  - 97% functional hit rate in antimicrobial peptide design
  - CAR T-cell integration in primary human T cells

**Relevance to preventive medicine:** Gene therapy design for hereditary disease prevention. Could enable programmable cures — cancer prevention at the genetic level.

> Sources: [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.12.699009v1) | [NVIDIA Healthcare](https://x.com/i/status/2015868598724477407) | [GEN News](https://www.genengnews.com/topics/artificial-intelligence/basecamp-research-achieves-programmable-gene-insertion-with-eden-ai-models/)

---

### 1.3 AlphaGenome — Google DeepMind (Nature, January 2026)

- **Input:** Up to 1 megabase of DNA sequence
- **Output:** ~6,000 genome tracks — gene expression, splicing, chromatin accessibility, histone modifications, TF binding, 3D contact maps — at single-base-pair resolution
- **Architecture:** U-Net backbone + transformer blocks (convolutions for local motifs, transformers for long-range enhancer-promoter interactions)
- **Training:** Two-stage — pretraining on experimental data, then distillation from ensemble of teachers
- **Performance:** State-of-the-art on 25/26 variant effect prediction benchmarks; 25% improvement on eQTL direction prediction
- **Adoption:** ~3,000 scientists from 160 countries; ~1M API calls/day
- **Open-sourced** for noncommercial use

**Relevance to preventive medicine:** Interprets the 98% of the genome that's non-coding — the "dark matter" where most disease risk variants live. Critical for understanding regulatory variants that drive cancer, neurodegeneration, and metabolic disease.

> Sources: [DeepMind Blog](https://deepmind.google/blog/alphagenome-ai-for-better-understanding-the-genome/) | [Nature](https://www.nature.com/articles/s41586-025-10014-0) | [Scientific American](https://www.scientificamerican.com/article/google-deepmind-unleashes-new-ai-alphagenome-to-investigate-dnas-dark-matter/)

---

### 1.4 Nucleotide Transformer v3 (NTv3) — InstaDeep (December 2025)

- **Architecture:** U-Net-like, single-base tokenization, contexts up to 1 Mb
- **Training:** 9 trillion base pairs from OpenGenome2 + supervised post-training on ~16,000 functional tracks from 24 species
- **Capabilities:** Combines representation learning, functional-track prediction, genome annotation, and controllable sequence generation in one backbone
- **Performance:** State-of-the-art across functional-track and genome-annotation benchmarks

> Source: [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.22.695963v1) | [InstaDeep](https://instadeep.com/research/paper/a-foundational-model-for-joint-sequence-function-multi-species-modeling-at-scale-for-long-range-genomic-prediction/)

---

## 2. Health-Specific Foundation Models (Beyond Genomics)

### 2.1 GluFormer — Weizmann Institute / NVIDIA Israel (Nature, 2026)

A foundation model for **continuous glucose monitoring (CGM)** data — directly relevant to preventive metabolic health.

- **Training:** Self-supervised on 10M+ glucose measurements from 10,812 adults (mostly without diabetes)
- **Validation:** Transferred across 19 external cohorts (n=6,044), 5 countries, 8 CGM devices
- **Key results:**
  - In prediabetics, stratified who would experience clinically significant HbA1c increases over 2 years
  - In a cohort with 11-year follow-up: **66% of incident diabetes** and **69% of cardiovascular deaths** occurred in the model's top risk quartile (vs. 7% and 0% in bottom quartile)
  - Outperformed HbA1c and standard CGM metrics
- **Multimodal extension:** Integrates dietary data to predict individual glycemic responses to food

**Relevance:** A CGM + AI system that predicts diabetes and cardiovascular death **years in advance**. This is exactly the kind of component a preventive health FM needs.

> Sources: [Nature](https://www.nature.com/articles/s41586-025-09925-9) | [MBZUAI](https://mbzuai.ac.ae/news/ai-foundation-model-gluformer-outperforms-clinical-standards-in-forecasting-diabetes-and-cardiovascular-risk/)

---

### 2.2 SleepFM — Stanford Medicine (Nature Medicine, January 2026)

A multimodal foundation model trained on **sleep polysomnography** data.

- **Training:** ~600,000 hours of sleep data from 65,000 participants (contrastive learning)
- **Modalities:** EEG, ECG, EMG, pulse oximetry, breathing airflow
- **Capabilities:** From one night of sleep, predicts risk of 130+ disease categories
- **Standout predictions:**
  - Parkinson's disease: C-index 0.89
  - Prostate cancer: 0.89
  - Breast cancer: 0.87
  - Dementia: 0.85
  - Heart attack: 0.81

**Relevance:** Sleep as a diagnostic window — a single night reveals disease trajectories years out. Perfect sensor-based preventive signal.

> Sources: [Stanford Report](https://news.stanford.edu/stories/2026/01/ai-model-sleep-disease-risk-research-sleepfm) | [Nature Medicine](https://www.nature.com/articles/s41591-025-04133-4)

---

### 2.3 Delphi-2M — EMBL-EBI / DKFZ (2026)

- Trained on 400K UK Biobank records
- Predicts probabilities for ~1,200 diseases up to **20 years out**
- Uses diagnoses + lifestyle factors
- Mentioned on X as part of the "health LLMs as weather forecasts for your future health" trend

### 2.4 DT-GPT — Roche/Flatiron (2026)

- Oncology "digital twins" — simulates disease trajectories for lung cancer and Alzheimer's over weeks to months

### 2.5 Multimodal EHR + Genomics Models — Verily / All of Us (October 2025)

- Integrates **Polygenic Risk Scores (PRS)** as a foundational data modality into EHR foundation models
- Uses transformer architecture with modality-specific encoders and cross-modal attention
- Built on the All of Us Research Program data
- Enables proactive interventions: targeted screening, lifestyle modifications, tailored prevention

> Source: [arXiv](https://arxiv.org/abs/2510.23639)

### 2.6 Longitudinal EHR + Wearable Model (January 2026)

- Jointly represents EHR data and wearable sensor streams as a unified continuous-time latent process
- Integrates medical imaging, longitudinal EHR, wearable streams, clinical text, genomics, and proteomics

> Source: [arXiv](https://arxiv.org/html/2601.12227v1)

---

## 3. NVIDIA's Health AI Ecosystem

NVIDIA is positioning itself as the **infrastructure layer** for all of biology AI. Key pieces:

| Component | Role |
|-----------|------|
| **BioNeMo** | Open platform for training/deploying biological foundation models. Includes Evo 2, GenMol, DiffDock, RNAPro NIM microservices |
| **Parabricks** | Accelerated bioinformatics (75.7% faster, 59.2% cheaper genomic analysis) |
| **NeMo Agent Toolkit** | Multi-agent AI systems for clinical and research use |
| **Clara** | Open healthcare models (RNAPro for RNA structure, ReaSyn v2 for drug synthesis) |

### Key NVIDIA Partnerships for Health FMs:

1. **ARC/Sheba + Mount Sinai + NVIDIA** (Nov 2025) — 3-year collaboration to build a **Genomic Foundation Model** targeting the 98% unexplored non-coding genome. Goal: disease prevention, diagnosis, precision medicine.

2. **Natera + NVIDIA** (Jan 2026) — Scaling multimodal AI foundation models for precision medicine. Integrating longitudinal in-vivo datasets with BioNeMo/Parabricks.

3. **Basecamp Research + NVIDIA** — EDEN model (see above). Gene therapy design.

> Sources: [NVIDIA BioNeMo](https://nvidianews.nvidia.com/news/nvidia-bionemo-platform-adopted-by-life-sciences-leaders-to-accelerate-ai-driven-drug-discovery) | [Mount Sinai](https://www.mountsinai.org/about/newsroom/2025/arc-at-sheba-medical-center-and-mount-sinai-launch-collaboration-with-nvidia-to-crack-the-hidden-code-of-the-human-genome-through-ai) | [Natera](https://www.natera.com/company/news/natera-to-scale-ai-foundation-models-in-precision-medicine-with-nvidia/)

---

## 4. The Preventive Health Foundation Model Opportunity

### 4.1 The Vision

A **multimodal preventive health foundation model** that integrates:

```
Genomics (WGS/WES + PRS)
  + Continuous biomarkers (CGM, HRV, sleep, activity)
    + Clinical data (EHR, labs, imaging)
      + Lifestyle signals (diet, exercise, stress)
        + Environmental data (exposures, microbiome)
          = Personalized health trajectory prediction
            + Intervention recommendations
```

The X post from @0xDevShah captured the vision perfectly: *"A personal AI that knows your genome, your microbiome, your blood markers, your sleep data, your stress patterns, your family history, and your environment. And runs a continuous simulation of your health trajectory."*

### 4.2 What Exists vs. What's Missing

| Layer | What Exists | What's Missing |
|-------|-------------|----------------|
| **Genomics** | Evo 2, AlphaGenome, EDEN, NTv3 | Integration with clinical phenotype at scale |
| **Metabolic** | GluFormer (CGM) | Broader metabolomics, proteomics FMs |
| **Sleep/Wearables** | SleepFM | Unified wearable FM (Apple Watch, Oura, Whoop, etc.) |
| **Clinical/EHR** | Delphi-2M, Verily EHR+PRS model | Longitudinal preventive-focused EHR FM |
| **Lifestyle** | Fragmented | No foundation model for diet/exercise/stress patterns |
| **Integration** | Early-stage multimodal EHR+wearable models | **No unified preventive health FM exists yet** |

### 4.3 Architecture Blueprint

Based on the current state of the art, a preventive health FM could be built as:

**Option A: Modular Ensemble (Pragmatic)**
- Use existing specialized FMs as frozen encoders (Evo 2 for genomics, GluFormer for metabolic, SleepFM for sleep)
- Train a cross-modal fusion layer that learns to integrate their representations
- Add a predictive head for disease risk trajectories
- Advantage: Leverage billions of dollars of existing pre-training. Ship faster.

**Option B: End-to-End Multimodal (Ambitious)**
- Train a single model on all modalities from scratch
- Architecture: Modality-specific tokenizers + shared transformer backbone + cross-attention
- Similar to what Verily is exploring with EHR + genomics
- Advantage: Learns cross-modal interactions that separate encoders miss (e.g., how a specific genetic variant interacts with a specific sleep pattern)

**Option C: Agent-Based System (Fastest to Market)**
- Use existing specialized models as tools
- Build an LLM-based health agent that queries and synthesizes across them
- Similar to the RespondHealth approach (LLMs constrained by medical knowledge graphs)
- Advantage: Deployable now. Iterative. Can incorporate new models as they ship.

### 4.4 Data Sources for Training

| Data Type | Source | Scale |
|-----------|--------|-------|
| Genomics | UK Biobank, All of Us, OpenGenome2 | 500K-1M genomes |
| CGM/Metabolic | GluFormer datasets, Levels Health, clinical cohorts | 10M+ measurements |
| Sleep | SHHS, MESA, WSC, clinical PSG databases | 600K+ hours |
| Wearables | Apple HealthKit exports, Oura API, Whoop | Millions of users (consent needed) |
| EHR | MIMIC-IV, eICU, institutional EHRs | Millions of records |
| Lifestyle | NHANES, dietary databases, MyFitnessPal, Cronometer | Population-scale |
| Longitudinal outcomes | UK Biobank (20+ year follow-up), Framingham, ARIC | Decades of data |

### 4.5 Key Technical Challenges

1. **Data heterogeneity** — Different sampling rates (genomics = static, CGM = every 5 min, EHR = irregular). Needs continuous-time modeling.
2. **Privacy and federation** — Health data is siloed. Federated learning or synthetic data may be required.
3. **Temporal alignment** — Aligning a genome (static) with a wearable stream (real-time) with an EHR (episodic).
4. **Causal inference** — Moving from prediction ("you're at risk") to intervention ("do X to reduce risk by Y%").
5. **Regulatory** — FDA clearance for clinical decision support. The first foundation model AI was just cleared by FDA in 2026.
6. **Evaluation** — Need prospective trials, not just retrospective AUCs.

### 4.6 Competitive Landscape / Who Could Build This

| Player | Strengths | Likelihood |
|--------|-----------|------------|
| **NVIDIA** | Infrastructure (BioNeMo), partnerships (Natera, ARC/Sinai), compute | Building the platform, not the end model |
| **Google DeepMind** | AlphaGenome, AlphaFold, Health AI team, Fitbit/Pixel wearable data | High — multimodal DNA + wearables |
| **Apple** | HealthKit data from 1B+ devices, health studies, ResearchKit | High — has the consumer data moat |
| **Arc Institute** | Evo 2, open-source ethos, Stanford ecosystem | Research-focused, not clinical product |
| **Verily (Alphabet)** | EHR + genomics FM, All of Us data, clinical trial infrastructure | High — already building multimodal health FMs |
| **Startup opportunity** | Agility, focus on prevention specifically | **This is where the gap is** |

---

## 5. Recommendations: What Could Be Done

### Immediate (0-6 months)
1. **Build on existing open models.** Evo 2 is fully open. AlphaGenome is open for research. GluFormer data is published. Start by combining these.
2. **Focus on a specific preventive use case** — e.g., cardiovascular disease prevention using PRS + CGM + wearable data. This has the most validation (GluFormer, PRS for CAD, SleepFM for cardiac events).
3. **Ship an agent-based MVP (Option C)** — Use existing models as tools, build an LLM health agent that synthesizes across them. Prove the concept.

### Medium-term (6-18 months)
4. **Train a cross-modal fusion model** — Take frozen Evo 2, GluFormer, and SleepFM embeddings and train a lightweight transformer to fuse them for disease prediction.
5. **Partner with a data holder** — UK Biobank (500K participants, genomics + wearables + EHR + 20-year outcomes) is the most complete dataset for this.
6. **Build the wearable FM that doesn't exist yet** — A foundation model trained on continuous multi-sensor wearable data (HR, HRV, SpO2, temperature, activity, sleep) is a massive gap.

### Long-term (18+ months)
7. **End-to-end preventive health FM** — Full multimodal architecture trained from scratch on genomics + metabolomics + wearables + EHR + lifestyle.
8. **Prospective validation trials** — Partner with health systems to run prospective prevention studies.
9. **Consumer product** — "Your AI health trajectory" — personalized, continuously updated disease risk predictions with actionable interventions.

---

## 6. Key Takeaways

1. **The DNA model you saw on X is Evo 2** — 40B params, 9T base pairs, just published in Nature (March 5, 2026). It's the biggest biological FM ever, and it's fully open.

2. **NVIDIA is everywhere** — BioNeMo is the platform, and they're partnered with everyone (Basecamp/EDEN, ARC/Sinai, Natera). They're not building the health FM directly but enabling everyone who is.

3. **Specialized health FMs are shipping fast** — GluFormer (metabolic), SleepFM (sleep), Delphi-2M (EHR disease prediction), AlphaGenome (non-coding DNA). Each solves one piece.

4. **Nobody has built the unified preventive health FM yet.** The pieces exist. The integration doesn't. This is the opportunity.

5. **The data exists** — UK Biobank, All of Us, and clinical cohorts have the multimodal longitudinal data needed. Wearable companies have the real-time sensor streams.

6. **2026 is the year to start** — The genomic FMs just hit Nature-publication maturity, wearable FMs are proven, and the regulatory environment is opening (first FM cleared by FDA). The window is open.

---

## Bibliography

### Primary Papers
1. Brixi, G. et al. "Genome modelling and design across all domains of life with Evo 2." *Nature* (2026). [Link](https://www.nature.com/articles/s41586-026-10176-5)
2. Avsec, Z. et al. "Advancing regulatory variant effect prediction with AlphaGenome." *Nature* (2026). [Link](https://www.nature.com/articles/s41586-025-10014-0)
3. Gowers, G. et al. "Designing AI-programmable therapeutics with the EDEN family of foundation models." *bioRxiv* (2026). [Link](https://www.biorxiv.org/content/10.64898/2026.01.12.699009v1)
4. Almeida, B. et al. "A foundational model for joint sequence-function multi-species modeling." *bioRxiv* (2025). [Link](https://www.biorxiv.org/content/10.64898/2025.12.22.695963v1)
5. "A foundation model for continuous glucose monitoring data." *Nature* (2026). [Link](https://www.nature.com/articles/s41586-025-09925-9)
6. "A multimodal sleep foundation model for disease prediction." *Nature Medicine* (2026). [Link](https://www.nature.com/articles/s41591-025-04133-4)
7. "Integrating Genomics into Multimodal EHR Foundation Models." *arXiv/bioRxiv* (2025). [Link](https://arxiv.org/abs/2510.23639)

### Industry & News
8. NVIDIA BioNeMo Expansion. [NVIDIA Newsroom](https://nvidianews.nvidia.com/news/nvidia-bionemo-platform-adopted-by-life-sciences-leaders-to-accelerate-ai-driven-drug-discovery)
9. ARC/Sheba + Mount Sinai + NVIDIA Collaboration. [Mount Sinai Newsroom](https://www.mountsinai.org/about/newsroom/2025/arc-at-sheba-medical-center-and-mount-sinai-launch-collaboration-with-nvidia-to-crack-the-hidden-code-of-the-human-genome-through-ai)
10. Natera + NVIDIA Precision Medicine. [Natera](https://www.natera.com/company/news/natera-to-scale-ai-foundation-models-in-precision-medicine-with-nvidia/)
11. "AI tool predicts over 1,000 diseases years before they happen." *Nature Biotechnology* (2026). [Link](https://www.nature.com/articles/s41587-026-03019-1)
12. "AI in Preventive Healthcare: 2026 Guide." [Keragon](https://www.keragon.com/blog/ai-in-preventive-healthcare)
13. Stanford SleepFM Report. [Stanford Medicine](https://med.stanford.edu/news/all-news/2026/01/ai-sleep-disease.html)
14. "Foundation Model for Advancing Healthcare: Challenges, Opportunities and Future Directions." [arXiv](https://arxiv.org/abs/2404.03264)
