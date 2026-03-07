# Medical Training Datasets Catalogue

Last updated: 2026-03-07

## Reasoning / Chain-of-Thought Datasets

| Dataset | Size | Date | Thinking | Format | Source | License | Notes |
|---------|------|------|----------|--------|--------|---------|-------|
| **[OpenMed Medical-Reasoning-SFT-Mega](https://huggingface.co/datasets/OpenMed/Medical-Reasoning-SFT-Mega)** | **1.79M samples, 3.78B tokens** | Feb 2026 | Deep CoT reasoning | Question + CoT + Response | 7 SOTA AI models, deduplicated from 2.9M | Apache 2.0 | **Largest medical reasoning SFT dataset.** Combines outputs from 7 models with fair distribution dedup. |
| **[ReasonMed](https://huggingface.co/datasets/lingshu-medical-mllm/ReasonMed)** | **370K** (1.1M across 3 formats) | Jun 2025 | Full CoT + summary | CoT + Response, CoT-only, Response-only | Qwen-2.5-72B, DeepSeek-R1-70B, HuatuoGPT-o1-70B | Open | Multi-agent verified. 3 formats: ReasonMed (CoT+answer), CoTMed (CoT only), ResponseMed (answer only). By Alibaba DAMO. |
| **[II-Medical-Reasoning-SFT](https://huggingface.co/datasets/Intelligent-Internet/II-Medical-Reasoning-SFT)** | **2.2M samples** | 2025 | Deep reasoning chains | Instruction + reasoning + answer | Multiple sources | Open | Used to train II-Medical-8B (top of Open Medical LLM leaderboard). |
| **[Medical-O1-Reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)** | **~50K** (58MB medical, 73MB mixed) | Dec 2024 | Deep CoT (`Complex_CoT` column) | Question + Complex_CoT + Response | GPT-4o generated, medical verifier validated | Open | Built for HuatuoGPT-o1. Uses medical verifier to validate reasoning. Columns: Question, Complex_CoT, Response. |
| **[MedReason (UCSC-VLAA)](https://huggingface.co/datasets/UCSC-VLAA/MedReason)** | **32,682** | Apr 2025 | KG-grounded CoT | Question + thinking path + answer | Medical Knowledge Graph traces from MedQA, MedMCQA, PubMedQA, MMLU, HuatuoGPT-o1, MedXpert | Open | **3rd place HF Reasoning Competition.** Uses knowledge graphs for factual reasoning paths. +7.7% on DeepSeek-8B. MedReason-8B beats HuatuoGPT-o1-8B by 4.2%. |
| **[MedReason-Stenographic (OpenMed)](https://huggingface.co/OpenMed)** | **31K** | Jan 2026 | Compressed symbolic CoT | Compressed reasoning traces | MiniMax M2.1 | Apache 2.0 | "CoT but 10x denser" — compressed symbolic reasoning format. By OpenMed community. |
| **[Meerkat Instructions](https://huggingface.co/datasets/dmis-lab/meerkat-instructions)** | **441K** (incl. 78K textbook CoT) | May 2025 | GPT-4 generated CoT | Instruction + CoT reasoning | 18 medical textbooks + MedQA-CoT (9.3K) + ChatDoctor (112K) | Open | Published in npj Digital Medicine. Textbook-sourced CoT is unique — covers 16 medical disciplines. |
| **[MedQA-Mixtral-CoT](https://huggingface.co/datasets/HPAI-BSC/MedQA-Mixtral-CoT)** | **~10K** | 2024 | Mixtral-generated CoT | MedQA questions + step-by-step reasoning | Mixtral-8x7B | Open | Drop-in CoT replacement for MedQA. |
| **[Medprompt-MedQA-CoT](https://huggingface.co/datasets/HPAI-BSC/Medprompt-MedQA-CoT)** | **~10K** | 2024 | RAG + CoT | MedQA + retrieved context + reasoning | Llama-3.1-70B-Instruct | Open | Retrieval-augmented CoT reasoning for MedQA. |

## General Medical Instruction Tuning

| Dataset | Size | Date | Thinking | Format | Source | License | Notes |
|---------|------|------|----------|--------|--------|---------|-------|
| **[MedS-Ins](https://huggingface.co/datasets/Henrychur/MedS-Ins)** | **5M instances, 19K instructions** | 2024 | No (task-oriented) | 58 medical corpora, 122 clinical tasks | 5 text sources, 19 task categories | Open | Massive scale. 122 clinical NLP tasks. Published in npj Digital Medicine. |
| **[AlpaCare-MedInstruct-52k](https://huggingface.co/datasets/lavita/AlpaCare-MedInstruct-52k)** | **52K** | 2024 | No | Medical instruction-response pairs | GPT-4 generated | Open | General medical instruction following. |
| **[ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)** | **100K** | 2023 | No | Patient question + doctor answer | Real doctor-patient conversations | Open | Real clinical Q&A from HealthCareMagic platform. |
| **[Medical-Instruction-120k](https://huggingface.co/datasets/Mohammed-Altaf/medical-instruction-120k)** | **120K** | 2024 | No | Instruction + response | Curated medical instructions | Open | Designed for edge cases and complex clinical scenarios. |
| **[MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD)** | **47K** | 2023 | No | Question + answer | 12 NIH sources | Open | Authoritative NIH-sourced medical Q&A. |

## Multimodal / Specialized

| Dataset | Size | Date | Thinking | Format | Source | License | Notes |
|---------|------|------|----------|--------|--------|---------|-------|
| **[MultiCaRe](https://huggingface.co/collections/openmed-community/multicare)** | **85K cases, 160K images** | Sep 2025 | No | Clinical narratives + medical images + metadata | PubMed Central case reports | Open | Multimodal: text + images. 85K de-identified case reports with demographics. 140+ image label classes. |
| **[169Pi Medical Psychology](https://huggingface.co/datasets/169Pi/medical_psychology)** | **296K examples, 260M tokens** | Sep 2025 | Structured CoT | Diagnostics, therapy plans, case studies | Curated | Open | Covers 15+ medical specialties. Structured clinical CoT reasoning. |
| **[OpenMed_MedReason RL Environment](https://huggingface.co/OpenMed)** | **32K+** | Mar 2026 | Step-by-step reasoning | RL training environment | OpenMed | Apache 2.0 | **Reinforcement Learning** environment for medical reasoning. Not just SFT — supports GRPO/PPO training. |
| **[MedAlign](https://huggingface.co/papers/2308.14089)** | **983 instructions, 276 EHRs** | 2023 | No | Natural language instructions for EHR data | 15 clinicians, 7 specialties | Open | Clinician-curated EHR instruction dataset. Small but gold-standard quality. |

## Process Reward / Verification

| Dataset | Size | Date | Thinking | Format | Source | License | Notes |
|---------|------|------|----------|--------|--------|---------|-------|
| **[Med-PRM](https://github.com/eth-medical-ai-lab/Med-PRM)** | Training data TBD | Jun 2025 | Stepwise verified reasoning | Reasoning steps + step-level labels | Clinical guidelines + retrieval | Open | **EMNLP 2025.** Process reward model that verifies each reasoning step against guidelines. First 8B model to exceed 80% on MedQA. Up to +13.5% improvement. |

## Longevity, Aging & Biological Clocks

| Dataset / Resource | Size | Date | Type | Access | Notes |
|---|---|---|---|---|---|
| **[LongevityBench](https://www.biorxiv.org/content/10.64898/2026.01.12.698650v1.full)** | **30,193 prompts (~50M tokens)** | Jan 2026 | Benchmark: transcriptomics, epigenetics, proteomics, clinical biochemistry, genetic interventions | Open (bioRxiv) | By Insilico Medicine. 7 independent data sources. Tasks: lifespan/mortality/age prediction across species. Leaderboard at bench.insilico.com. **Perfect eval benchmark for our longevity model.** |
| **[ComputAgeBench](https://huggingface.co/datasets/computage/computage_bench)** | **66 datasets (blood DNA methylation)** | Feb 2025 | Epigenetic aging clock training + benchmark | HuggingFace | 13 published clock models benchmarked. 46 separate datasets for training new clocks. 19 clinical conditions. |
| **[HALL Database](https://pubmed.ncbi.nlm.nih.gov/37870433/)** | Comprehensive | 2023 | Human aging and longevity studies database | Open | Curated database of aging/longevity research. Gene-level annotations. |
| **[LongevityMap](https://pubmed.ncbi.nlm.nih.gov/23998809/)** | Human genetic variants | 2013+ | Genetic variants associated with longevity | Open | Maps SNPs to longevity phenotypes. Useful for genomic integration. |

## Wearable & Digital Twin Foundation Models

| Model / Dataset | Size | Date | Modality | Access | Notes |
|---|---|---|---|---|---|
| **[JETS](https://arxiv.org/abs/2507.00191)** | **3M person-days** | Jul 2025 (ICML) | 63-channel wearable (HR, SpO2, sleep, activity) | Paper (Apple/Empirical Health) | Wearable foundation model using JEPA. Detects hypertension (87%), atrial flutter (70%), ME/CFS (81%). Handles irregular multivariate time series. |
| **[GluFormer](https://github.com/Guylu/GluFormer)** | **10M+ glucose measurements, 10,812 adults** | 2025 (Nature) | Continuous glucose monitoring (CGM) | Code open, data restricted | Foundation model for CGM. Predicts diabetes, CVD mortality over 11yr follow-up. Outperforms HbA1c. 19 external cohorts, 5 countries, 8 devices. |
| **[SSL-Wearables (UK Biobank)](https://oxwearables.github.io/ssl-wearables/)** | **700,000 person-days** | 2025 | Accelerometry (wrist-worn) | Model open, data via UK Biobank | Self-supervised activity recognition FM. F1 improvement 2.5-100% over baselines. Oxford/UK Biobank. |
| **[Learning Longitudinal Health Representations](https://arxiv.org/abs/2601.12227)** | Research paper | Jan 2026 | EHR + Wearable (multimodal) | Paper | **Directly relevant to LHM.** Multimodal FM jointly representing EHR + wearable as continuous-time latent process. Outperforms MOTOR, CEHR-BERT, PaPaGei, HiMAE. |
| **[SleepFM](https://arxiv.org/abs/2405.17766)** | Large-scale sleep data | 2024 | EEG, EOG, EMG (polysomnography) | Paper | Predicts 130+ diseases from sleep data. Multimodal contrastive learning. |
| **[DT-GPT](https://arxiv.org/abs/2410.11011)** | EHR text | 2024 | EHR-to-text LLM | Paper | Converts structured EHR to natural language for LLM training. |

## Precision Medicine & Multimodal

| Dataset / Resource | Size | Date | Type | Access | Notes |
|---|---|---|---|---|---|
| **[MedTrinity-25M](https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M)** | **25M images, 10 modalities, 65+ diseases** | 2025 (ICLR) | Medical images + multigranular annotations | HuggingFace | Largest multimodal medical dataset. Covers radiology, pathology, dermatology, ophthalmology. |
| **[Integrating Genomics into EHR FMs](https://arxiv.org/abs/2510.23639)** | Research paper | 2025 | Genomics + EHR | Paper (Verily) | Architecture for combining WGS/WES with clinical EHR in foundation models. |
| **[BioMedGraphica](https://huggingface.co/datasets/FuhaiLiAiLab/BioMedGraphica)** | Large knowledge graph | 2025 | Drug-gene-disease-protein interactions | HuggingFace | Graph AI-ready. Therapeutic target discovery. Precision medicine. |
| **[HPA10M](https://huggingface.co/datasets/nirschl-lab/hpa10m)** | **10.5M images** | 2025 | Immunohistochemistry (protein atlas) | HuggingFace | Human Protein Atlas pathology images. Protein localization. |

## Regenerative Medicine & Peptides

| Resource | Size | Date | Type | Access | Notes |
|---|---|---|---|---|---|
| **[PubChem (2025)](https://huggingface.co/datasets/molssiai-hub/pubchem-04-18-2025)** | Millions of compounds | Apr 2025 | Small molecules, peptides, lipids, nucleotides | HuggingFace | Includes peptide structures and bioactivity data. |
| **[AI for MSC Therapies](https://pmc.ncbi.nlm.nih.gov/articles/PMC12729526/)** | Review/methods | 2025 | Stem cell differentiation prediction | Papers | ML predicts MSC differentiation, immunomodulatory function using multi-omics + imaging. |
| **[Stem Cell Foundation Model](https://phys.org/news/2026-02-ai-foundation-aims-stem-cell.html)** | Foundation model | Feb 2026 | Cell development prediction | Paper | AI FM to make stem cell therapies more predictable. Analyzes large experimental datasets. |
| **Custom GPT-5.4 Distillation** | Scalable | 2026 | Regenerative/peptide Q&A | Self-generated | Distill GPT-5.4 reasoning on peptide therapy, stem cells, growth factors, exosomes. Our pipeline already works (95% accuracy on medical QA). |

## Preventive Medicine & Lifestyle

| Dataset / Resource | Size | Date | Type | Access | Notes |
|---|---|---|---|---|---|
| **[NHANES 2021-2023](https://www.cdc.gov/nchs/nhanes/)** | **11,933 participants** | 2023 | Demographics, labs, lifestyle, diet, examination | Public | **Already downloaded.** Comprehensive US population health survey. |
| **[BRFSS (CDC)](https://catalog.data.gov/dataset/nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system)** | 400K+ annual | Annual | Nutrition, physical activity, obesity | Public | Behavioral Risk Factor Surveillance System. State-level US health data. |
| **[chibbss/fitness-chat](https://huggingface.co/datasets/chibbss/fitness-chat-prompt-completion-dataset)** | Chat dataset | 2025 | Fitness Q&A | HuggingFace | Fitness/exercise instruction tuning data. |
| **[Digital Twin for T2D](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1710829/full)** | Framework | 2026 | Lifestyle → diabetes onset prediction | Paper | Uses retrospective lifestyle, behavioral, psychosocial data. Directly relevant to LHM preventive approach. |

## Large-Scale Clinical Cohorts (Application Required)

| Dataset | Size | Modalities | Access | Notes |
|---|---|---|---|---|
| **[UK Biobank](https://www.ukbiobank.ac.uk/)** | **500K participants** | Genomics (WGS), imaging, wearables (accelerometry), EHR, biomarkers, lifestyle | Application required | **Gold standard.** 700K person-days accelerometry. WGS for 500K. Imaging (brain, cardiac, abdominal). Linked to NHS health records. |
| **[All of Us (NIH)](https://allofus.nih.gov/)** | **800K+ enrolled** | Genomics (WGS), EHR, wearables (Fitbit), surveys, biosamples | Application required | Most diverse US cohort. Fitbit data for 30K+. WGS for 245K+. Cloud platform (Researcher Workbench). |
| **[MIMIC-IV (full)](https://physionet.org/content/mimiciv/)** | **300K admissions** | EHR (vitals, labs, meds, notes, procedures) | PhysioNet DUA | Needs CITI training. Demo subset already used. |
| **[eICU](https://physionet.org/content/eicu-crd/)** | **200K stays, 139 hospitals** | ICU vitals, labs, treatments | PhysioNet DUA | Multi-center ICU data. Complements MIMIC-IV. |

## Currently Used (What We Have)

| Dataset | Size | Thinking | Status |
|---------|------|----------|--------|
| MedQA (USMLE) | 10,178 | No CoT | Downloaded |
| MedMCQA | 30,000 (capped) | 65% have explanations (short) | Downloaded |
| PubMedQA labeled | 1,000 | Has long_answer | Downloaded |
| PubMedQA artificial | 211,269 | Has long_answer | Downloaded, not used |
| GPT-5.4 distilled | 100 | Full CoT (<think>) | Generated, used |
| Filtered CoT subset | 2,153 | Existing <think> | Filtered from above, used |
| NHANES 2021-2023 | 11,933 | N/A (tabular) | Downloaded |

---

## Strategy: Data Roadmap for LHM

### Phase 1: Immediate (Text Model — Medical QA)

**Goal**: Get MedQA accuracy from 36% → 60%+

1. **Scale GPT-5.4 distillation to 5,000 examples** — Our pipeline works (95% accuracy). Focus on:
   - Longevity/aging questions (create custom prompts)
   - Preventive medicine scenarios
   - Biomarker interpretation
   - Peptide/regenerative medicine Q&A
2. **Download MedReason** (32K) — KG-grounded, highest signal-to-noise
3. **Download ReasonMed** (370K) — Filter for preventive/longevity topics
4. **Use LongevityBench** (30K prompts) as evaluation benchmark

### Phase 2: Multimodal Foundation Model

**Goal**: Build the actual health digital twin architecture

1. **Complete PhysioNet DUA** → Full MIMIC-IV (300K admissions)
2. **Apply for UK Biobank** → Genomics + wearables + EHR + imaging
3. **Apply for All of Us** → Diverse population, Fitbit wearables, WGS
4. **Integrate NHANES** (already have) → Demographics + labs + lifestyle
5. **Study arxiv 2601.12227** — Their architecture (EHR + wearable continuous-time latent process) is closest to our LHM vision

### Phase 3: Domain-Specific Specialization

**Goal**: Differentiate in longevity/regenerative/precision medicine

1. **ComputAgeBench** → Train epigenetic aging clock module
2. **GluFormer approach** → Metabolic digital twin from CGM data
3. **JETS approach** → Wearable behavioral patterns → disease prediction
4. **Custom distillation** → GPT-5.4 reasoning on:
   - Peptide therapy protocols (BPC-157, GHK-Cu, thymosin alpha-1)
   - Hormone optimization (TRT, HGH, thyroid)
   - Supplement-drug interactions
   - Longevity interventions (rapamycin, metformin, NAD+)
   - Biomarker interpretation (biological age, inflammation panels)
5. **PubMed abstract mining** → Filter 37M abstracts for longevity/regen/preventive topics

### The Moat

No existing dataset covers the **intersection** of:
- Clinical EHR reasoning + wearable time series + genomics + longevity biomarkers + lifestyle interventions

**Our differentiation is building this dataset ourselves** through:
- GPT-5.4 distillation (reasoning layer)
- MIMIC-IV + UK Biobank + All of Us (clinical + genomic layer)
- Wearable FM integration (JETS/GluFormer patterns)
- Custom longevity/regenerative knowledge (our domain expertise)

This is why the model is called a "foundational" model — it needs to be trained on data that doesn't exist yet as a unified dataset.
