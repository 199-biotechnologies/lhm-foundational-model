# PRISM Mechanistic Reference — Molecular Pathways of Aging and Disease

## The 12 Hallmarks of Aging (Lopez-Otin et al., Cell 2023)

The updated framework organizes aging into three tiers of hallmarks:

### Primary Hallmarks (Causes of Damage)

**1. Genomic Instability**
- Mechanism: Accumulation of DNA lesions from endogenous ROS (8-oxoguanine), replication errors, exogenous mutagens. Defective base excision repair (BER), nucleotide excision repair (NER), and double-strand break repair (NHEJ, HR) with age.
- Key molecules: p53, ATM/ATR kinases, PARP1 (consumes NAD+ for repair), BRCA1/2, Ku70/80
- Clinical link: Somatic mutations → clonal expansion (CHIP), cancer predisposition, tissue dysfunction
- Biomarker proxy: Micronuclei frequency, γH2AX foci, somatic mutation burden

**2. Telomere Attrition**
- Mechanism: Replicative shortening (50-200 bp/division) due to end-replication problem. Shelterin complex (TRF1, TRF2, POT1, TIN2, TPP1, RAP1) protects chromosome ends. When critically short → uncapped telomeres activate DDR → replicative senescence or apoptosis.
- Key molecules: Telomerase (TERT + TERC), shelterin complex, CST complex
- Clinical link: Telomere length correlates with biological age but has high interindividual variability. Not a reliable standalone biomarker. Short telomeres → stem cell exhaustion, immune senescence, pulmonary fibrosis.
- Biomarker: Leukocyte telomere length (LTL) — context-dependent, not actionable in isolation

**3. Epigenetic Alterations**
- Mechanism: Age-associated DNA methylation drift (global hypomethylation + focal CpG island hypermethylation), histone modification changes (loss of H3K9me3, H4K20me3 heterochromatin marks), chromatin remodeling (loss of nuclear lamina integrity via lamin A/progerin), deregulated non-coding RNAs.
- Key molecules: DNMTs (maintenance methylation), TET enzymes (active demethylation via 5-hmC, require α-ketoglutarate), sirtuins (SIRT1/6 — NAD+-dependent histone deacetylases), polycomb/trithorax complexes
- Clinical link: Epigenetic clocks (Horvath, GrimAge, DunedinPACE) measure methylation patterns that track aging. Partial reprogramming (Yamanaka factors OSKM) can reset epigenetic age without dedifferentiation.
- Mechanistic insight: α-ketoglutarate is a TET enzyme cofactor — its decline with age impairs active DNA demethylation, contributing to epigenetic drift. This is why AKG supplementation shows lifespan extension in model organisms.

**4. Loss of Proteostasis**
- Mechanism: Decline in chaperone capacity (HSP70, HSP90), impaired ubiquitin-proteasome system (UPS), reduced autophagy-lysosomal pathway efficiency. Accumulation of misfolded/aggregated proteins → proteotoxic stress.
- Key molecules: HSF1 (heat shock response master regulator), LAMP2A (CMA receptor), p62/SQSTM1 (autophagy adaptor), mTORC1 (autophagy suppressor), TFEB (lysosomal biogenesis TF)
- Clinical link: Protein aggregation diseases (Alzheimer's Aβ/tau, Parkinson's α-synuclein, ALS TDP-43). Rapamycin enhances autophagy via mTORC1 inhibition → TFEB nuclear translocation → lysosomal gene activation.

### Antagonistic Hallmarks (Responses to Damage — Beneficial When Young, Harmful When Chronic)

**5. Disabled Macroautophagy**
- Mechanism: Three forms — macroautophagy (bulk), chaperone-mediated autophagy (CMA, selective), microautophagy. Initiation: ULK1 complex (activated by AMPK, inhibited by mTORC1) → Beclin-1/VPS34 nucleation → LC3 lipidation → autophagosome formation → lysosomal fusion.
- Key molecules: ULK1, Beclin-1, ATG5/7/12, LC3-II, p62, TFEB
- Mechanistic insight: mTORC1 phosphorylates ULK1 and TFEB to SUPPRESS autophagy. When nutrients are abundant → mTORC1 active → autophagy off → damaged proteins/organelles accumulate. Rapamycin inhibits mTORC1 → derepresses ULK1 and TFEB → autophagy ON.
- Drug connections: Rapamycin (mTORC1 → autophagy), spermidine (eIF5A hypusination → TFEB activation), metformin (AMPK → ULK1 activation)

**6. Deregulated Nutrient Sensing**
- Mechanism: Four interconnected nutrient-sensing pathways constitute the aging network:
  1. **mTOR pathway**: Senses amino acids + growth factors → activates S6K1 → protein synthesis, suppresses autophagy. Overactivation accelerates aging.
  2. **AMPK pathway**: Senses low energy (high AMP:ATP ratio) → activates catabolic programs, inhibits mTORC1 via TSC2 phosphorylation, activates ULK1 for autophagy. Protective.
  3. **Sirtuin pathway**: NAD+-dependent deacetylases (SIRT1-7). SIRT1 deacetylates PGC-1α (mitochondrial biogenesis), FOXO (stress resistance), p53. SIRT3 regulates mitochondrial function. SIRT6 maintains genomic stability. All decline with age as NAD+ falls.
  4. **Insulin/IGF-1 signaling (IIS)**: PI3K → AKT → mTORC1 activation. Reduced IIS extends lifespan across species. But too low IGF-1 → sarcopenia/frailty.
- Mechanistic insight: These four pathways form a seesaw — growth/anabolism (mTOR, IIS high) vs maintenance/repair (AMPK, sirtuins high). Chronic caloric excess tips toward growth → accelerated aging. CR, rapamycin, metformin tip toward maintenance.
- Drug connections: Rapamycin (mTORC1), metformin (AMPK/complex I), acarbose (postprandial glucose → reduced IIS), NAD+ precursors (sirtuin cofactor)

**7. Mitochondrial Dysfunction**
- Mechanism: Age-related decline in ETC complex activity (especially complex I and IV), increased ROS generation from complex I and III, mtDNA mutations (lacks histones, limited repair), impaired mitophagy (PINK1/Parkin pathway), reduced PGC-1α expression → decreased mitochondrial biogenesis.
- Key molecules: PGC-1α (master biogenesis regulator, activated by AMPK/SIRT1), PINK1/Parkin (mitophagy), UCP2/3 (uncoupling), SIRT3 (mitochondrial deacetylase), NAD+ (ETC cofactor + sirtuin substrate)
- NAD+ decline mechanism: CD38 (NADase) expression increases with age and inflammation → consumes NAD+ → depletes sirtuin/PARP substrate. PARP1 activation by DNA damage also consumes NAD+. Competition: PARP1 has higher affinity for NAD+ than SIRT1 → under DNA damage, repair wins over longevity programs.
- Metabolic consequence: Reduced oxidative phosphorylation → Warburg-like shift to glycolysis → lactate accumulation → pseudohypoxic state → HIF-1α stabilization → further suppression of mitochondrial genes.
- Drug connections: NAD+ precursors (NMN/NR), CoQ10 (ETC electron carrier), urolithin A (mitophagy inducer), SGLT2i (induce mild ketogenesis → ketone bodies as alternative fuel)

**8. Cellular Senescence**
- Mechanism: Permanent cell cycle arrest via p53/p21 or p16INK4a/Rb pathways in response to telomere dysfunction, DNA damage, oncogene activation, oxidative stress. Senescent cells resist apoptosis (upregulate BCL-2/BCL-xL anti-apoptotic proteins).
- SASP (Senescence-Associated Secretory Phenotype): Senescent cells secrete IL-6, IL-8, IL-1β, MCP-1, MMPs, VEGF, TGF-β via NF-κB and C/EBPβ transcription. SASP is paracrine-toxic: spreads senescence to neighbors, recruits immune cells, remodels ECM, promotes fibrosis, fuels chronic inflammation.
- Key molecules: p16INK4a (CDK4/6 inhibitor), p21 (CDK2 inhibitor), NF-κB (SASP driver), cGAS-STING (cytoplasmic DNA sensing → type I IFN → SASP amplification), BCL-2 family (apoptosis resistance)
- Senolytic mechanism: D+Q — Dasatinib inhibits tyrosine kinases (SRC, ephrin) in senescent cell pro-survival networks; Quercetin inhibits PI3K/AKT/BCL-2. Together they overcome the apoptosis resistance of senescent cells while sparing normal cells.
- SGLT2i senolytic mechanism (Nature 2025): Canagliflozin increases AMPK-activating metabolites → downregulates PD-L1 on senescent cells → restores immune surveillance (T cells can now recognize and eliminate senescent cells). This is immunosenolysis, not direct drug-mediated killing.
- Drug connections: D+Q/Fisetin (senolytics), rapamycin (senomorphic — suppresses SASP via mTORC1 inhibition without killing cells), SGLT2i (immunosenolysis via PD-L1)

### Integrative Hallmarks (Functional Consequences)

**9. Stem Cell Exhaustion**
- Mechanism: Reduced self-renewal capacity, lineage skewing (HSCs shift toward myeloid at expense of lymphoid → immunosenescence), niche deterioration, increased quiescence. Clonal dominance in hematopoiesis (CHIP).
- Key molecules: Wnt/β-catenin (stemness), Notch, Hedgehog, p16INK4a (progressive expression silences stem cells)
- Clinical link: CHIP — age-related somatic mutations (DNMT3A, TET2, ASXL1, JAK2) in HSCs → clonal expansion. CHIP is an independent CVD risk factor (2-4x increased MI/stroke risk). TET2 loss-of-function → increased IL-1β/IL-6 from macrophages → accelerated atherosclerosis.

**10. Altered Intercellular Communication**
- Mechanism: Inflammaging — chronic, sterile, low-grade inflammation driven by SASP, DAMPs, gut permeability, visceral adiposity.
  - NF-κB pathway: Master inflammatory TF. Activated by TLR signaling, cytokines, ROS. Drives IL-6, TNF-α, IL-1β, iNOS.
  - NLRP3 inflammasome: Senses DAMPs (mitochondrial DNA, urate crystals, cholesterol crystals, Aβ aggregates, ATP) → assembles ASC speck → activates caspase-1 → cleaves pro-IL-1β and pro-IL-18 to active forms → pyroptotic inflammation.
  - cGAS-STING: Cytoplasmic DNA sensor. Detects leaked mtDNA and micronuclei → type I interferon response → chronic immune activation.
- Gut-inflammation axis: Age-related dysbiosis → reduced butyrate producers (Faecalibacterium, Roseburia) → impaired tight junction integrity → LPS translocation → TLR4/NF-κB → systemic inflammation. TMAO (from choline/carnitine via gut bacteria) → promotes foam cell formation and thrombosis.
- Drug connections: LDN (TLR4 antagonism), colchicine (NLRP3 inhibition), anti-IL-1β (canakinumab — CANTOS trial showed CVD benefit), telmisartan (NF-κB suppression via PPAR-γ)

**11. Chronic Inflammation (Inflammaging)**
- Now recognized as a standalone hallmark. See altered intercellular communication above. Key distinguishing feature: unlike acute inflammation, inflammaging is persistent, low-grade, and systemic.
- Biomarker signature: hsCRP >1.0, IL-6 >1.5 pg/mL, NLR >2.0, elevated fibrinogen, low albumin

**12. Dysbiosis**
- Now recognized as a standalone hallmark. Age-associated changes: decreased diversity, reduced Bifidobacterium/Faecalibacterium, increased Proteobacteria/Enterobacteriaceae. Short-chain fatty acids (butyrate, propionate, acetate) decline → impaired colonocyte energy → barrier dysfunction.
- Mechanistic links: Microbiome → metabolite production (butyrate = HDAC inhibitor → epigenetic regulation; TMAO → cardiovascular risk; secondary bile acids → FXR signaling → metabolic regulation; indole derivatives → AhR activation → immune regulation)

---

## Mechanistic Drug Connections

### mTORC1 Inhibition Network (Rapamycin)
```
Rapamycin → FKBP12 binding → mTORC1 inhibition
  ├── ↓ S6K1 → ↓ protein synthesis → ↓ proteotoxic stress
  ├── ↓ 4E-BP1 phosphorylation → ↓ cap-dependent translation
  ├── ↑ ULK1 (dephosphorylation) → ↑ autophagy initiation
  ├── ↑ TFEB nuclear translocation → ↑ lysosomal biogenesis
  ├── ↓ HIF-1α → ↓ glycolytic shift
  ├── ↓ SASP (via ↓ NF-κB/IL-6 axis) — senomorphic effect
  └── ↑ stem cell function (HSC, ISC self-renewal)

Chronic/daily dosing risk: mTORC2 disruption
  ├── ↓ AKT phosphorylation → insulin resistance
  ├── ↓ SGK1 → renal sodium handling issues
  └── ↓ PKCα → cytoskeletal organization defects
Solution: Pulsed/weekly dosing preserves mTORC2 integrity
```

### AMPK Activation Network (Metformin, Exercise, SGLT2i)
```
AMPK activation (↑AMP:ATP) →
  ├── ↓ mTORC1 (via TSC2 phosphorylation) → ↑ autophagy
  ├── ↑ ULK1 (direct phosphorylation) → ↑ autophagy
  ├── ↑ SIRT1 activity (↑ NAD+/NADH ratio) → ↑ PGC-1α
  ├── ↑ PGC-1α → ↑ mitochondrial biogenesis
  ├── ↑ FOXO → ↑ stress resistance genes, ↑ autophagy genes
  ├── ↓ SREBP1c → ↓ lipogenesis
  ├── ↑ ACC phosphorylation → ↑ fatty acid oxidation
  └── ↓ NF-κB → ↓ inflammation

Metformin specifically:
  ├── Complex I inhibition (mild) → ↑ AMP:ATP → AMPK
  ├── ↓ hepatic gluconeogenesis (AMPK-independent: ↓ G6Pase)
  ├── Caution: blunts exercise-induced ROS signaling
  │   (Konopka 2019: ↓ VO2 max adaptation)
  └── ↓ Complex I → ↓ electron leakage → ↓ mitochondrial ROS
```

### GLP-1R Agonism Network (Semaglutide, Tirzepatide)
```
GLP-1R activation →
  ├── Hypothalamic GLP-1R → ↓ appetite, ↑ satiety
  ├── Pancreatic β-cell → ↑ insulin secretion (glucose-dependent)
  ├── Anti-inflammatory:
  │   ├── ↓ NF-κB activation in macrophages
  │   ├── ↓ NLRP3 inflammasome activation
  │   ├── ↑ IL-10 (anti-inflammatory)
  │   └── ↓ monocyte migration to atherosclerotic plaques
  ├── Neuroprotective:
  │   ├── ↓ neuroinflammation (microglial NF-κB)
  │   ├── ↑ BDNF expression
  │   ├── ↓ Aβ accumulation (enhanced clearance)
  │   └── ↑ neuronal insulin signaling
  ├── Cardiovascular:
  │   ├── ↓ atherosclerotic plaque inflammation
  │   ├── ↓ endothelial dysfunction
  │   └── ↓ platelet activation
  └── Multi-omic aging counteraction (Cell Metabolism 2025):
      GLP-1R agonism closely resembles mTOR inhibition at
      the molecular level — overlapping transcriptomic signatures
      in liver, adipose, brain. Dependent on hypothalamic GLP-1R.
```

### SGLT2 Inhibition Network (Empagliflozin, Canagliflozin)
```
SGLT2 inhibition →
  ├── ↓ Glucose reabsorption → glycosuria → ↓ plasma glucose
  ├── Metabolic fuel shift:
  │   ├── Mild ketogenesis → ↑ β-hydroxybutyrate
  │   │   ├── HDAC inhibitor → epigenetic modulation
  │   │   ├── Alternative cardiac fuel → ↑ cardiac efficiency
  │   │   └── NLRP3 inflammasome inhibitor
  │   └── ↑ fatty acid oxidation → ↓ lipotoxicity
  ├── AMPK activation (via ↑ AMP:ATP from glucose loss)
  │   ├── ↑ autophagy/mitophagy
  │   ├── ↓ mTORC1
  │   └── ↑ SIRT1 → ↑ PGC-1α
  ├── Senolytic (Nature 2025):
  │   ├── ↑ AMPK-activating metabolites
  │   ├── ↓ PD-L1 on senescent cells
  │   └── ↑ T-cell immunosurveillance → senescent cell clearance
  ├── Cardioprotection:
  │   ├── ↓ preload/afterload (natriuresis)
  │   ├── ↓ sympathetic tone
  │   ├── ↓ uric acid → ↓ oxidative stress
  │   └── Direct cardiomyocyte mitochondrial protection
  └── Renal:
      ├── Restored tubuloglomerular feedback
      ├── ↓ glomerular hyperfiltration
      └── ↓ tubulointerstitial fibrosis
```

### NAD+ Metabolism and Decline
```
NAD+ homeostasis:
  Synthesis:
    ├── De novo: Tryptophan → QPRT → NAD+ (liver-dominant)
    ├── Salvage: Nicotinamide → NAMPT (rate-limiting) → NMN → NMNAT → NAD+
    ├── NR → NRK → NMN → NMNAT → NAD+ (Preiss-Handler variant)
    └── NMN (oral) → NMN → NMNAT → NAD+ (direct supplementation)

  Consumption (three competing demands):
    ├── SIRT1-7 (deacetylases) → gene regulation, stress response
    │   Produce nicotinamide as byproduct (product inhibition)
    ├── PARPs (ADP-ribosyltransferases) → DNA repair
    │   PARP1 has HIGHER affinity for NAD+ than SIRT1
    │   Under DNA damage: repair > longevity programs
    └── CD38/CD157 (NADases) → immune signaling
        CD38 expression INCREASES with age/inflammation
        → primary driver of age-related NAD+ decline
        → each CD38 molecule degrades ~100 NAD+ molecules

  Age-related decline: ~50% reduction by age 50-60
  Consequence: SIRT1 activity ↓ → PGC-1α ↓ → mitochondrial dysfunction
               SIRT3 activity ↓ → mitochondrial protein hyperacetylation
               SIRT6 activity ↓ → genomic instability
               PARP1 activity ↓ → impaired DNA repair
```

### Insulin Resistance Molecular Cascade
```
Progression (mechanistic):
  1. Caloric excess → ↑ hepatic lipogenesis → ↑ DAG + ceramides
  2. DAG activates PKCε (liver) / PKCθ (muscle) →
     ├── ↓ IRS-1/2 tyrosine phosphorylation
     ├── ↑ IRS-1 serine phosphorylation (inhibitory)
     └── ↓ PI3K/AKT signaling → ↓ GLUT4 translocation
  3. Ceramides independently:
     ├── Activate PP2A → ↓ AKT phosphorylation
     ├── Activate PKCζ → ↓ AKT membrane recruitment
     └── ↓ mitochondrial ETC complex activity → ↑ ROS
  4. Compensatory hyperinsulinemia:
     ├── ↑ insulin → ↑ mTORC1 → ↓ autophagy
     ├── ↑ insulin → ↑ lipogenesis (via SREBP1c) → more DAG/ceramide
     └── Positive feedback loop → progressive β-cell exhaustion
  5. Hepatic steatosis → NASH progression:
     ├── ↑ ALT/GGT = early signal
     ├── ↑ Oxidative stress (CYP2E1 induction)
     ├── ↑ NF-κB/NLRP3 → hepatic inflammation
     └── Stellate cell activation → fibrosis

Biomarker mapping to mechanism:
  - Fasting insulin >5 = compensatory hyperinsulinemia (stage 1)
  - TG/HDL >2.0 = hepatic lipogenesis overflow → VLDL overproduction
  - ALT >20 + GGT >20 = hepatocyte lipotoxicity + oxidative stress
  - Uric acid >5.5 = fructose metabolism → purine degradation
  - HbA1c >5.3% = glycemic excursions exceeding β-cell compensation
```

### Atherosclerosis Molecular Mechanism
```
ApoB-mediated atherogenesis (Response-to-Retention model):
  1. ApoB-containing particles (LDL, VLDL remnants, Lp(a)) cross
     endothelium at sites of disturbed shear stress
  2. Subendothelial retention via ApoB-proteoglycan binding
     (electrostatic interaction, ApoB site 3359-3369)
  3. Retained particles undergo oxidative/enzymatic modification:
     ├── OxLDL → recognized by scavenger receptors (SR-A, CD36, LOX-1)
     ├── Macrophage foam cell formation (unregulated uptake)
     └── OxLDL triggers endothelial activation → VCAM-1, MCP-1
  4. Inflammatory amplification:
     ├── Cholesterol crystals → NLRP3 inflammasome → IL-1β
     ├── Foam cell death → necrotic core formation
     ├── Smooth muscle cell migration → fibrous cap
     └── MMP secretion → cap thinning → plaque vulnerability
  5. Lp(a)-specific mechanism:
     ├── Contains OxPL (oxidized phospholipids) → TLR2 activation
     ├── Homology to plasminogen → anti-fibrinolytic → thrombotic risk
     └── Genetically determined (LPA gene KIV2 repeats)

Why ApoB > LDL-C:
  - ApoB = 1 molecule per atherogenic particle (LDL, VLDL remnant, IDL, Lp(a))
  - LDL-C = cholesterol mass, discordant with particle number in 30% of patients
  - When TG high → LDL is cholesterol-depleted but small/dense → more particles
    per unit of cholesterol → LDL-C underestimates risk, ApoB captures it
  - Mendelian randomization (2024-2025): causal relationship confirmed for
    ApoB-containing particles independent of cholesterol content
```

---

## CHIP (Clonal Hematopoiesis of Indeterminate Potential)

An emerging hallmark-adjacent risk factor connecting aging and cardiovascular disease:

```
Definition: Somatic mutations in hematopoietic stem cells (HSCs) leading to
clonal expansion without hematologic malignancy. Prevalence: <1% at age 40,
10-20% by age 70, >30% by age 80.

Common mutations: DNMT3A (60%), TET2 (20%), ASXL1, JAK2, TP53, PPM1D

Cardiovascular mechanism (TET2 loss-of-function as model):
  TET2 loss → ↓ active DNA demethylation in macrophages →
  ├── ↑ NLRP3 inflammasome activation
  ├── ↑ IL-1β and IL-6 production
  ├── Enhanced macrophage infiltration into atherosclerotic plaques
  └── Accelerated plaque progression

Risk: 2-4x increased MI/stroke risk (comparable to traditional risk factors)
      Also increases HF risk, aortic stenosis progression

Emerging interventions:
  - Anti-IL-1β therapy (CANTOS extrapolation)
  - Colchicine (NLRP3 inhibition) — COLCOT/LoDoCo2 show CVD benefit
  - Lifestyle: exercise reduces clonal expansion in some models
  - Screening: Not yet standard, but VAF >2% warrants monitoring

AHA Scientific Statement 2024: Recognizes CHIP as cardiovascular risk factor
```

---

## Gut-Aging-Disease Axis

```
Age-related microbiome changes:
  ├── ↓ Diversity (Shannon index)
  ├── ↓ Butyrate producers (Faecalibacterium prausnitzii, Roseburia)
  ├── ↑ Proteobacteria, Enterobacteriaceae (pathobionts)
  ├── ↓ Akkermansia muciniphila (mucin layer integrity)
  └── ↑ Bacteroides/Firmicutes ratio shift

Mechanistic consequences:
  1. ↓ Butyrate production →
     ├── ↓ HDAC inhibition → epigenetic deregulation in colonocytes
     ├── ↓ Treg differentiation → immune dysregulation
     ├── ↓ Tight junction proteins (claudin, occludin) → ↑ permeability
     └── ↓ Colonocyte energy (butyrate is primary fuel) → barrier failure

  2. ↑ Intestinal permeability ("leaky gut") →
     ├── LPS translocation → TLR4/NF-κB activation → systemic inflammation
     ├── Bacterial DNA → cGAS-STING → type I IFN
     └── Contributes to inflammaging independent of senescent cells

  3. ↑ TMAO (trimethylamine N-oxide) →
     ├── Promotes macrophage foam cell formation
     ├── Enhances platelet hyperreactivity → thrombosis
     ├── Activates endothelial NF-κB → ↑ VCAM-1
     └── Source: dietary choline/carnitine → TMA by gut bacteria → FMO3 → TMAO

  4. ↓ Secondary bile acid diversity →
     ├── ↓ FXR activation → metabolic deregulation
     ├── ↓ TGR5 activation → ↓ GLP-1 secretion
     └── Gut-liver axis dysfunction

Biomarker proxies: hsCRP, LPS-binding protein, zonulin, TMAO (research-grade)
```

---

## Epigenetic Reprogramming (Emerging Tier C)

```
Yamanaka factors (Oct4, Sox2, Klf4, c-Myc — OSKM):
  Full reprogramming → iPSCs (dedifferentiation, cancer risk)
  Partial/cyclic reprogramming → age reversal WITHOUT dedifferentiation
    ├── Resets DNA methylation clocks (Horvath, Skin&Blood)
    ├── Restores H3K9me3 heterochromatin marks
    ├── Does NOT elongate telomeres (separate mechanism)
    └── Improves tissue function in aged mice (neurons, muscle, liver)

Chemical reprogramming (avoid viral delivery of TFs):
  Multi-omics 2024 (eLife): Cocktails of small molecules can induce
  partial reprogramming without transgenes
    ├── Valproic acid (HDAC inhibitor)
    ├── CHIR99021 (GSK-3β inhibitor)
    ├── Repsox (TGF-β inhibitor)
    ├── Tranylcypromine (LSD1 inhibitor)
    └── DZNep (EZH2 inhibitor)

Current status (updated March 2026):
  Life Biosciences — FDA-cleared first human partial reprogramming trial
  (Jan 2026). Targets age-related macular degeneration using in vivo
  OSK (no c-Myc) gene therapy via AAV delivery to retinal cells.
  First trial to test epigenetic reprogramming in humans.

  Other players: Altos Labs, Calico, Turn Biotechnologies, Retro Bio.
  Main risk: uncontrolled reprogramming → teratoma formation.
  Clinical translation for systemic applications: 5-10+ years.
  Tissue-specific applications (eye, liver): 2-5 years.
```

---

## Lipotoxicity and Ceramide Signaling

```
Ceramide synthesis pathways:
  1. De novo: Palmitoyl-CoA + serine → SPT → dihydroceramide → ceramide
     (↑ by saturated fat intake, ↑ by TNF-α/IL-6, ↑ by ER stress)
  2. Sphingomyelinase: Sphingomyelin → ceramide (stress-responsive)
  3. Salvage: Sphingosine → ceramide kinase

Ceramide toxicity mechanisms:
  ├── Mitochondria: ↓ Complex III activity, opens mPTP → apoptosis
  ├── ER stress: ↑ CHOP, ↑ IRE1α → UPR activation
  ├── Insulin signaling: Activates PP2A → dephosphorylates AKT
  ├── Inflammation: Activates NLRP3 inflammasome
  └── Cell death: Pro-apoptotic (↑ Bax, ↓ Bcl-2)

Clinical measurement: Ceramide risk score (Mayo Clinic) — predicts
cardiovascular events independent of LDL-C. Uses C16:0, C18:0, C24:1
ceramide ratios.

Emerging biomarker: Plasma ceramide panel may outperform LDL-C for
CVD risk stratification in patients with metabolic syndrome [Tier C]
```
