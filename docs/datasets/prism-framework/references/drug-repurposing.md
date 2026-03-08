# Drug Repurposing for Longevity — Evidence Summary

## ITP (Interventions Testing Program) Results Summary

The NIA-funded ITP is the gold standard for preclinical longevity drug testing — multi-site, adequately powered, pre-registered endpoints, standardized protocols across 3 labs.

| Compound | Lifespan Effect | Sex | Evidence Level | Human Translation |
|----------|----------------|-----|----------------|-------------------|
| Rapamycin | +23% median (M), +26% (F) | Both | ITP confirmed | PEARL trial (human immune aging) |
| Acarbose | +22% median (M), +5% (F) | Mainly M | ITP confirmed | Standard T2DM drug, off-label use |
| 17α-Estradiol | +19% median | M only | ITP confirmed | No feminizing effects. No human data |
| Canagliflozin | +14% median | M only | ITP confirmed | SGLT2i, approved for T2DM/CKD/HF |
| Metformin | Modest (Gehan re-analysis positive) | M mainly | ITP (reanalyzed) | TAME trial pending |
| Enalapril | Modest (Gehan positive) | M mainly | ITP (reanalyzed) | ACEi, standard hypertension |
| Aspirin | Modest lifespan extension | M only | ITP | ASPREE: harmful in healthy >70. Complicated |
| Rapamycin + Trametinib | Additive (>rapa alone) | Both | Nature Aging 2025 | Both are approved drugs (oncology) |
| Protandim | Positive | M only | ITP | Nrf2 activator supplement |

**Failed / Negative ITP Results**: Resveratrol, curcumin, green tea extract, fish oil, nicotinamide riboside, oxaloacetate, medium-chain triglycerides

## Drug-by-Drug Evidence Profiles

### Rapamycin (Sirolimus)
```
Molecular mechanism:
  Rapamycin binds FKBP12 (FK506-binding protein) → complex allosterically inhibits
  mTOR complex 1 (mTORC1). mTORC1 is a serine/threonine kinase that integrates
  amino acid, growth factor, and energy signals to promote anabolism.

  mTORC1 inhibition cascade:
    ├── ↓ S6K1 phosphorylation → ↓ ribosomal protein synthesis → ↓ proteotoxic burden
    ├── ↓ 4E-BP1 phosphorylation → ↓ cap-dependent translation (growth genes, SASP)
    ├── ULK1 dephosphorylation → autophagy initiation
    ├── TFEB nuclear translocation (no longer sequestered by mTORC1) →
    │   ↑ CLEAR gene network → lysosomal biogenesis → autophagosome clearance
    ├── ↓ HIF-1α stabilization → reverses age-related Warburg glycolytic shift
    ├── ↓ NF-κB/IL-6 transcription → senomorphic SASP suppression
    ├── ↑ hematopoietic stem cell self-renewal (ISC, HSC function restored)
    └── Immune rejuvenation: ↑ naive T-cell generation, ↑ vaccine response

  mTORC2 distinction (critical for dosing):
    mTORC2 (contains Rictor, not Raptor) is NOT directly inhibited by rapamycin
    acutely. However, chronic daily exposure → progressive mTORC2 disassembly →
    ├── ↓ AKT Ser473 phosphorylation → insulin resistance
    ├── ↓ SGK1 → sodium handling disruption
    └── ↓ PKCα → cytoskeletal defects
    Solution: Pulsed weekly dosing → mTORC1 inhibited during peak,
    mTORC2 recovers between doses.

  Combination: Rapamycin + Trametinib (MEK inhibitor, ↓ Ras/MAPK/ERK)
    Nature Aging 2025: Nearly completely additive lifespan extension.
    Orthogonal mechanisms suggest mTOR and MAPK/ERK are independent aging axes.

Hallmarks addressed: 4 (proteostasis), 5 (autophagy), 6 (nutrient sensing),
  8 (SASP suppression), 9 (stem cell function)

ITP: Up to 23% (M), 26% (F) lifespan extension (strongest single-drug effect)
Human evidence:
  - Mannick 2014: Low-dose mTOR inhibitor improved vaccine response in elderly
  - Mannick 2018: mTOR inhibitor reduced infections in elderly by 40%
  - PMC12794675 (2025): Enhanced DNA damage resilience in human immune cells
  - PEARL trial: ongoing (human aging)
  - Dog Aging Project: rapamycin in companion dogs
Dosing (off-label): 3-6mg once weekly (cyclic to avoid mTORC2 inhibition)
Monitoring: Lipids (↑TG via ↓lipoprotein lipase, transient), HbA1c, CBC, mouth ulcers
Risks: Immunosuppression if dosed daily/continuously, insulin resistance via mTORC2
Evidence tier: B (strong animal, emerging human)
```

### Metformin
```
Molecular mechanism:
  Primary: Mild inhibition of mitochondrial complex I (NADH:ubiquinone
  oxidoreductase) → ↑ AMP:ATP ratio → AMPK activation

  AMPK activation cascade:
    ├── TSC2 phosphorylation → ↓ Rheb → ↓ mTORC1 → ↑ autophagy
    ├── ULK1 direct phosphorylation → autophagy initiation
    ├── ↑ NAD+/NADH ratio → ↑ SIRT1 activity →
    │   ├── PGC-1α deacetylation → mitochondrial biogenesis
    │   ├── FOXO3 deacetylation → stress resistance genes
    │   └── p53 deacetylation → apoptosis regulation
    ├── ACC phosphorylation → ↑ fatty acid β-oxidation
    └── ↓ SREBP1c → ↓ de novo lipogenesis

  AMPK-independent:
    ├── ↓ Hepatic gluconeogenesis (↓ G6Pase, ↓ adenylyl cyclase → ↓ cAMP/PKA)
    ├── ↓ Insulin/IGF-1 signaling → ↓ mitogenic proliferation (anti-cancer)
    └── ↑ GLP-1 secretion, microbiome modulation

  Exercise interaction (Konopka 2019):
    Complex I inhibition → ↓ exercise-induced ROS signaling for PGC-1α
    adaptation → ↓ VO2 max gains in fit individuals.

Hallmarks: 5 (autophagy), 6 (nutrient sensing), 7 (mitochondrial)
ITP: Modest (positive on Gehan re-analysis, not log-rank)
Human evidence:
  - Bannister 2014: Metformin-treated T2DM patients outlived non-diabetic controls
  - Observational: 30-40% cancer risk reduction
  - TAME trial: Barzilai et al. — "positive evidence is mounting" (Aug 2025 interview); results pending
  - Exceptional longevity in women on metformin (ScienceAlert, Nov 2025) — consistent with Bannister data
Dosing: 500mg titrated to 1500-2000mg daily
Monitoring: B12 annually (↓intrinsic factor-mediated absorption), renal function
Contraindications: eGFR <30, active alcohol use disorder, hepatic failure
  DO NOT use in highly active, insulin-sensitive individuals
Evidence tier: B (strong observational, pending TAME for tier A)
```

### GLP-1 Receptor Agonists (Semaglutide, Tirzepatide)
```
Molecular mechanism:
  GLP-1R is a class B GPCR → Gs → adenylyl cyclase → ↑ cAMP → PKA/EPAC

  Multi-organ signaling:
    Pancreas: ↑ cAMP → ↑ glucose-dependent insulin secretion, ↑ β-cell survival
    Hypothalamus: GLP-1R on POMC/CART neurons → ↓ appetite, ↑ satiety
    Immune system:
      ├── ↓ NF-κB activation in macrophages (↓ TNF-α, IL-6)
      ├── ↓ NLRP3 inflammasome assembly
      ├── ↑ IL-10 (anti-inflammatory cytokine)
      └── ↓ monocyte migration to atherosclerotic plaques
    CNS neuroprotection:
      ├── ↓ Microglial NF-κB → ↓ neuroinflammation
      ├── ↑ BDNF expression → neuroplasticity
      ├── ↓ Aβ accumulation (enhanced clearance pathways)
      └── ↑ Neuronal insulin signaling (↓ central IR)
    Cardiovascular:
      ├── ↓ Plaque inflammation (macrophage NF-κB)
      ├── ↓ Endothelial dysfunction (↑ eNOS)
      └── ↓ Platelet activation

  Body-wide aging counteraction (Cell Metabolism 2025):
    GLP-1R agonism in aging mice produces multi-omic signatures that closely
    resemble mTOR inhibition at the transcriptomic level in liver, adipose,
    and brain. This effect is DEPENDENT on hypothalamic GLP-1R signaling.
    Positions GLP-1R agonists as potential geroprotectors beyond metabolic drugs.

  Tirzepatide: Dual GIP/GLP-1 receptor agonist → superior weight loss
    and glycemic control vs semaglutide (SURPASS data)

Hallmarks: 6 (nutrient sensing), 10 (intercellular communication), 11 (inflammaging)
Human evidence:
  - SELECT: 17% MACE reduction in overweight/obese with CVD
  - STEP: 15-22% weight loss
  - SURPASS: Tirzepatide superior weight loss
  - FLOW: Semaglutide renal protection
  - Emerging: AD, PD signal in observational data
CRITICAL WARNING: Sarcopenia risk. Up to 40% of weight loss can be lean mass.
  MUST mandate: protein ≥1.6g/kg/day + resistance training 3x/week
  MUST monitor: ALMI via DXA at baseline and q3-6 months
Evidence tier: A (CVD, weight), B (neurodegeneration), C (longevity/geroprotection)
```

### SGLT2 Inhibitors (Empagliflozin, Dapagliflozin, Canagliflozin)
```
Molecular mechanism:
  SGLT2 blockade in proximal tubule → glycosuria → ↓ plasma glucose

  Downstream cascade:
    Metabolic fuel shift:
      ├── ↓ Glucose availability → mild ketogenesis → ↑ β-hydroxybutyrate (BHB)
      │   ├── BHB is an HDAC inhibitor → epigenetic modulation
      │   ├── BHB inhibits NLRP3 inflammasome → ↓ IL-1β
      │   └── BHB is efficient cardiac fuel → ↑ cardiac energetics
      └── ↑ Fatty acid oxidation → ↓ lipotoxicity (↓ DAG/ceramide)

    AMPK activation (via ↑ AMP:ATP from glucose loss):
      ├── ↑ autophagy/mitophagy (ULK1, PINK1/Parkin)
      ├── ↓ mTORC1
      ├── ↑ SIRT1 → ↑ PGC-1α → mitochondrial biogenesis
      └── ↑ FOXO → stress resistance

    Senolytic (Nature Aging 2025):
      Canagliflozin → ↑ AMPK-activating metabolites →
      ├── ↓ PD-L1 expression on senescent cells
      ├── T cells no longer inhibited by PD-L1 checkpoint
      └── ↑ Immune surveillance → senescent cell clearance
      This is immunosenolysis — the immune system clears senescent cells,
      not the drug directly. Distinct mechanism from D+Q senolytics.

    Cardioprotection:
      ├── ↓ Preload/afterload (natriuresis + osmotic diuresis)
      ├── ↓ Sympathetic tone (central mechanism)
      ├── ↓ Uric acid (↑ renal excretion) → ↓ NLRP3 crystal activation
      └── Direct cardiomyocyte mitochondrial protection

    Renal:
      ├── Restored tubuloglomerular feedback (↑ NaCl to macula densa)
      ├── ↓ Glomerular hyperfiltration → ↓ mechanical stress
      └── ↓ Tubulointerstitial fibrosis

    RCT (Henagliflozin, Cell Reports Medicine 2025):
      SGLT2i directly improved aging biomarkers (epigenetic clock,
      inflammatory markers) in T2DM patients — first human RCT
      demonstrating anti-aging effects of an SGLT2i.

Hallmarks: 7 (mitochondrial), 8 (senescence/immunosenolysis),
  10 (inflammation), 11 (inflammaging)
ITP: Canagliflozin +14% lifespan in male mice
Human evidence: EMPA-REG, DAPA-HF, DAPA-CKD, CREDENCE, FLOW
Off-label longevity: Normoglycemic with HFpEF, CKD, or elevated uric acid
Monitoring: Euglycemic DKA (rare), genital mycotic infections, volume status
Evidence tier: A (cardio-renal), B (longevity/anti-aging)
```

### Statins (Rosuvastatin, Pitavastatin)
```
Mechanism: HMG-CoA reductase inhibition + pleiotropic anti-inflammatory
Human evidence: Decades of RCT data. Most evidence-based CVD prevention drug class.
Longevity context:
  - Hydrophilic (rosuvastatin, pravastatin) → less BBB penetration → lower cognitive risk
  - Pitavastatin: minimal metabolic side effects, less myopathy
  - Bempedoic acid + ezetimibe as alternative if intolerant
Monitoring: LFTs, CK (if symptomatic), CoQ10 supplementation recommended
PGx: Check SLCO1B1 if myopathy concern (especially simvastatin)
Evidence tier: A
```

### Telmisartan
```
Mechanism: ARB + partial PPAR-γ agonist (insulin sensitizing)
Human evidence: ONTARGET (equivalent to ramipril with better tolerability)
Longevity context: Premier longevity ARB due to dual BP + metabolic action
Target BP: <120/80, optimally 110/70 (longevity target)
ITP: Enalapril showed sex-dependent effects on reanalysis
Evidence tier: A (BP/cardioprotection), B (metabolic/longevity)
```

### Low-Dose Naltrexone (LDN)
```
Mechanism: Transient opioid blockade → endorphin upregulation; TLR4 antagonism
Human evidence:
  - Younger 2014: Small RCT in fibromyalgia (significant pain reduction)
  - Multiple open-label series in Hashimoto's, Crohn's, MS
Dosing: 1.5-4.5mg at night (compounding pharmacy)
Application: Autoimmune conditions, chronic fatigue, unexplained ↑hsCRP
Evidence tier: B (autoimmune/pain), C (longevity)
```

### Acarbose
```
Mechanism: Alpha-glucosidase inhibitor → blocks complex carb absorption → blunts
  postprandial glucose spikes without systemic absorption
ITP: +22% male lifespan (one of the strongest ITP results)
Human evidence: STOP-NIDDM (diabetes prevention)
Longevity context: Synergistic with rapamycin in animals. Good for patients with
  post-meal glucose spikes who can't tolerate metformin
Evidence tier: B
```

### Senolytics
```
Dasatinib + Quercetin (D+Q):
  Mechanism: Dasatinib (tyrosine kinase inhibitor) + Quercetin (flavonoid) →
    selective apoptosis of senescent cells → ↓SASP
  Evidence: Kirkland 2017 (mouse healthspan), Justice 2019 (human IPF pilot),
    AFFIRM-LITE trial
  Dosing: "Hit and run" — D 100mg + Q 1000mg for 2 consecutive days/month

Fisetin:
  Mechanism: Similar to quercetin, natural flavonoid
  Evidence: AFFIRM-LITE trial, preclinical promising
  Dosing: High-dose intermittent (20mg/kg for 2 days/month)

Evidence tier: C (human data very early)
```

### NAD+ Precursors (NMN, NR)
```
Mechanism: NAD+ repletion → sirtuin activation, PARP support, cellular energy
Human evidence:
  - Yoshino 2021: NMN improved insulin sensitivity in prediabetic women
  - Martens 2018: NR lowered blood pressure, improved aortic stiffness
  - ITP: NR FAILED to extend lifespan in mice
CRITICAL: Avoid in active cancer — NAD+ fuels both healthy and oncogenic metabolism
Dosing: NMN 250-1000mg/day or NR 300-1000mg/day
Evidence tier: C (no lifespan data, ITP negative for NR, mixed metabolic signals)
```

### Taurine
```
Mechanism: Mitochondrial function, anti-inflammatory, osmoregulation, bile acid conjugation
Evidence: Singh et al., Science 2023 — reversal of aging hallmarks in mice/monkeys
  Taurine levels decline ~80% from youth to old age across species
Human data: Limited to cardiovascular markers, exercise performance
Dosing: 3-6g/day
Evidence tier: C (strong preclinical, minimal human longevity data)
```

### 17α-Estradiol
```
Mechanism: Non-feminizing estrogen; metabolic/anti-inflammatory effects in males
ITP: +19% male lifespan. Consistent across 3 labs.
Human data: None
Notes: No feminizing effects (does not activate ERα in reproductive tissues)
Evidence tier: C (strong animal, no human translation yet)
```

### Spermidine
```
Mechanism: Autophagy inducer via eIF5A hypusination
Evidence:
  - Eisenberg 2016: Lifespan extension in yeast, worms, flies, mice
  - Observational: Dietary spermidine inversely associated with mortality
  - SmartAge trial: Modest cognitive benefit in elderly
Dosing: 1-6mg/day (supplement) or high-spermidine diet (wheat germ, aged cheese, mushrooms)
Evidence tier: C (observational + small RCT)
```
