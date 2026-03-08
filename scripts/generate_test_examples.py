#!/usr/bin/env python3
"""
Generate test PRISM skill pack examples directly (no external API needed).
Produces ~50 hand-crafted reference examples to test the pipeline and iterate on quality.
"""

import json
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path("docs/datasets/prism-packs")

def make_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def make_entry(question, reasoning, answer, metadata):
    return {
        "messages": [
            {"role": "user", "content": f"{question}\n\nAnswer:"},
            {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n{answer}"},
        ],
        "metadata": metadata,
    }

def save_pack(pack_dir, entries):
    out = BASE_DIR / pack_dir / "train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"  {pack_dir}: {len(entries)} examples → {out}")

# ──────────────────────────────────────────────────────────────────
# P1: BIOMARKER (8 examples)
# ──────────────────────────────────────────────────────────────────

P1 = []

P1.append(make_entry(
    question="""A 52-year-old male presents for a longevity-focused health assessment. The following lab results are obtained:

  Fasting Insulin: 18.2 mIU/L
  Fasting Glucose: 94 mg/dL
  HbA1c: 5.4%
  HOMA-IR: 4.2
  hsCRP: 1.8 mg/L
  ApoB: 88 mg/dL
  Uric Acid: 6.8 mg/dL

Which finding is most concerning from a longevity perspective?

A. Fasting Insulin
B. HbA1c
C. hsCRP
D. ApoB""",
    reasoning="""This panel reveals a classic Insulin Resistance Cascade constellation that standard screening would largely miss. Fasting glucose at 94 mg/dL and HbA1c at 5.4% both fall within the standard reference range (<99 and <5.7% respectively), and most clinicians would reassure the patient. However, applying dual-threshold interpretation exposes the underlying pathology.

Fasting insulin at 18.2 mIU/L is within the standard range (<25 mIU/L) but dramatically exceeds the longevity-optimal threshold of <7 mIU/L — representing compensatory hyperinsulinemia. The pancreas is producing 2.6× the optimal insulin output to maintain glucose in a nominally "normal" range. HOMA-IR at 4.2 (optimal <1.0) confirms frank insulin resistance. This pattern reflects deranged nutrient sensing, one of the primary hallmarks of aging, mediated through chronic mTOR/S6K1 hyperactivation and suppressed AMPK signaling.

The elevated uric acid at 6.8 mg/dL (optimal 4.0–5.5) reinforces the IR constellation — hyperinsulinemia reduces renal urate clearance via URAT1 transporter upregulation. The hsCRP at 1.8 mg/L (optimal <0.5) suggests early inflammaging, likely downstream of visceral adipose tissue macrophage activation driven by the insulin resistance.

ApoB at 88 mg/dL is suboptimal (optimal <70) but represents a secondary cardiovascular risk marker rather than the primary driver. The most concerning finding is fasting insulin because it represents the earliest and most actionable signal in a cascade that, left unaddressed, progresses through metabolic syndrome → type 2 diabetes → accelerated vascular aging. Intervening at the hyperinsulinemia stage (through dietary carbohydrate modification, exercise-induced GLUT4 translocation, and potential metformin or berberine) offers the widest therapeutic window.""",
    answer="A. Fasting Insulin",
    metadata={
        "pack": "P1", "pack_name": "prism-biomarker", "idx": 0,
        "content_hash": make_hash("52M insulin resistance cascade"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P1.append(make_entry(
    question="""A 45-year-old female presents for a longevity-focused health assessment. The following lab results are obtained:

  TSH: 3.8 mIU/L
  Free T4: 0.9 ng/dL
  Free T3: 2.3 pg/mL
  TPO Antibodies: 142 IU/mL
  hsCRP: 2.4 mg/L
  Vitamin D: 22 ng/mL
  Ferritin: 18 ng/mL

Which finding is most concerning from a longevity perspective?

A. TSH
B. TPO Antibodies
C. Vitamin D
D. Ferritin""",
    reasoning="""This panel reveals a Thyroid Spectrum constellation with compounding autoimmune and micronutrient depletion. TSH at 3.8 mIU/L falls within the standard range (0.4–4.0) but exceeds the longevity-optimal threshold of 0.5–2.5 mIU/L. Crucially, this mildly elevated TSH is accompanied by TPO antibodies at 142 IU/mL (optimal <9 IU/mL) — a combination that transforms an "unremarkable" TSH into a clear autoimmune thyroiditis signal.

Free T4 at 0.9 ng/dL sits at the low end of the standard range (0.8–1.8) and below the optimal target of 1.1–1.5 ng/dL. Free T3 at 2.3 pg/mL is similarly suboptimal (optimal 3.0–3.8), suggesting impaired T4→T3 peripheral conversion, which may be exacerbated by the low selenium status commonly associated with Hashimoto's. This thyroid dysfunction activates the hallmark of altered intercellular communication through chronic low-grade immune dysregulation, with TPO-positive autoimmunity conferring risk of progression to overt hypothyroidism at approximately 4.3% per year.

The hsCRP at 2.4 mg/L (optimal <0.5) reflects systemic inflammation that may be partially thyroid-autoimmune-driven. Vitamin D at 22 ng/mL is below the standard threshold (30 ng/mL) and far below optimal (40–60 ng/mL) — vitamin D deficiency has been independently associated with increased TPO antibody titers. Ferritin at 18 ng/mL indicates functional iron deficiency (optimal 30–80 for females), which impairs thyroid peroxidase enzyme activity and worsens the autoimmune cycle.

While all values warrant attention, TPO antibodies at 142 IU/mL represent the root pathological driver — a confirmed autoimmune process that is actively degrading thyroid tissue and will progressively worsen the entire constellation. The micronutrient deficiencies (vitamin D, iron) are both contributors to and consequences of the autoimmune process and should be corrected as part of a comprehensive intervention.""",
    answer="B. TPO Antibodies",
    metadata={
        "pack": "P1", "pack_name": "prism-biomarker", "idx": 1,
        "content_hash": make_hash("45F thyroid spectrum autoimmune"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P1.append(make_entry(
    question="""A 61-year-old male presents for a longevity-focused health assessment. The following lab results are obtained:

  ApoB: 112 mg/dL
  Lp(a): 78 nmol/L
  hsCRP: 3.2 mg/L
  LDL-C: 142 mg/dL
  TG/HDL ratio: 3.1
  Homocysteine: 14.2 µmol/L
  eGFR: 72 mL/min

Which finding is most concerning from a longevity perspective?

A. ApoB
B. Lp(a)
C. Homocysteine
D. eGFR""",
    reasoning="""This panel reveals a high-risk Atherogenic Risk Enhancement constellation. ApoB at 112 mg/dL exceeds both the standard primary prevention threshold (<130) contextually and dramatically exceeds the longevity-optimal target (<70 mg/dL). LDL-C at 142 mg/dL confirms significant atherogenic lipoprotein burden, though ApoB is the superior metric per NLA 2024 guidelines as it captures all atherogenic particles including VLDL and IDL remnants.

However, the most concerning finding is Lp(a) at 78 nmol/L. This exceeds both the standard risk threshold (<75 nmol/L) and the longevity-optimal target (<30 nmol/L). Lp(a) is genetically determined and non-modifiable by lifestyle intervention. Elevated Lp(a) carries independent cardiovascular risk through three mechanisms: atherogenic particle burden (additional ApoB-containing particles), pro-thrombotic effects (structural homology to plasminogen inhibiting fibrinolysis), and pro-inflammatory properties (carrying oxidized phospholipids that activate the innate immune system via TLR2). This directly accelerates the hallmarks of genomic instability and altered intercellular communication.

The combination of ApoB >80 mg/dL with hsCRP >1 mg/L triggers the PRISM Atherogenic Risk Enhancement constellation — synergistic risk where inflammatory modification of ApoB-containing lipoproteins accelerates plaque progression. The hsCRP at 3.2 mg/L (optimal <0.5) reflects chronic inflammaging, and homocysteine at 14.2 µmol/L (optimal <8) adds vascular endothelial damage risk.

The eGFR at 72 mL/min is below optimal (>90) but above the CKD threshold (60), representing early renal decline consistent with long-standing vascular damage. TG/HDL at 3.1 (optimal <1.0) suggests underlying insulin resistance further compounding vascular risk.

Lp(a) is prioritized because it is (1) genetically fixed, (2) independently atherogenic, thrombotic, and inflammatory, and (3) requires the most aggressive compensatory intervention on modifiable risk factors — specifically targeting ApoB to <60 mg/dL when Lp(a) is elevated, per ESC 2024 guidance.""",
    answer="B. Lp(a)",
    metadata={
        "pack": "P1", "pack_name": "prism-biomarker", "idx": 2,
        "content_hash": make_hash("61M atherogenic risk Lpa"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P1.append(make_entry(
    question="""A 38-year-old male presents for a longevity-focused health assessment. The following lab results are obtained:

  ALT: 52 U/L
  GGT: 48 U/L
  Fasting Glucose: 92 mg/dL
  Fasting Insulin: 12.4 mIU/L
  Ferritin: 385 ng/mL
  Transferrin Sat: 38%
  Uric Acid: 7.0 mg/dL

Which finding is most concerning from a longevity perspective?

A. ALT
B. Ferritin
C. Fasting Insulin
D. Uric Acid""",
    reasoning="""This panel reveals overlapping Insulin Resistance Cascade and hepatic stress signals with iron overload concern. ALT at 52 U/L is elevated above the standard range (<40 U/L) and far above the longevity-optimal target (<20 U/L). GGT at 48 U/L is similarly above optimal (<20 U/L). The combination of ALT >25 and GGT >25 in the presence of metabolic risk factors triggers the PRISM hepatic component of the Insulin Resistance Cascade — highly suggestive of non-alcoholic fatty liver disease (NAFLD/MASLD).

Ferritin at 385 ng/mL is within the standard range (20–300 for some labs, up to 400 in others) but dramatically exceeds the longevity-optimal range (40–150 ng/mL). However, transferrin saturation at 38% is within normal limits (25–40% optimal), which argues against true hereditary hemochromatosis and suggests the elevated ferritin is an acute phase reactant driven by hepatic inflammation and insulin resistance rather than primary iron overload. In this context, ferritin elevation reflects metabolic inflammation rather than Fenton-chemistry iron toxicity.

Fasting insulin at 12.4 mIU/L is within standard (<25) but nearly double the longevity-optimal threshold (<7 mIU/L). Combined with fasting glucose at 92 mg/dL (optimal 72–85), this yields a HOMA-IR of approximately 2.8 (optimal <1.0), confirming clinically significant insulin resistance. Uric acid at 7.0 mg/dL (optimal 4.0–5.5) is elevated and consistent with hyperinsulinemia reducing renal urate clearance.

ALT is the most concerning finding because it signals active hepatic steatosis — an organ-level manifestation of the metabolic dysfunction. Hepatic fat accumulation at age 38 accelerates the hallmark of deregulated nutrient sensing through hepatic insulin resistance → increased de novo lipogenesis → VLDL overproduction. NAFLD at this age, left untreated, carries risk of progression to NASH, fibrosis, and hepatocellular carcinoma, while also driving systemic cardiovascular risk through hepatic CRP production and atherogenic dyslipidemia. The liver is both a target and an amplifier of metabolic aging.""",
    answer="A. ALT",
    metadata={
        "pack": "P1", "pack_name": "prism-biomarker", "idx": 3,
        "content_hash": make_hash("38M hepatic steatosis IR"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P1.append(make_entry(
    question="""A 55-year-old female presents for a longevity-focused health assessment. The following lab results are obtained:

  Vitamin D: 18 ng/mL
  Vitamin B12: 280 pg/mL
  RBC Magnesium: 4.4 mg/dL
  Omega-3 Index: 3.2%
  Folate: 5.1 ng/mL
  hsCRP: 1.1 mg/L
  Homocysteine: 12.8 µmol/L

Which finding is most concerning from a longevity perspective?

A. Vitamin D
B. Omega-3 Index
C. Homocysteine
D. RBC Magnesium""",
    reasoning="""This panel reveals pervasive micronutrient insufficiency with downstream functional consequences. The most striking finding is the Omega-3 Index at 3.2%, which falls below even the standard minimum threshold (>4%) and is far below the longevity-optimal target (>8%). The Omega-3 Index is measured as EPA+DHA in red blood cell membranes, making it a stable 120-day biomarker of long-chain omega-3 status. An index <4% is associated with the highest risk category for sudden cardiac death (Harris & von Schacky) and represents the hallmark of altered intercellular communication — omega-3 fatty acids directly modulate membrane fluidity, eicosanoid balance (shifting from pro-inflammatory AA-derived PGE2/LTB4 toward anti-inflammatory EPA-derived PGE3/LTB5), and resolution of inflammation via specialized pro-resolving mediators (resolvins, protectins, maresins).

This is Tier A evidence — the strongest evidence tier among all markers in this panel. Vitamin D at 18 ng/mL is below standard (30 ng/mL) and far below optimal (40–60 ng/mL) but carries only Tier C evidence for specific longevity targets. Vitamin B12 at 280 pg/mL is within standard (200–900) but below optimal (400–800); checking methylmalonic acid (MMA) would clarify functional insufficiency.

Homocysteine at 12.8 µmol/L is within standard (<15) but above optimal (<8 µmol/L). This elevation likely reflects the combination of suboptimal B12 and marginal folate (5.1 ng/mL, optimal >15 ng/mL) — both serve as methyl donors in the remethylation of homocysteine via methionine synthase. The elevated homocysteine represents impaired methylation capacity, relevant to the hallmark of epigenetic alterations (DNA methylation dysregulation).

RBC Magnesium at 4.4 mg/dL is near the bottom of the standard range (4.2–6.8) and below optimal (6.0–6.5), contributing to subclinical magnesium depletion that affects >300 enzymatic reactions.

The Omega-3 Index is prioritized because (1) it has the highest evidence tier (A), (2) it is the only value below even the standard threshold, (3) it reflects a systemic anti-inflammatory deficit that amplifies the hsCRP elevation, and (4) it is readily correctable with marine omega-3 supplementation (2-4g EPA+DHA daily typically raises the index by 4-5 points over 4 months).""",
    answer="B. Omega-3 Index",
    metadata={
        "pack": "P1", "pack_name": "prism-biomarker", "idx": 4,
        "content_hash": make_hash("55F micronutrient omega3"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P1.append(make_entry(
    question="""A 67-year-old male presents for a longevity-focused health assessment. The following lab results are obtained:

  eGFR: 58 mL/min
  Cystatin C: 1.3 mg/L
  Uric Acid: 8.4 mg/dL
  hsCRP: 4.1 mg/L
  NLR: 4.2
  RDW: 15.1%
  Fibrinogen: 445 mg/dL

Which finding is most concerning from a longevity perspective?

A. eGFR
B. NLR
C. RDW
D. Fibrinogen""",
    reasoning="""This panel reveals two converging pathological patterns: Stage 3a chronic kidney disease and a severe Chronic Inflammation constellation. The eGFR at 58 mL/min falls below the CKD Stage 3 threshold (<60) and is significantly below the longevity-optimal target (>90 mL/min). Cystatin C at 1.3 mg/L confirms the renal impairment (optimal <0.8) — cystatin C is superior to creatinine-based eGFR as it is not confounded by muscle mass, diet, or race. This represents Tier A evidence for renal function assessment.

The inflammatory panel is uniformly alarming. hsCRP at 4.1 mg/L is 8× the longevity-optimal threshold (0.5 mg/L), indicating significant chronic inflammation — the hallmark of inflammaging. NLR at 4.2 exceeds both the standard (<3.0) and optimal (<2.0) thresholds, reflecting a neutrophil-predominant immune activation state associated with all-cause mortality in multiple large cohort studies.

RDW at 15.1% exceeds the standard range (11.5–14.5%) and the longevity-optimal threshold (<13.0%). Elevated RDW is an independent all-cause mortality predictor across multiple populations — it reflects ineffective erythropoiesis from chronic inflammation, nutritional deficiency, or bone marrow stress, representing the hallmark of stem cell exhaustion.

Fibrinogen at 445 mg/dL exceeds the standard range (200–400) and the optimal target (<300), reflecting both thrombotic risk and inflammatory burden as an acute-phase reactant.

Uric acid at 8.4 mg/dL (optimal 4.0–5.5) is markedly elevated and contributes to both renal damage (crystal nephropathy) and inflammation (NLRP3 inflammasome activation).

The eGFR at 58 mL/min is the most concerning finding because (1) it is the only value that has already crossed into frank disease territory (CKD Stage 3a), (2) renal decline is largely irreversible unlike inflammatory markers which can be modulated, (3) CKD creates a vicious cycle — reduced renal clearance worsens uric acid retention, which exacerbates both inflammation and further nephron loss, and (4) eGFR carries Tier A evidence as a mortality predictor. The inflammatory constellation is both a cause and consequence of the renal decline, but the kidney function represents the most advanced and least reversible pathology.""",
    answer="A. eGFR",
    metadata={
        "pack": "P1", "pack_name": "prism-biomarker", "idx": 5,
        "content_hash": make_hash("67M CKD inflammation"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P1.append(make_entry(
    question="""A 44-year-old male presents for a longevity-focused health assessment. The following lab results are obtained:

  Total Testosterone: 320 ng/dL
  Free Testosterone: 6.2 ng/dL
  SHBG: 58 nmol/L
  Fasting Insulin: 4.8 mIU/L
  Vitamin D: 52 ng/mL
  HbA1c: 5.0%

Which finding is most concerning from a longevity perspective?

A. Total Testosterone
B. Free Testosterone
C. SHBG
D. Fasting Insulin""",
    reasoning="""This panel reveals an endocrine pattern where the metabolic markers are reassuringly optimal but the androgenic axis is significantly suboptimal. Fasting insulin at 4.8 mIU/L is within the longevity-optimal range (<7), HbA1c at 5.0% is optimal (<5.3%), and vitamin D at 52 ng/mL sits squarely in the optimal window (40–60 ng/mL). These findings rule out metabolic dysfunction as a driver.

Total testosterone at 320 ng/dL is within the standard range (300–1000) but sits at the very bottom, below the longevity-optimal floor of 500 ng/dL. Free testosterone at 6.2 ng/dL is similarly borderline standard (5–21) but below optimal (10–18 ng/dL). The low free testosterone is explained by the SHBG at 58 nmol/L, which exceeds both the standard range (16–55) and the optimal range (20–40) — elevated SHBG sequesters bioavailable testosterone, reducing tissue-level androgenic signaling.

SHBG is the most concerning finding from a longevity perspective because it is the mechanistic driver of the low free testosterone state. While total T is borderline, the elevated SHBG explains WHY free T is low — correcting SHBG would normalize free T without requiring exogenous testosterone. SHBG elevation in a metabolically healthy 44-year-old male (low insulin, normal HbA1c) may reflect hepatic overproduction or hyperthyroid tendency. This pattern connects to the hallmarks of altered intercellular communication and loss of proteostasis — adequate androgenic signaling is required for muscle protein synthesis, bone mineral density maintenance, and neuroplasticity.

The clinical significance is that this patient's tissue-level testosterone exposure is that of a much older man despite nominally "normal" labs. A standard panel checking only total T would report "within range." The SHBG-adjusted interpretation reveals the true deficit and points toward specific interventions: evaluate thyroid function, consider boron supplementation (which may lower SHBG), optimize magnesium and zinc status, and reassess in 3 months before considering TRT.""",
    answer="C. SHBG",
    metadata={
        "pack": "P1", "pack_name": "prism-biomarker", "idx": 6,
        "content_hash": make_hash("44M SHBG testosterone"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P1.append(make_entry(
    question="""A 50-year-old male presents for a longevity-focused health assessment. The following lab results are obtained:

  Ferritin: 412 ng/mL
  Transferrin Sat: 52%
  ALT: 28 U/L
  GGT: 32 U/L
  hsCRP: 0.4 mg/L
  HbA1c: 5.1%
  eGFR: 95 mL/min

Which finding is most concerning from a longevity perspective?

A. Ferritin
B. Transferrin Sat
C. ALT
D. GGT""",
    reasoning="""This panel presents an important diagnostic distinction. Most markers are reassuringly close to optimal: hsCRP at 0.4 mg/L is within the longevity-optimal range (<0.5), HbA1c at 5.1% is optimal (<5.3%), eGFR at 95 mL/min is optimal (>90). ALT at 28 U/L and GGT at 32 U/L are within standard but above optimal (<20 U/L each), suggesting mild hepatic stress.

The critical finding is transferrin saturation at 52%, which exceeds both the standard range ceiling (50%) and the longevity-optimal range (25–40%). Combined with ferritin at 412 ng/mL — well above the longevity-optimal ceiling of 150 ng/mL — this pattern is consistent with early hereditary hemochromatosis or significant iron overload. Crucially, the hsCRP is low (0.4 mg/L), which means the elevated ferritin cannot be dismissed as an acute-phase response. When ferritin is elevated without inflammation, it more likely reflects genuine iron stores excess.

Transferrin saturation >45% is the most sensitive screening marker for hereditary hemochromatosis (HFE gene mutations). Iron overload drives oxidative damage through Fenton chemistry: Fe²⁺ + H₂O₂ → Fe³⁺ + OH· + OH⁻, generating hydroxyl radicals that cause lipid peroxidation, protein carbonylation, and DNA damage. This directly accelerates the hallmarks of genomic instability and mitochondrial dysfunction. Excess hepatic iron explains the mildly elevated ALT and GGT even in the absence of metabolic syndrome.

Transferrin saturation is prioritized over ferritin because (1) it is the earlier and more specific marker for iron overload pathophysiology, (2) at 52% it has crossed the standard threshold while ferritin elevation alone can have many causes, (3) it mandates immediate workup including HFE genotyping, and (4) undiagnosed hemochromatosis is one of the most treatable causes of accelerated biological aging — simple therapeutic phlebotomy can normalize iron stores and halt organ damage.""",
    answer="B. Transferrin Sat",
    metadata={
        "pack": "P1", "pack_name": "prism-biomarker", "idx": 7,
        "content_hash": make_hash("50M iron overload transferrin"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

# ──────────────────────────────────────────────────────────────────
# P4: DRUG REPURPOSING (8 examples)
# ──────────────────────────────────────────────────────────────────

P4 = []

P4.append(make_entry(
    question="""A 55-year-old male with early insulin resistance (HOMA-IR 2.1), no diabetes. From a geroprotective/longevity medicine perspective, which drug would be most appropriate to consider as an adjunct intervention?

A. Dasatinib+Quercetin
B. Metformin
C. Rapamycin
D. Low-Dose Naltrexone""",
    reasoning="""Standard care for HOMA-IR 2.1 without diabetes would focus on lifestyle modification: dietary carbohydrate reduction, resistance training, and weight management. This patient does not yet meet diagnostic criteria for pharmacological intervention under conventional guidelines.

From a geroprotective perspective, metformin is the strongest candidate for this clinical scenario. Metformin's primary mechanism is AMPK activation → suppression of hepatic gluconeogenesis → improved peripheral insulin sensitivity. At the molecular level, metformin inhibits Complex I of the mitochondrial electron transport chain, increasing the AMP/ATP ratio, which activates AMPK. Downstream effects include: AMPK → inhibition of mTORC1 → enhanced autophagy (hallmark: loss of proteostasis), and AMPK → activation of PGC-1α → improved mitochondrial biogenesis (hallmark: mitochondrial dysfunction).

Evidence tier: [B] observational. The Diabetes Prevention Program showed metformin reduced progression to T2DM by 31% vs placebo in prediabetic patients. The TAME trial (Targeting Aging with Metformin) has moved to ARPA-H but has not yet reported results as of March 2026. Large observational studies suggest diabetic patients on metformin may have lower all-cause mortality than age-matched non-diabetic controls, though the key Bannister 2014 study has been criticized and a 2022 re-analysis reached contradictory conclusions.

Dosing: 500 mg once daily, titrating to 1000–1500 mg/day. Monitor: B12 levels every 6-12 months (metformin reduces B12 absorption via calcium-dependent ileal uptake). Monitor renal function (contraindicated if eGFR <30). Caution: GI side effects are common (20-30%); extended-release formulation improves tolerability. Avoid in patients with hepatic insufficiency or active alcohol use disorder.

Rapamycin (C) would be premature for isolated early IR — its intermittent dosing protocols are more appropriate for patients without active metabolic dysfunction. Dasatinib+Quercetin (A) is a senolytic cocktail targeting senescent cells; this patient shows no markers of high senescent cell burden. Low-Dose Naltrexone (D) modulates immune function but does not address the insulin resistance pathway.""",
    answer="B. Metformin",
    metadata={
        "pack": "P4", "pack_name": "prism-repurposing", "idx": 0,
        "content_hash": make_hash("55M IR metformin"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "drug": "Metformin", "scenario": "55M with early insulin resistance",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P4.append(make_entry(
    question="""A 62-year-old female with hsCRP 2.8, ApoB 95, moderate ASCVD risk. From a geroprotective/longevity medicine perspective, which drug would be most appropriate to consider as an adjunct intervention?

A. Taurine
B. Telmisartan
C. Rapamycin
D. Empagliflozin""",
    reasoning="""This patient presents with the PRISM Atherogenic Risk Enhancement constellation: ApoB >80 mg/dL combined with hsCRP >1 mg/L. Standard care would include high-intensity statin therapy per ACC/AHA guidelines for moderate ASCVD risk, with lifestyle modification.

From a geroprotective perspective, telmisartan offers a unique dual mechanism that addresses both components of this patient's risk constellation. Telmisartan is an angiotensin II receptor blocker (ARB) that also acts as a partial PPARγ agonist — a property unique among ARBs. Mechanism: AT1 receptor blockade → reduced vascular inflammation, reduced oxidative stress, improved endothelial function. PPARγ activation → improved insulin sensitivity, anti-inflammatory transcriptional program, reduced NF-κB signaling. This targets the hallmarks of altered intercellular communication and cellular senescence (telmisartan has been shown to reduce senescence-associated secretory phenotype markers in preclinical models).

Evidence tier: [B] observational/mechanistic. The ONTARGET trial demonstrated cardiovascular protection comparable to ACE inhibitors. Telmisartan's PPARγ activity is supported by in vitro binding studies and clinical data showing improved metabolic parameters compared to other ARBs. Preclinical data suggest telmisartan extends lifespan in animal models through reduction of inflammatory markers and improved vascular health.

Dosing: 40–80 mg once daily. Monitor: blood pressure, potassium, renal function (creatinine/eGFR) at baseline and 2-4 weeks after initiation. Caution: contraindicated in bilateral renal artery stenosis and pregnancy. Avoid combination with ACE inhibitors or aliskiren (hyperkalemia risk).

Empagliflozin (D) would be more appropriate if this patient had diabetes, heart failure, or CKD as the primary concern. Rapamycin (C) is not first-line for vascular inflammation with available alternatives. Taurine (A) has emerging evidence for anti-aging properties but limited clinical trial data for cardiovascular risk reduction [C].""",
    answer="B. Telmisartan",
    metadata={
        "pack": "P4", "pack_name": "prism-repurposing", "idx": 1,
        "content_hash": make_hash("62F ASCVD telmisartan"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "drug": "Telmisartan", "scenario": "62F moderate ASCVD",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P4.append(make_entry(
    question="""A 73-year-old male with HFpEF, eGFR 45, uric acid 8.1. From a geroprotective/longevity medicine perspective, which drug would be most appropriate to consider as an adjunct intervention?

A. Metformin
B. Rapamycin
C. Empagliflozin
D. Acarbose""",
    reasoning="""This patient has three concurrent pathologies: heart failure with preserved ejection fraction, Stage 3b chronic kidney disease, and hyperuricemia. Standard care includes diuretics, blood pressure optimization, and guideline-directed medical therapy for HFpEF.

Empagliflozin is the clear geroprotective choice because it simultaneously addresses all three conditions through a single mechanism. SGLT2 inhibition in the proximal tubule reduces glucose and sodium reabsorption → osmotic diuresis → reduced preload → improved cardiac hemodynamics. This is the direct mechanism for the EMPEROR-Preserved trial result: empagliflozin reduced the composite of cardiovascular death and heart failure hospitalization by 21% in HFpEF patients (HR 0.79, 95% CI 0.69-0.90).

Evidence tier: [A] RCT for HFpEF and CKD. EMPA-KIDNEY demonstrated renal protection with 28% risk reduction for kidney disease progression independent of diabetes status. The renal mechanism involves restoration of tubuloglomerular feedback → reduced intraglomerular pressure → reduced hyperfiltration → slower GFR decline. For uric acid, SGLT2 inhibitors reduce serum uric acid by 10-15% through increased renal urate excretion via GLUT9 transporter competition.

Beyond the standard cardiorenal benefits, empagliflozin has emerging geroprotective evidence. Katsuumi et al. (Nature Aging 2024) demonstrated that SGLT2 inhibitors clear senescent cells via immunosenolysis — restoring immune surveillance of senescent cells through enhanced natural killer cell and macrophage-mediated clearance. This directly targets the hallmarks of cellular senescence and mitochondrial dysfunction (SGLT2i promote a metabolic shift from glucose to ketone body and fatty acid oxidation, enhancing mitochondrial efficiency).

Dosing: 10 mg once daily. No dose adjustment needed for eGFR 45 (effective down to eGFR 20 for cardiorenal indication). Monitor: renal function, ketones (rare euglycemic DKA risk, primarily in T1DM), volume status, genital mycotic infections. Caution: risk of volume depletion if combined with high-dose loop diuretics — may need diuretic dose reduction.

Metformin (A) is contraindicated with eGFR <30 and should be used cautiously at eGFR 30-45; it does not address HFpEF. Rapamycin (B) lacks RCT evidence for HFpEF and carries immunosuppression risk in elderly patients. Acarbose (D) showed ITP lifespan extension in males but has no evidence for cardiorenal benefit.""",
    answer="C. Empagliflozin",
    metadata={
        "pack": "P4", "pack_name": "prism-repurposing", "idx": 2,
        "content_hash": make_hash("73M HFpEF CKD empagliflozin"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "drug": "Empagliflozin", "scenario": "73M HFpEF CKD",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P4.append(make_entry(
    question="""A 50-year-old female with breast cancer history, now in remission, seeking longevity optimization. From a geroprotective/longevity medicine perspective, which drug would be most appropriate to consider as an adjunct intervention?

A. Rapamycin
B. NMN
C. 17α-Estradiol
D. Dasatinib+Quercetin""",
    reasoning="""This is a clinically complex scenario requiring careful risk-benefit analysis. The patient's cancer history imposes significant constraints on geroprotective options because several longevity agents may promote cancer recurrence through growth factor signaling.

Rapamycin is the optimal choice for this specific patient because it simultaneously provides geroprotection AND anti-cancer surveillance. Rapamycin inhibits mTORC1 → suppresses cap-dependent translation of oncogenic mRNAs (including cyclin D1, c-Myc, HIF-1α), reduces cell proliferation, and enhances autophagy-mediated clearance of damaged organelles. This targets the hallmarks of deregulated nutrient sensing and disabled macroautophagy.

Evidence tier: [B] mechanistic + observational. In the Interventions Testing Program (ITP), rapamycin at 42 ppm extended median lifespan by 23% in males and 26% in females (Miller 2014). mTOR inhibitors (everolimus, temsirolimus) are FDA-approved for certain cancers, demonstrating anti-neoplastic activity. The PEARL trial showed everolimus + exemestane improved PFS in HR+ breast cancer. Low-dose intermittent rapamycin (e.g., 5-6 mg once weekly) is being explored in longevity protocols to capture geroprotective benefits while minimizing immunosuppression.

Dosing protocol: 5-6 mg once weekly (intermittent, not daily). This pulsatile dosing preferentially inhibits mTORC1 while allowing mTORC2 recovery between doses — daily dosing suppresses both complexes and causes metabolic side effects. Monitor: fasting lipids (rapamycin can elevate LDL/TG), fasting glucose, CBC with differential (lymphocyte count). Caution: avoid in patients with active infections; monitor for mouth ulcers (mucositis). Obtain baseline lipid panel and glucose before initiation.

NMN (B) raises NAD+ levels which promotes cell survival and proliferation — potentially problematic in cancer survivors due to enhanced tumor cell metabolism via NAMPT-dependent pathways. 17α-Estradiol (C) is a non-feminizing estrogen shown to extend male mouse lifespan in ITP but is specifically inappropriate for a breast cancer survivor due to estrogen receptor concerns regardless of its reduced ER binding affinity. Dasatinib+Quercetin (D) is a senolytic cocktail with emerging evidence but currently lacks cancer-specific safety data and quercetin has complex effects on estrogen receptors.""",
    answer="A. Rapamycin",
    metadata={
        "pack": "P4", "pack_name": "prism-repurposing", "idx": 3,
        "content_hash": make_hash("50F breast cancer rapamycin"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "drug": "Rapamycin", "scenario": "50F breast cancer remission",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P4.append(make_entry(
    question="""A 48-year-old male with metabolic syndrome, BMI 34, ALT 45. From a geroprotective/longevity medicine perspective, which drug would be most appropriate to consider as an adjunct intervention?

A. Spermidine
B. Low-Dose Naltrexone
C. Semaglutide
D. Taurine""",
    reasoning="""This patient presents with metabolic syndrome (likely meeting ≥3 ATP III criteria given BMI 34) and elevated ALT suggesting hepatic steatosis (NAFLD/MASLD). Standard care includes intensive lifestyle intervention targeting ≥7% weight loss, with consideration of pharmacotherapy for obesity.

Semaglutide is the strongest geroprotective intervention for this clinical profile. As a GLP-1 receptor agonist, semaglutide's mechanism extends far beyond glucose regulation: GLP-1R activation → enhanced insulin secretion (glucose-dependent) → improved peripheral insulin sensitivity, but also central GLP-1R signaling in the hypothalamus → reduced appetite → significant weight loss. The downstream effects include: reduced hepatic de novo lipogenesis → improved NAFLD, reduced visceral adipose tissue → decreased inflammatory cytokine production (TNF-α, IL-6), and emerging evidence for direct anti-inflammatory effects in the vasculature.

Evidence tier: [A] RCT. SELECT trial (2023): semaglutide 2.4 mg weekly reduced MACE by 20% in overweight/obese patients without diabetes. STEP trials demonstrated 15-17% body weight loss. For NAFLD specifically, the Phase 2 trial showed histological resolution of NASH in 59% of patients on semaglutide vs 17% placebo. Regarding geroprotective mechanisms, Li et al. (Cell Metabolism 2025, DOI: 10.1016/j.cmet.2025.10.014) demonstrated that exenatide (a GLP-1 RA) administered to healthy older adults (60-80 years) for 40 weeks reversed epigenetic aging markers and improved immunosenescence profiles.

Dosing: start 0.25 mg SC weekly × 4 weeks, titrate monthly (0.25 → 0.5 → 1.0 → 1.7 → 2.4 mg). Monitor: liver function (expected to improve), HbA1c, lipids, thyroid function (theoretical medullary thyroid carcinoma risk from rodent studies — no human cases confirmed). Caution: contraindicated with personal/family history of medullary thyroid carcinoma or MEN2. GI side effects (nausea, vomiting) are dose-dependent and typically resolve with slow titration. Monitor for gallbladder disease with rapid weight loss.

Spermidine (A) promotes autophagy but has [C] evidence and would not address the magnitude of metabolic dysfunction present. Low-Dose Naltrexone (B) has immunomodulatory properties but lacks evidence for metabolic syndrome. Taurine (D) showed lifespan extension in mice but has insufficient human clinical trial evidence [C] for this indication.""",
    answer="C. Semaglutide",
    metadata={
        "pack": "P4", "pack_name": "prism-repurposing", "idx": 4,
        "content_hash": make_hash("48M metabolic semaglutide"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "drug": "Semaglutide", "scenario": "48M metabolic syndrome",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P4.append(make_entry(
    question="""A 60-year-old male with early CKD (eGFR 55), hypertension, gout. From a geroprotective/longevity medicine perspective, which drug would be most appropriate to consider as an adjunct intervention?

A. Acarbose
B. Empagliflozin
C. Rapamycin
D. NMN""",
    reasoning="""This patient has a triad of chronic kidney disease, hypertension, and gout — three conditions with shared pathophysiology through the cardiorenal-metabolic axis. Standard care includes antihypertensives (ACEi/ARB first-line), urate-lowering therapy (allopurinol/febuxostat if recurrent gout), and CKD management.

Empagliflozin addresses all three conditions through a single molecular mechanism. SGLT2 inhibition in the proximal tubule reduces sodium-glucose co-transport → natriuresis and osmotic diuresis → blood pressure reduction of approximately 3-5/1-2 mmHg systolic/diastolic. For CKD, empagliflozin restores tubuloglomerular feedback → reduces intraglomerular hypertension → slows GFR decline. EMPA-KIDNEY trial [A]: 28% relative risk reduction in kidney disease progression, effective regardless of diabetes status and down to eGFR 20.

For hyperuricemia/gout, SGLT2 inhibitors reduce serum uric acid by approximately 10-15% through competitive inhibition of GLUT9-mediated urate reabsorption in the proximal tubule and enhanced uricosuria. This is mechanistically distinct from xanthine oxidase inhibitors and provides additional urate-lowering beyond standard therapy.

Evidence tier: [A] RCT for CKD and cardiovascular outcomes. EMPA-KIDNEY, CREDENCE, and DAPA-CKD collectively provide robust evidence for SGLT2i renal protection. Additional geroprotective evidence: Katsuumi et al. (Nature Aging 2024) demonstrated immunosenolytic activity — SGLT2 inhibitors restore immune clearance of senescent cells, targeting the hallmark of cellular senescence.

Dosing: 10 mg once daily. Safe at eGFR 55 (cardiorenal benefit persists to eGFR 20). Monitor: eGFR (expected acute 10-15% decline that stabilizes — do not discontinue), potassium, volume status, uric acid. Caution: hold during acute illness, surgery, or prolonged fasting (euglycemic DKA risk); genital mycotic infection risk ~5%.

Acarbose (A) showed lifespan extension in male mice (ITP, Harrison 2014) but has no CKD or gout evidence and GI side effects limit tolerability. Rapamycin (C) has immunosuppressive risk concerning in CKD. NMN (D) has [C] evidence with no renal protection data.""",
    answer="B. Empagliflozin",
    metadata={
        "pack": "P4", "pack_name": "prism-repurposing", "idx": 5,
        "content_hash": make_hash("60M CKD gout empagliflozin"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "drug": "Empagliflozin", "scenario": "60M CKD hypertension gout",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P4.append(make_entry(
    question="""A 70-year-old female with osteoporosis, sarcopenia, declining grip strength. From a geroprotective/longevity medicine perspective, which drug would be most appropriate to consider as an adjunct intervention?

A. Rapamycin
B. Semaglutide
C. Taurine
D. Dasatinib+Quercetin""",
    reasoning="""This patient presents with the hallmark of stem cell exhaustion manifesting as musculoskeletal decline: osteoporosis (bone-forming osteoblast decline) and sarcopenia with declining grip strength (satellite cell dysfunction). Standard care includes bisphosphonates or denosumab for osteoporosis, resistance training, adequate protein intake (1.2-1.6 g/kg/day), and vitamin D/calcium supplementation.

Dasatinib+Quercetin (D+Q) is the most appropriate geroprotective intervention for this clinical phenotype. D+Q is a senolytic combination: dasatinib (a tyrosine kinase inhibitor) targets senescent preadipocytes and endothelial cells via inhibition of ephrin/SFK/PAI pathways, while quercetin (a flavonoid) targets senescent fibroblasts and bone marrow stromal cells via BCL-2/BCL-XL inhibition. Together, they selectively eliminate senescent cells that accumulate with aging and secrete the senescence-associated secretory phenotype (SASP) — a cocktail of IL-6, IL-8, MMPs, and other factors that degrade the tissue microenvironment.

In the musculoskeletal context, senescent cells in bone marrow suppress osteoblast differentiation and promote osteoclastogenesis via SASP-mediated RANKL upregulation. In skeletal muscle, senescent satellite cells impair regenerative capacity. Preclinical evidence: D+Q improved physical function, reduced SASP markers, and enhanced muscle regenerative capacity in aged mice. First-in-human trials in idiopathic pulmonary fibrosis showed reduced SASP markers (p16INK4a positive cells) and improved physical function metrics.

Evidence tier: [B-C] — translational. Human safety data from phase I/II trials in IPF and diabetic kidney disease. No large RCTs for sarcopenia-specific endpoints yet, but mechanistic rationale is strong.

Dosing protocol: Dasatinib 100 mg + Quercetin 1000 mg for 3 consecutive days, repeated monthly or quarterly (hit-and-run approach — senolytics need only transient exposure since senescent cells do not regenerate). Monitor: CBC (dasatinib is a kinase inhibitor — check platelets), liver function, GI symptoms. Caution: dasatinib can cause fluid retention, QTc prolongation, and pulmonary arterial hypertension with chronic use, though the intermittent 3-day protocol minimizes these risks.

Rapamycin (A) inhibits mTOR which could worsen sarcopenia by suppressing muscle protein synthesis via the mTORC1→S6K1→4E-BP1 pathway. Semaglutide (B) causes weight loss that could exacerbate sarcopenia (20-40% of GLP-1RA-induced weight loss is lean mass). Taurine (C) has emerging longevity evidence but [C] tier, insufficient for this severity.""",
    answer="D. Dasatinib+Quercetin",
    metadata={
        "pack": "P4", "pack_name": "prism-repurposing", "idx": 6,
        "content_hash": make_hash("70F sarcopenia DQ"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "drug": "Dasatinib+Quercetin", "scenario": "70F osteoporosis sarcopenia",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P4.append(make_entry(
    question="""A 52-year-old male with prediabetes (HbA1c 5.9%), considering prevention. From a geroprotective/longevity medicine perspective, which drug would be most appropriate to consider as an adjunct intervention?

A. 17α-Estradiol
B. Acarbose
C. Metformin
D. Spermidine""",
    reasoning="""This patient sits at the critical inflection point of the diabetes trajectory — HbA1c 5.9% is one decimal point below the diagnostic threshold (6.5%) and represents the last window for prevention before frank disease. Standard care: intensive lifestyle intervention (DPP showed 58% diabetes risk reduction with lifestyle alone).

Metformin is the evidence-based geroprotective choice for prediabetes prevention. The Diabetes Prevention Program (DPP) is the definitive trial: metformin 850 mg BID reduced progression to T2DM by 31% vs placebo over 2.8 years (NNT = 14). The 15-year DPP Outcomes Study showed durable benefit with 18% risk reduction sustained. Metformin was most effective in patients with BMI ≥35, fasting glucose ≥110, and age <60 — this patient's age (52) falls within the optimal range.

Mechanistically, metformin inhibits mitochondrial Complex I → increased AMP/ATP ratio → AMPK activation → suppression of hepatic gluconeogenesis + enhanced peripheral glucose uptake via GLUT4 translocation. AMPK activation also inhibits mTORC1 → enhanced autophagy (hallmark: disabled macroautophagy) and activates SIRT1 → improved NAD+ metabolism and epigenetic regulation (hallmark: epigenetic alterations). The multi-hallmark engagement makes metformin among the most well-studied geroprotective agents.

Evidence tier: [A] RCT for diabetes prevention. [B] for longevity-specific outcomes (TAME trial moved to ARPA-H, not yet reporting as of March 2026).

Dosing: start 500 mg with dinner × 2 weeks, titrate to 500 mg BID, then 850 mg BID if tolerated. Extended-release formulation preferred for GI tolerability. Monitor: B12 annually (metformin reduces B12 absorption by 20-30% over 4 years per DPP data), renal function (eGFR), lactate if acute illness (rare lactic acidosis risk, primarily with renal impairment). Caution: hold 48 hours before iodinated contrast procedures; GI side effects (diarrhea, nausea) in 20-30%.

Acarbose (B) showed ITP lifespan extension in males (Strong 2016, ~5% median extension at 1000 ppm) but alpha-glucosidase inhibition is less evidence-based than metformin for diabetes prevention and causes significant flatulence limiting adherence. 17α-Estradiol (A) extends lifespan in male mice only and is not applicable. Spermidine (D) promotes autophagy but has [C] tier evidence insufficient for this clinical decision.""",
    answer="C. Metformin",
    metadata={
        "pack": "P4", "pack_name": "prism-repurposing", "idx": 7,
        "content_hash": make_hash("52M prediabetes metformin"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "drug": "Metformin", "scenario": "52M prediabetes",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

# ──────────────────────────────────────────────────────────────────
# P5: TRAJECTORY (8 examples)
# ──────────────────────────────────────────────────────────────────

P5 = []

P5.append(make_entry(
    question="""A 58-year-old male has serial labs over 24 months:

| Marker | Month 0 | Month 6 | Month 12 | Month 18 | Month 24 |
| --- | --- | --- | --- | --- | --- |
| HbA1c (%) | 5.4 | 5.5 | 5.6 | 5.7 | 5.8 |
| ApoB (mg/dL) | 82 | 85 | 84 | 86 | 83 |

What is the most concerning trend in this patient's longitudinal data?

A. Rising HbA1c trajectory
B. Stable elevated ApoB
C. ApoB crossing into abnormal range
D. Stable inflammatory markers""",
    reasoning="""Rate-of-change analysis reveals two fundamentally different patterns. HbA1c shows a monotonic upward trajectory: 5.4% → 5.5% → 5.6% → 5.7% → 5.8% over 24 months. The rate of change is +0.4 percentage points over 24 months = +0.2%/year. While each individual measurement is within the standard range (<5.7% for the first three, and the final two just at or above the prediabetes threshold), the trajectory is the critical signal.

At this rate, HbA1c will reach 6.0% in approximately 12 additional months and cross the diabetes threshold of 6.5% within 3-4 years. The trajectory pattern reveals progressive loss of glycemic control that no single static measurement would flag — at Month 0, this patient would be classified as "perfectly normal." The threshold crossing between Month 12 (5.6%) and Month 18 (5.7%) represents the transition from normal to prediabetes, but the clinically actionable signal is the consistent upward slope preceding this crossing.

ApoB shows a stable pattern at 82-86 mg/dL (mean ~84), fluctuating within measurement noise. While this is suboptimal from a longevity perspective (optimal <70 mg/dL), there is no directional trend — the static elevation is concerning but not worsening. Comparing trajectory significance vs absolute value significance: a stable elevated value at 84 mg/dL is less concerning than an actively deteriorating value with a clear temporal vector, because the deteriorating trajectory indicates an underlying pathological process that is still progressing.

The rising HbA1c trajectory is most concerning because (1) it demonstrates the highest rate of clinically meaningful change (+0.2%/year), (2) it has already crossed a diagnostic threshold during the observation period, (3) the monotonic nature (no reversals) suggests a persistent driver (likely progressive insulin resistance) rather than random variation, and (4) early intervention at this stage (lifestyle + potential metformin) has the greatest impact on preventing the progression cascade to full T2DM.""",
    answer="A. Rising HbA1c trajectory",
    metadata={
        "pack": "P5", "pack_name": "prism-trajectory", "idx": 0,
        "content_hash": make_hash("58M HbA1c trajectory"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P5.append(make_entry(
    question="""A 64-year-old female has serial labs over 18 months:

| Marker | Month 0 | Month 6 | Month 12 | Month 18 |
| --- | --- | --- | --- | --- |
| eGFR (mL/min) | 78 | 72 | 65 | 61 |
| hsCRP (mg/L) | 1.2 | 1.8 | 2.5 | 3.1 |
| ApoB (mg/dL) | 95 | 92 | 90 | 88 |

What is the most concerning trend in this patient's longitudinal data?

A. Declining eGFR trajectory
B. Rising hsCRP trend
C. Persistently elevated ApoB
D. ApoB improvement trend""",
    reasoning="""Rate-of-change analysis across three markers reveals one alarm-level trajectory. eGFR is declining from 78 → 72 → 65 → 61 mL/min over 18 months. The rate of change is -17 mL/min over 18 months = -11.3 mL/min/year. Normal age-related eGFR decline is approximately 0.7-1.0 mL/min/year after age 40 — this patient is losing renal function at more than 10× the expected rate. This pace of decline is consistent with accelerated nephropathy rather than physiological aging.

The eGFR trajectory is approaching the CKD Stage 3b threshold (45 mL/min). At the current rate of decline, this patient would reach Stage 3b in approximately 17 months and Stage 4 (<30) within about 3 years. The trajectory vs absolute value distinction is critical here: an eGFR of 61 at a single timepoint in a 64-year-old might prompt modest concern, but a -11.3 mL/min/year decline rate transforms this from "monitor" to "urgent investigation" — potential causes include progressive diabetic nephropathy, hypertensive nephrosclerosis, or glomerulonephritis requiring immediate workup (renal ultrasound, urinalysis with microscopy, urine ACR).

The rising hsCRP (1.2 → 3.1 mg/L, +1.06/year) is concerning but may be reactive to the underlying renal inflammation. ApoB shows a favorable downward trend (95 → 88, likely medication-mediated) and is the least concerning trajectory.

The declining eGFR trajectory is most concerning because (1) the rate of decline (-11.3/year) is pathological, not physiological, (2) renal function loss is largely irreversible — once nephrons are destroyed, they do not regenerate, (3) eGFR decline at this rate will reach ESRD within a decade without intervention, and (4) eGFR is a Tier A mortality predictor. Immediate intervention: blood pressure optimization (<130/80), SGLT2 inhibitor initiation (empagliflozin or dapagliflozin), nephrology referral.""",
    answer="A. Declining eGFR trajectory",
    metadata={
        "pack": "P5", "pack_name": "prism-trajectory", "idx": 1,
        "content_hash": make_hash("64F eGFR decline"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P5.append(make_entry(
    question="""A 46-year-old male has serial labs over 18 months:

| Marker | Month 0 | Month 6 | Month 12 | Month 18 |
| --- | --- | --- | --- | --- |
| Fasting Insulin (mIU/L) | 8.2 | 10.1 | 12.8 | 15.4 |
| Fasting Glucose (mg/dL) | 84 | 86 | 88 | 90 |
| ALT (U/L) | 18 | 22 | 29 | 35 |

What is the most concerning trend in this patient's longitudinal data?

A. Rising fasting insulin trajectory
B. Gradually increasing glucose
C. ALT crossing into suboptimal range
D. Stable metabolic markers""",
    reasoning="""Rate-of-change analysis reveals a classic insulin resistance progression pattern where the most informative signal is the earliest marker in the cascade — fasting insulin. Fasting insulin is rising from 8.2 → 10.1 → 12.8 → 15.4 mIU/L over 18 months, a rate of change of +7.2 mIU/L over 18 months = +4.8 mIU/L/year. At Month 0, the value was mildly above the longevity-optimal threshold (<7 mIU/L) but close. By Month 18, it has nearly doubled and sits at 2.2× the optimal ceiling.

The critical trajectory insight: glucose is barely moving (84 → 90, +4 mIU/L/year = +3.3/year) because the rising insulin is COMPENSATING — the pancreatic beta cells are working harder to maintain euglycemia. This compensatory hyperinsulinemia is the hallmark "silent phase" of insulin resistance. A static glucose of 90 mg/dL looks reassuring; the trajectory pairing of "flat glucose + rising insulin" unmasks the pathology.

ALT is trending from 18 → 35 U/L. While still within the standard range (<40), it has crossed the longevity-optimal threshold (<20 U/L) and the trajectory mirrors the insulin rise — consistent with early hepatic steatosis driven by the developing insulin resistance. This is confirmatory evidence, not the primary signal.

The fasting insulin trajectory is most concerning because (1) it has the highest proportionate rate of change (nearly 2× over 18 months), (2) it represents the earliest and most sensitive marker of the insulin resistance cascade — by the time glucose rises meaningfully, beta cell decompensation has already begun, (3) the compensatory pattern (rising insulin + flat glucose) is the highest-yield early intervention window, and (4) at current trajectory, this patient will likely reach HOMA-IR >3.0 within 6 months, entering frank insulin resistance with downstream risks for NAFLD progression, cardiovascular disease, and accelerated biological aging through chronic mTOR hyperactivation.""",
    answer="A. Rising fasting insulin trajectory",
    metadata={
        "pack": "P5", "pack_name": "prism-trajectory", "idx": 2,
        "content_hash": make_hash("46M insulin trajectory"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

P5.append(make_entry(
    question="""A 53-year-old female has serial labs over 24 months:

| Marker | Month 0 | Month 8 | Month 16 | Month 24 |
| --- | --- | --- | --- | --- |
| TSH (mIU/L) | 1.8 | 2.4 | 3.1 | 3.9 |
| Vitamin D (ng/mL) | 45 | 38 | 32 | 28 |

What is the most concerning trend in this patient's longitudinal data?

A. Rising TSH trajectory
B. Declining Vitamin D trend
C. TSH approaching upper limit
D. Vitamin D crossing below standard threshold""",
    reasoning="""Rate-of-change analysis reveals two concurrent deteriorating trajectories that may be mechanistically linked. TSH is rising from 1.8 → 2.4 → 3.1 → 3.9 mIU/L over 24 months, a rate of change of +2.1 mIU/L over 24 months = +1.05 mIU/L/year. This trajectory has crossed from well within the longevity-optimal range (0.5–2.5 mIU/L at Month 0) to nearly exceeding the standard range ceiling (4.0 mIU/L at Month 24). The monotonic rise with no reversal suggests a progressive pathological process rather than physiological variation.

Vitamin D is declining from 45 → 38 → 32 → 28 ng/mL, a rate of change of -17 ng/mL over 24 months = -8.5 ng/mL/year. This trajectory has crossed from the longevity-optimal range (40–60 ng/mL) to below the standard threshold (30 ng/mL). However, vitamin D decline may be secondary to reduced outdoor activity, seasonal variation, or reduced supplementation compliance.

The rising TSH trajectory is most concerning because (1) TSH has the steeper clinically significant trajectory — approaching the standard range ceiling at Month 24 suggests evolving subclinical hypothyroidism, likely autoimmune thyroiditis (TPO antibodies should be checked immediately), (2) this trajectory pattern carries a 4.3% annual risk of progression to overt hypothyroidism, (3) the combination of rising TSH + falling vitamin D may be synergistic — vitamin D deficiency has been independently associated with increased TPO antibody titers and Hashimoto's progression, and (4) thyroid dysfunction at this trajectory rate will impact multiple downstream systems: lipid metabolism (rising LDL), energy expenditure (weight gain), cognitive function, and cardiovascular risk.

Comparing trajectory vs absolute value: TSH at 3.9 as a static value might prompt "watch and wait." But TSH at 3.9 on a trajectory from 1.8 at +1.05/year demands immediate investigation because extrapolation suggests it will exceed 5.0 mIU/L within one year, entering overt subclinical hypothyroidism requiring treatment.""",
    answer="A. Rising TSH trajectory",
    metadata={
        "pack": "P5", "pack_name": "prism-trajectory", "idx": 3,
        "content_hash": make_hash("53F TSH trajectory"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

# ──────────────────────────────────────────────────────────────────
# P6: ROUTING (8 examples — 3 firewall, 3 default, 2 geriatric)
# ──────────────────────────────────────────────────────────────────

P6 = []

# ACUTE — firewall active
P6.append(make_entry(
    question="""A 45-year-old male presents to the emergency department with sudden onset chest pain, diaphoresis, and ST elevation in leads II, III, and aVF. Troponin I is 8.2 ng/mL. What is the most appropriate immediate intervention?

A. Aspirin + clopidogrel + heparin + emergent PCI
B. Oral metoprolol and observation
C. CT angiography of the chest
D. Fibrinolytic therapy with alteplase

Answer:""",
    reasoning="""Route: ACUTE. This presentation — sudden chest pain, diaphoresis, and inferior ST elevation (leads II, III, aVF) with markedly elevated troponin I (8.2 ng/mL, normal <0.04) — is diagnostic of acute ST-elevation myocardial infarction (STEMI). This is a time-sensitive emergency where every minute of delay increases myocardial necrosis ("time is muscle").

The pathophysiology is acute thrombotic occlusion of the right coronary artery (given inferior lead involvement), causing transmural ischemia. The standard of care per AHA/ACC STEMI guidelines is dual antiplatelet therapy (aspirin 325 mg chewed + P2Y12 inhibitor) plus anticoagulation (unfractionated heparin) followed by emergent percutaneous coronary intervention (PCI) within 90 minutes of first medical contact (door-to-balloon time).

Option A provides the complete evidence-based STEMI protocol: aspirin inhibits COX-1 → blocks TXA2-mediated platelet aggregation; clopidogrel (or prasugrel/ticagrelor) blocks P2Y12 ADP receptors on platelets; heparin prevents clot propagation; and PCI achieves mechanical reperfusion with balloon angioplasty ± stent placement.

Option B (oral metoprolol and observation) is inappropriate as sole therapy — while beta-blockers reduce myocardial oxygen demand and are part of STEMI management, they are adjunctive, not primary. Option C (CT angiography) would delay reperfusion and is used for evaluating aortic dissection or pulmonary embolism, not STEMI with diagnostic ECG. Option D (fibrinolytic therapy) is second-line when PCI cannot be performed within 120 minutes of first medical contact — PCI is preferred when available as it achieves superior TIMI 3 flow rates (>95% vs ~55%).""",
    answer="A. Aspirin + clopidogrel + heparin + emergent PCI",
    metadata={
        "pack": "P6", "pack_name": "prism-routing", "idx": 0,
        "content_hash": make_hash("45M STEMI acute"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "route": "ACUTE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

# PEDIATRIC — firewall active
P6.append(make_entry(
    question="""A 4-year-old boy presents with a 3-day history of fever (39.5°C), bilateral non-exudative conjunctival injection, strawberry tongue, and a polymorphous rash on his trunk. His hands and feet show erythema and edema. What is the most likely diagnosis?

A. Kawasaki disease
B. Scarlet fever
C. Measles
D. Stevens-Johnson syndrome

Answer:""",
    reasoning="""Route: PEDIATRIC. This presentation in a 4-year-old boy meets classic criteria for Kawasaki disease (KD), a medium-vessel vasculitis of childhood. The American Heart Association diagnostic criteria require fever ≥5 days (this child has 3 days but the criteria have been met with 4/5 clinical features present, allowing "incomplete" KD diagnosis):

1. Bilateral non-exudative conjunctival injection — present
2. Changes in lips/oral mucosa (strawberry tongue) — present
3. Polymorphous rash — present
4. Changes in extremities (erythema/edema of hands and feet) — present
5. Cervical lymphadenopathy — not mentioned but 4/5 features are present

Kawasaki disease is the leading cause of acquired heart disease in children in developed countries. The primary concern is coronary artery aneurysm development, which occurs in approximately 25% of untreated cases but drops to <5% with timely treatment. The pathophysiology involves immune-mediated inflammation of the coronary arterial wall: CD8+ T cells and macrophages infiltrate all three arterial layers → elastic lamina destruction → aneurysm formation.

Treatment is time-sensitive: intravenous immunoglobulin (IVIG 2 g/kg as single infusion) plus high-dose aspirin (80-100 mg/kg/day until afebrile, then 3-5 mg/kg/day for 6-8 weeks). Echocardiography should be obtained at diagnosis, 2 weeks, and 6-8 weeks to assess coronary artery dimensions.

Scarlet fever (B) presents with sandpaper-like rash and pharyngitis with GAS-positive culture, but does not cause conjunctival injection or extremity changes. Measles (C) has cough, coryza, conjunctivitis, and Koplik spots followed by cephalocaudal rash, but does not cause extremity erythema/edema. Stevens-Johnson syndrome (D) involves mucosal erosions and targetoid lesions, typically drug-related, and presents with skin detachment rather than erythema/edema.""",
    answer="A. Kawasaki disease",
    metadata={
        "pack": "P6", "pack_name": "prism-routing", "idx": 1,
        "content_hash": make_hash("4M kawasaki peds"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "route": "PEDIATRIC",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

# PREGNANCY — firewall active
P6.append(make_entry(
    question="""A 32-year-old woman at 28 weeks' gestation presents with blood pressure 165/105 mmHg, proteinuria 3+, headache, and visual disturbances. Platelets are 95,000/µL, AST 142 U/L. What is the most appropriate management?

A. Magnesium sulfate + immediate delivery
B. Oral nifedipine and outpatient monitoring
C. Expectant management until 37 weeks
D. Labetalol + bed rest

Answer:""",
    reasoning="""Route: PREGNANCY. This is severe preeclampsia with features of HELLP syndrome (Hemolysis, Elevated Liver enzymes, Low Platelets). The diagnostic criteria for severe preeclampsia are met: blood pressure ≥160/110 on two occasions, significant proteinuria (3+), and concerning features including headache and visual disturbances (indicating cerebral edema/vasospasm risk). The thrombocytopenia (95,000/µL, normal >150,000) and markedly elevated AST (142 U/L) indicate HELLP syndrome progression.

At 28 weeks' gestation, the management must balance maternal safety against fetal prematurity. However, severe preeclampsia with HELLP syndrome mandates delivery regardless of gestational age because the condition can rapidly progress to eclampsia (seizures), placental abruption, hepatic rupture, DIC, or maternal death. ACOG Practice Bulletin 222 recommends delivery for severe preeclampsia at <34 weeks when there are unstable maternal or fetal conditions.

The correct management is (A): magnesium sulfate for seizure prophylaxis (loading dose 4-6g IV over 20 minutes, then 1-2g/hour maintenance) plus planning for delivery. Corticosteroids (betamethasone 12 mg × 2 doses 24h apart) should be administered for fetal lung maturity if time permits, but delivery should not be delayed beyond 24-48 hours for steroid benefit in the setting of HELLP syndrome. Mode of delivery depends on cervical favorability and fetal presentation — if cervix is unfavorable, cesarean delivery may be indicated.

Option B (oral nifedipine outpatient) is inappropriate — this patient requires inpatient management with IV magnesium. Option C (expectant to 37 weeks) is contraindicated with HELLP syndrome. Option D (labetalol + bed rest) is insufficient — while labetalol is appropriate for acute blood pressure management, it does not address the seizure risk or the need for delivery.""",
    answer="A. Magnesium sulfate + immediate delivery",
    metadata={
        "pack": "P6", "pack_name": "prism-routing", "idx": 2,
        "content_hash": make_hash("32F preeclampsia pregnancy"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "route": "PREGNANCY",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

# DEFAULT — longevity lens applied
P6.append(make_entry(
    question="""A 55-year-old male with BMI 28, fasting glucose 102 mg/dL, TG/HDL ratio 3.8, and family history of type 2 diabetes presents for routine evaluation. What is the most important next step?

A. Fasting insulin and HOMA-IR calculation
B. HbA1c only
C. Oral glucose tolerance test
D. Referral to endocrinology

Answer:""",
    reasoning="""Route: DEFAULT. This patient presents with multiple early metabolic warning signs that standard care might underweight. Applying the PRISM dual-threshold framework:

Fasting glucose at 102 mg/dL is classified as "impaired fasting glucose" by standard criteria (100-125 mg/dL) but is significantly above the longevity-optimal range of 72–85 mg/dL — a value >85 represents an early insulin resistance signal. The TG/HDL ratio of 3.8 exceeds both the standard threshold (<3.5) and the longevity-optimal target (<1.0), serving as a validated surrogate for insulin resistance. Combined with BMI 28 and family history of T2DM, this triggers the PRISM Insulin Resistance Cascade constellation.

The most important next step is (A): obtaining fasting insulin and calculating HOMA-IR. This is the key insight from the longevity-optimal approach — while HbA1c (B) provides a 3-month glycemic average, it misses the early compensatory hyperinsulinemia phase where the pancreas maintains near-normal glucose levels through excessive insulin secretion. Fasting insulin is the earliest detectable marker in the insulin resistance trajectory, and HOMA-IR (insulin × glucose / 405) quantifies the degree of resistance.

If this patient's fasting insulin returns at, for example, 18 mIU/L (standard range <25 but far above optimal <7), HOMA-IR would be 4.5 (optimal <1.0), confirming significant insulin resistance that is currently being masked by pancreatic compensation. This early identification opens the therapeutic window for lifestyle intervention ± metformin before beta-cell exhaustion leads to frank diabetes.

The oral glucose tolerance test (C) provides complementary information but is less specific for the underlying insulin resistance mechanism. Direct endocrinology referral (D) is premature without first characterizing the metabolic phenotype with basic insulin metrics. HbA1c alone (B) would likely return 5.6-5.8% — mildly elevated but not actionable in isolation.

From a hallmarks-of-aging perspective, hyperinsulinemia drives deregulated nutrient sensing through chronic mTOR hyperactivation, suppresses autophagy, and promotes inflammaging through visceral adipose tissue macrophage polarization.""",
    answer="A. Fasting insulin and HOMA-IR calculation",
    metadata={
        "pack": "P6", "pack_name": "prism-routing", "idx": 3,
        "content_hash": make_hash("55M default metabolic"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "route": "DEFAULT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

# DEFAULT — longevity lens applied
P6.append(make_entry(
    question="""A 48-year-old female with isolated elevated Lp(a) of 92 nmol/L, otherwise normal lipid panel (LDL-C 88, ApoB 72), no family history of ASCVD. What is the most appropriate clinical approach?

A. No intervention needed — LDL-C is normal
B. Aggressive ApoB reduction below 60 mg/dL
C. Aspirin for primary prevention
D. Repeat Lp(a) in 6 months

Answer:""",
    reasoning="""Route: DEFAULT. This patient presents with a seemingly paradoxical lipid profile: LDL-C at 88 mg/dL and ApoB at 72 mg/dL both appear near the longevity-optimal targets (<70 mg/dL for both). However, Lp(a) at 92 nmol/L dramatically exceeds both the standard threshold (<75 nmol/L) and the longevity-optimal target (<30 nmol/L).

Applying PRISM dual-threshold interpretation: Lp(a) is a genetically determined, largely non-modifiable risk factor that carries Tier A evidence as an independent cardiovascular risk marker. Lp(a) particles are LDL-like lipoproteins with an additional apolipoprotein(a) molecule that confers three pathological mechanisms: (1) atherogenic — Lp(a) penetrates the arterial intima and is retained by extracellular matrix, (2) pro-thrombotic — apo(a) has structural homology to plasminogen and competitively inhibits fibrinolysis, (3) pro-inflammatory — Lp(a) carries oxidized phospholipids that activate innate immunity via TLR2/TLR4 signaling.

The correct approach is (B): aggressive ApoB reduction below 60 mg/dL. Per ESC 2024 guidance, when Lp(a) is elevated (>50 nmol/L), the modifiable atherogenic risk (primarily ApoB-containing particles) should be driven to the lowest achievable level to compensate for the non-modifiable Lp(a) risk. The PRISM framework specifies ApoB <60 mg/dL when Lp(a) >50 or CAC >0. This patient's current ApoB of 72 needs to decrease by ~17%.

Option A is incorrect — dismissing this patient because LDL-C is "normal" ignores the Lp(a)-driven residual risk that standard lipid panels do not capture. Option C (aspirin) lacks net benefit in primary prevention for this risk profile and does not address the underlying atherogenic burden. Option D (repeat Lp(a)) is unnecessary — Lp(a) is genetically determined and does not meaningfully change over time; a single measurement is sufficient for lifetime risk stratification.

The longevity perspective transforms this from "normal lipids, no intervention" to "high residual risk requiring compensatory reduction of modifiable atherogenic particles." This represents the hallmark of genomic instability — a heritable variant driving accelerated vascular aging that requires proactive management.""",
    answer="B. Aggressive ApoB reduction below 60 mg/dL",
    metadata={
        "pack": "P6", "pack_name": "prism-routing", "idx": 4,
        "content_hash": make_hash("48F Lpa default"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "route": "DEFAULT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

# DEFAULT — longevity lens applied
P6.append(make_entry(
    question="""A 42-year-old male with no medical history presents for a health optimization assessment. Labs: HbA1c 5.0%, fasting insulin 3.2 mIU/L, ApoB 62 mg/dL, hsCRP 0.3 mg/L, eGFR 105 mL/min. What is the most appropriate recommendation?

A. No changes — all values are optimal
B. Start a statin for primary prevention
C. Functional fitness assessment (VO2 max, grip strength, body composition)
D. Repeat labs in 12 months

Answer:""",
    reasoning="""Route: DEFAULT. This is a rare clinical presentation — a patient whose biomarkers are uniformly within or near the longevity-optimal ranges. HbA1c 5.0% (optimal <5.3%), fasting insulin 3.2 mIU/L (optimal <7), ApoB 62 mg/dL (optimal <70), hsCRP 0.3 mg/L (optimal <0.5), eGFR 105 mL/min (optimal >90). No constellation patterns are triggered. No dual-threshold concerns are identified.

When the blood biomarker panel is optimized, the PRISM framework directs attention to functional biomarkers — which carry the strongest mortality prediction evidence (Tier A) and represent dimensions of health that blood tests cannot capture. The correct next step is (C): comprehensive functional fitness assessment including VO2 max, grip strength, and body composition (DXA for ALMI and VAT mass).

VO2 max is the single strongest predictor of all-cause mortality — individuals below the 25th percentile for age/sex have a 4× higher mortality risk compared to those above the 75th percentile. This dwarfs the predictive power of any individual blood biomarker. Grip strength is an independent mortality predictor (optimal >40 kg for males), reflecting neuromuscular reserve and sarcopenia risk. ALMI (appendicular lean mass index via DXA, optimal >7.5 kg/m² for males) quantifies muscle mass, while visceral adipose tissue mass (optimal <500g) captures metabolic risk invisible to BMI.

Option A ("no changes") is incomplete — optimal blood markers do not guarantee optimal functional health or body composition. A metabolically healthy patient could still have low VO2 max, inadequate muscle mass, or excessive visceral fat. Option B (statin) is inappropriate with ApoB already at 62. Option D (repeat in 12 months) misses the opportunity to establish functional baseline measurements that become increasingly important for tracking biological aging trajectory.

From a longevity perspective, functional markers represent the hallmarks of stem cell exhaustion (muscle regenerative capacity) and mitochondrial dysfunction (aerobic capacity), and provide the most actionable targets for exercise prescription.""",
    answer="C. Functional fitness assessment (VO2 max, grip strength, body composition)",
    metadata={
        "pack": "P6", "pack_name": "prism-routing", "idx": 5,
        "content_hash": make_hash("42M optimal functional"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "route": "DEFAULT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

# GERIATRIC — longevity lens with frailty considerations
P6.append(make_entry(
    question="""An 82-year-old female with mild cognitive impairment, gait speed 0.7 m/s, grip strength 16 kg, HbA1c 6.8%, and polypharmacy (12 medications) presents for geriatric assessment. What is the most appropriate priority?

A. Intensify diabetes control to HbA1c <6.5%
B. Deprescribing review with focus on anticholinergic and fall-risk medications
C. Add rapamycin for longevity optimization
D. Increase caloric restriction for metabolic improvement

Answer:""",
    reasoning="""Route: GERIATRIC. This patient is frail by objective criteria: gait speed 0.7 m/s is below the frailty threshold (<0.8 m/s per Fried criteria), grip strength 16 kg is well below optimal (>25 kg for females), and she has mild cognitive impairment. The PRISM framework for geriatric routing applies the longevity lens with critical frailty adjustments — interventions must be evaluated through the lens of functional preservation rather than biomarker optimization.

The most appropriate priority is (B): deprescribing review. Polypharmacy with 12 medications in a frail 82-year-old carries substantial iatrogenic risk. Using the Beers Criteria and STOPP/START framework, the immediate focus should be identifying and discontinuing anticholinergic medications (which worsen cognitive impairment through central muscarinic antagonism — accelerating the hallmark of altered intercellular communication in the CNS), medications associated with fall risk (benzodiazepines, first-generation antihistamines, alpha-blockers, long-acting sulfonylureas), and medications with diminishing benefit-to-risk ratio at advanced age.

From a longevity-optimal perspective, the HbA1c of 6.8% in an 82-year-old with frailty should NOT be intensified (A). ADA guidelines recommend HbA1c <8.0-8.5% for frail elderly patients — aggressive glycemic control increases hypoglycemia risk, which in patients with gait impairment (0.7 m/s) and cognitive decline can cause falls, fractures, and hospitalization. The longevity-optimal HbA1c target of <5.3% applies to robust adults where the risk-benefit favors prevention; in frailty, it reverses.

Rapamycin (C) is inappropriate — immunosuppression in an elderly frail patient with polypharmacy introduces infection risk that outweighs theoretical geroprotective benefit. Caloric restriction (D) would worsen sarcopenia in a patient already below grip strength thresholds — this patient needs INCREASED protein intake (1.2-1.5 g/kg/day) and resistance training, not restriction.

The geriatric route prioritizes: (1) reduce iatrogenic harm (deprescribing), (2) prevent catastrophic events (falls, fractures), (3) preserve function (gait, cognition), (4) then consider optimization.""",
    answer="B. Deprescribing review with focus on anticholinergic and fall-risk medications",
    metadata={
        "pack": "P6", "pack_name": "prism-routing", "idx": 6,
        "content_hash": make_hash("82F geriatric deprescribing"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "route": "GERIATRIC",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

# GERIATRIC
P6.append(make_entry(
    question="""A 75-year-old male with well-controlled hypertension, eGFR 68, ApoB 95, and VO2 max at the 60th percentile for his age presents for longevity assessment. What optimization should be prioritized?

A. Aggressive ApoB reduction to <70 mg/dL
B. VO2 max improvement through structured aerobic training
C. eGFR monitoring only
D. No intervention — metrics are age-appropriate

Answer:""",
    reasoning="""Route: GERIATRIC. This patient represents the "robust elderly" phenotype — well-controlled hypertension, functional independence implied by VO2 max at the 60th percentile, and no frailty markers. The PRISM geriatric route for robust elderly patients applies the longevity lens with age-appropriate targets.

The priority is (A): ApoB reduction to <70 mg/dL. ApoB at 95 mg/dL exceeds the longevity-optimal target (<70) by 36%. Given the patient's age (75), every year of residual atherogenic exposure contributes to coronary and cerebrovascular plaque burden. The cumulative "ApoB-years" concept means that even modest reductions in ApoB at this age yield meaningful absolute risk reduction because baseline event rates are higher. This is supported by Tier A evidence from statin and PCSK9 inhibitor trials demonstrating benefit even in patients >75 years (PROSPER, STAREE).

The eGFR at 68 mL/min (optimal >90) represents mild-moderate renal decline — below optimal but above the CKD Stage 3 threshold of 60. Monitoring is appropriate but not the highest priority action. Importantly, statin therapy for ApoB reduction is NOT contraindicated at this eGFR level.

VO2 max at the 60th percentile is good but not yet at the longevity-optimal target (>75th percentile). Structured aerobic training (B) would be valuable as a secondary priority — improvement from 60th to 75th percentile at age 75 confers significant mortality reduction. However, ApoB reduction is prioritized because (1) it has a higher evidence tier for mortality reduction at this age, (2) pharmacological intervention achieves results within weeks vs months for exercise adaptation, and (3) both can be pursued simultaneously.

Option D ("age-appropriate") reflects the standard-care complacency that the longevity lens is designed to overcome. An ApoB of 95 at age 75 is statistically "normal" but represents ongoing atherogenic damage. The longevity-optimal framework rejects age-adjusted complacency for modifiable risk factors — there is no age at which elevated ApoB becomes acceptable if the patient is otherwise robust enough to benefit from treatment.""",
    answer="A. Aggressive ApoB reduction to <70 mg/dL",
    metadata={
        "pack": "P6", "pack_name": "prism-routing", "idx": 7,
        "content_hash": make_hash("75M geriatric ApoB"),
        "generator": "claude-opus-4-6", "quality_issues": [],
        "route": "GERIATRIC",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
))

# ──────────────────────────────────────────────────────────────────
# SAVE ALL PACKS
# ──────────────────────────────────────────────────────────────────

print("Saving test examples...")
save_pack("P1_biomarker", P1)
save_pack("P4_repurposing", P4)
save_pack("P5_trajectory", P5)
save_pack("P6_routing", P6)

total = len(P1) + len(P4) + len(P5) + len(P6)
print(f"\nTotal: {total} examples across 4 packs (P2/P3 require MedReason filtering, skipped for now)")
print("\nRun validation: python3 scripts/generate_skill_packs.py --stats P1")
