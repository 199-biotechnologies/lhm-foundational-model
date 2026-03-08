PRISM is directionally strong, but it currently tries to do three jobs with one template: diagnostic reasoning, preventive/longevity interpretation, and experimental optimization. At scale, that will systematically bias outputs toward metabolic/longevity anchoring and false precision. The fix is not a rewrite; it is a routing layer plus evidence-tiering.

**1. Reasoning Structure Critique**

- Keep the core 6 steps, but put them inside a front-end router:
  `Step 0: Acuity/Triage`
  `Step 0b: Question-Type Router`
  `Step 0c: Data Sufficiency + Unit/Assay Check`
- `Problem Representation` should always include setting and time course. “45M with fatigue” is weaker than “45M outpatient, subacute fatigue, overweight, afternoon hormone draw, no acute red flags.”
- `Differential Generation` needs a broad etiologic sweep before ranking. Otherwise PRISM will overcall insulin resistance, inflammation, and “subclinical” endocrinology in ordinary cases. Add a VINDICATE/schema-style breadth check before narrowing.
- `Biomarker Interpretation` is useful, but it should be explicitly optional and evidence-tiered. Right now the table presents many contested targets as settled fact.
- `Convergence` should separate:
  `Most likely diagnosis/answer`
  `Immediate next best step`
  `Longevity optimization note`
  These are not the same task.
- `Clinical Implications` should be split into:
  `Standard-of-care action`
  `Specialty-supported optimization`
  `Experimental/longevity-only ideas`
- Add an `Evidence tier` field for every recommendation:
  `Tier A` guideline-backed / strong outcome evidence
  `Tier B` specialty consensus / strong observational support
  `Tier C` exploratory or longevity-extrapolated
- Avoid pseudo-precise probabilities unless the task is explicitly probabilistic. Use coarse bands: `highly likely`, `plausible`, `low but important`.

**2. Edge Cases and Failure Modes**

- `Psychiatry`: the framework is too biomarker-forward. It needs suicide/violence risk, mania/psychosis screen, substance withdrawal, capacity, collateral history, and medication adverse-effect review before any metabolic optimization.
- `Pediatrics`: most cutoffs are invalid. Pediatric reasoning needs age-specific vitals/labs, developmental stage, congenital disease priors, vaccination/infectious context, and growth trajectory.
- `Emergency/acute care`: PRISM currently risks discussing ApoB while missing sepsis, stroke, ACS, ectopic pregnancy, DKA, PE, or compartment syndrome. Acute stabilization must outrank prevention.
- `Rare disease`: pattern-recognition plus hard-coded “common longevity patterns” will suppress zebras. Add a “rare disease trigger” when multisystem findings, family history, unusual age of onset, dysmorphism, or treatment resistance appear.
- `Elderly multi-morbidity`: aggressive optimization can be wrong. You need function, frailty, fall risk, goals of care, life expectancy, polypharmacy, renal reserve, and competing-risk logic.
- `Pregnancy`: normal physiology shifts lipids, ALP, creatinine, hemoglobin, thyroid tests, iron markers, and glucose interpretation. Add trimester-aware ranges and fetal safety gates.
- `Post-surgical complications`: the missing variable is postoperative day plus procedure type. Differentials should branch by leak, bleed, abscess, PE/DVT, ileus, delirium, wound infection, medication effect, or device complication.

**3. Comparison With Existing Frameworks**

- `Peter Attia / Medicine 3.0`: PRISM aligns with proactive prevention, earlier risk detection, and healthspan focus. It is weaker than Attia on exercise, emotional health, and personalized strategy, and stronger on prompt-operational structure. Inference: Attia is a philosophy; PRISM is a reasoning scaffold. Do not turn Attia’s philosophy into fixed lab dogma.
- `Bryan Johnson / Blueprint`: PRISM shares the quantified-self posture and strong pre-analytic hygiene. Blueprint is measurement-heavy and longitudinal, but it is not a diagnostic reasoning framework and should not define universal “optimal” targets for a diverse dataset.
- `Sinclair / Information Theory of Aging`: Sinclair is mechanistic and epigenetic; PRISM is phenotypic and clinical. Add epigenetic-aging metrics as adjuncts, not as the explanatory center of everyday diagnosis.
- `VINDICATE / Illness Scripts / Schema Theory`: PRISM is better on lab nuance and preventive overlays. Standard reasoning frameworks are better at breadth, debiasing, and avoiding early closure. PRISM needs that breadth layer.
- `Med-PaLM`: Med-PaLM 2 added ensemble refinement, retrieval-based claim checking, adversarial evaluation, and physician preference studies. PRISM should copy that verification logic, not just the surface reasoning format.
- `Bredesen / ReCODE`: PRISM is potentially more disciplined than ReCODE because it can enforce uncertainty, evidence tiers, and refusal rules. But without those guardrails it will drift into the same failure mode: multi-factor optimization presented with more confidence than the evidence justifies.

**4. Prompt Engineering for 20K+ Chains**

- Prohibit more aggressively:
  `answer-choice anchoring`
  `post-hoc justification of a chosen option`
  `invented cutoffs or units`
  `causal claims from observational associations`
  `adult cutoffs applied to children/pregnancy`
  `off-label treatment presented as standard care`
  `one-biomarker tunnel vision`
  `pseudo-precision dosing when not asked`
- Add automatic quality checks:
  `unit sanity`
  `sex/age/pregnancy compatibility`
  `acute red-flag screen`
  `must-not-miss screen`
  `counterevidence present for top alternative`
  `probability bands internally consistent`
  `recommendation evidence tier present`
- Handle question types differently:
  `MCQ`: reason from stem first, map to options second.
  `Open-ended diagnosis`: full differential template.
  `Management`: immediate next step + contraindications + monitoring.
  `Lab interpretation`: confounders + pattern recognition + what to repeat.
  `Basic science`: mechanism-first, no faux differential.
- Yes, use separate templates. Minimum set:
  `acute care`
  `adult outpatient preventive`
  `pediatrics`
  `pregnancy/OB`
  `psychiatry`
  `geriatrics`
  `post-op/inpatient`
- Keep the rationale structured and bounded. You want auditable claim-evidence-counterevidence-decision, not long free-form “thinking.”

**5. Prompt-Ready Evidence-Based Additions**

```text
### Biological Age Assessment
Use biological-age metrics only as adjunct risk stratification and longitudinal tracking, never as sole diagnostic evidence.
Preferred hierarchy:
  1. Routine-lab phenotypic age calculators (e.g. Levine/PhenoAge) from standard biomarkers
  2. Epigenetic clocks with published validation (e.g. GrimAge, DunedinPACE, DNAm PhenoAge)
  3. Vendor organ-age reports (e.g. TruAge/Symphony-style outputs) for trend tracking only
Interpretation:
  - Compare biological age to chronological age and, more importantly, to prior trajectory on the same platform
  - DunedinPACE is centered on 1.0 years of biological aging per calendar year; interpret directionally, not as a stand-alone treatment trigger
Guardrails:
  - Do not prescribe drugs solely to improve a clock score
  - Do not compare results across platforms as if interchangeable
  - Repeat only under similar illness, fasting, and collection conditions
```

```text
### Sarcopenia and Frailty
Do not infer muscle health from weight or BMI alone.
Case-finding:
  - Suspect sarcopenia with low strength, slowed gait, recurrent falls, unintentional weight loss, low protein intake, chronic disease, or GLP-1-associated lean-mass concern
Probable sarcopenia:
  - Grip strength <27 kg (men) or <16 kg (women), or 5-chair-stand time >15 sec
Confirmation:
  - DXA appendicular skeletal muscle mass / height^2 <7.0 kg/m^2 (men) or <5.5 kg/m^2 (women)
Severity:
  - Gait speed <=0.8 m/s or poor physical performance indicates severe disease/frailty overlap
Guardrails:
  - Interpret lean mass together with strength and function
  - Obesity can mask sarcopenia
  - Track resistance training, protein adequacy, falls, gait, and body composition over time
```

```text
### Cognitive Decline Risk Markers
Start with phenotype before biomarkers:
  - history, informant report, MoCA/MMSE, depression, sleep apnea, hearing loss, alcohol, anticholinergic burden, vascular risk, delirium screen
Biomarker use:
  - For symptomatic patients or specialty workup, Alzheimer biomarkers may include amyloid PET, CSF Aβ42/40 or p-tau/Aβ ratios, and validated plasma p-tau217 assays
  - Neurofilament light chain (NfL) is a nonspecific neuroaxonal injury marker; it supports disease burden but is not Alzheimer-specific
  - Plasma p-tau217 is currently the highest-performing blood biomarker for AD pathology, but abnormal blood results should usually be confirmed with CSF or PET before high-stakes decisions
Guardrails:
  - Do not recommend Alzheimer biomarker testing for asymptomatic patients outside research
  - Do not use universal p-tau cutoffs across assays
  - Always look for mixed pathology: vascular disease, Lewy body disease, TDP-43, depression, medication effects
```

```text
### Cancer Screening Beyond Standard Guidelines
Guideline-concordant screening always comes first.
MCED / liquid biopsy:
  - Consider only as an add-on shared-decision tool in selected adults after discussing false positives, downstream workup, cost, and uncertain outcome benefit
  - Never replace mammography, cervical screening, colorectal screening, lung screening, or prostate shared decision-making
  - As of March 7, 2026, no MCED assay has shown randomized mortality reduction or has U.S. guideline endorsement for routine population screening
ctDNA:
  - Use for minimal residual disease or recurrence monitoring only when disease-specific evidence supports it
  - Do not use ctDNA as general-population screening
Risk-enhanced screening:
  - Prefer germline-risk, family-history, prior-polyp, dense-breast, smoking, viral-hepatitis, and cirrhosis pathways over vanity screening
```

```text
### Micronutrient Optimization
Use deficiency-oriented, risk-based interpretation. Do not chase blanket supraphysiologic levels.
Vitamin B12:
  - Serum B12 <200-250 pg/mL suggests deficiency
  - If B12 is 150-399 pg/mL, check methylmalonic acid (MMA); MMA >0.271 µmol/L supports deficiency
  - High-risk groups: metformin, PPIs, vegan diet, age, GI surgery, pernicious anemia
Magnesium:
  - Serum Mg <0.75 mmol/L is low, but serum is a weak marker of total-body status
  - Interpret with symptoms, diet, GI loss, diuretics, and PPI exposure; consider adjunct measures if suspicion remains high
Omega-3 index:
  - <4% = low; >8% = favorable cardiovascular-risk profile
  - Prefer food-first or indication-specific EPA therapy over indiscriminate supplement use
Zinc:
  - Serum/plasma zinc 80-120 mcg/dL is typical; <70 mcg/dL in women or <74 mcg/dL in men suggests inadequacy
  - Infection, inflammation, time of draw, and weight loss confound interpretation
Selenium:
  - Routine supplementation is usually unnecessary in selenium-replete populations
  - Avoid chronic high-dose selenium; upper limit is 400 mcg/day
Functional integrator:
  - If homocysteine is elevated, assess B12, folate, B6, renal function, thyroid status, alcohol use, and medications before empiric treatment
```

Comparisons to Attia, Blueprint, and Bredesen are partly inferential because those are public protocols/philosophies rather than formal reasoning frameworks.

**Key sources**
[Peter Attia / Early Medical](https://peterattiamd.com/about/), [Outlive](https://www.penguinrandomhouse.com/books/705161/outlive-by-peter-attia-md-with-bill-gifford/), [Blueprint Biomarkers](https://blueprint.bryanjohnson.com/pages/biomarkers), [Sinclair’s Information Theory of Aging](https://www.nature.com/articles/s43587-023-00527-6), [Med-PaLM 2](https://www.nature.com/articles/s41591-024-03423-7), [EWGSOP2 sarcopenia consensus](https://clinicalnutrition.science/en/nutribib/malnutrition-general-introduction/overview-malnutrition-in-outpatients/sarcopenia-revised-european-consensus-on-definition-and-diagnosi/), [Alzheimer’s Association 2024 criteria](https://www.alz.org/research/for_researchers/diagnostic-criteria-guidelines), [p-tau meta-analysis](https://pubmed.ncbi.nlm.nih.gov/40818474/), [ACS on MCED tests](https://www.cancer.org/cancer/screening/multi-cancer-early-detection-tests.html), [NCI PDQ on MCED](https://www.cancer.gov/about-cancer/screening/hp-screening-overview-pdq), [Levine DNAm PhenoAge](https://www.aging-us.com/article/101414/), [TruAge](https://www.trudiagnostic.com/truage), [NIH ODS B12](https://ods.od.nih.gov/factsheets/VitaminB12-HealthProfessional/), [NIH ODS Magnesium](https://ods.od.nih.gov/factsheets/Magnesium-HealthProfessional/), [NIH ODS Zinc](https://ods.od.nih.gov/factsheets/Zinc-HealthProfessional/), [NIH ODS Selenium](https://ods.od.nih.gov/factsheets/Selenium-HealthProfessional), [Omega-3 Index review](https://pubmed.ncbi.nlm.nih.gov/28511049/), [Bredesen case series](https://pubmed.ncbi.nlm.nih.gov/25324467/), [UCSF critique](https://www.ucsf.edu/news/2020/05/417431/pricey-protocol-not-proven-prevent-or-reverse-alzheimers-says-ucsf-neurologist)

If you want, I can turn this into a revised PRISM v2 system prompt with the routers and evidence-tier rules baked in.