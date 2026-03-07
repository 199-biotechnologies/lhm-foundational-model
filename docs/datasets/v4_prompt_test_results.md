# V4.1 System Prompt — Test Results

**Date**: 2026-03-07
**Tested by**: Claude Opus 4.6 (5 parallel agents) + Gemini 3.1 Pro review
**Prompt version**: v4.1 (incorporating PRISM v2 critical review + Gemini feedback)

## Test Matrix

| # | Domain | Dataset | GT | Answer Match | Longevity | Factual | Depth | Verdict |
|---|--------|---------|:--:|:---:|:---:|:---:|:---:|:---:|
| 1 | Anatomy | medmcqa | C | C | None (correct) | Pass | Pass | **PASS** |
| 2 | Oncology/Clinical | medqa | B | B | Appropriate | Pass | Pass | **PASS** |
| 3 | Transplant/Immunology | MMLU | B | B | Appropriate | Pass | Pass | **PASS** |
| 4 | Urogynecology/Aging | huatuo | GT match | Match | Appropriate | Pass | Pass | **PASS** |
| 5 | Plant Cell Biology | pubmedqa | A | A | None (correct) | Pass | Pass | **PASS** |

## Guardrail Performance

### 1. Answer Preservation (HARD constraint)
- **5/5 preserved** — No answer drift across any domain
- The "backtrack and re-reason" instruction provides redundancy

### 2. Longevity Lens Calibration
- **Correctly applied** (3/5): Oncology (smoking/screening), Transplant (BK monitoring), Urogynecology (pelvic floor/estrogen)
- **Correctly omitted** (2/5): Anatomy (urogenital diaphragm), Plant Biology (mitochondrial PCD)
- The conditional trigger ("ONLY if chronic disease, metabolic/CV, cancer, aging") works as designed

### 3. Factual Accuracy
- **5/5 accurate** — No errors introduced
- **1 error FIXED**: Example 5 — GPT-4o's original mentioned "caspase cascades" (incorrect for plants). V4.1 correctly uses metacaspases (clan CD, family C14)
- This demonstrates the upgrade CAN improve accuracy, not just verbosity

### 4. Domain Matching
- Anatomy Q got pure structural/fascial reasoning
- Clinical vignettes got pathophysiology + differentials
- Plant biology got molecular signaling only
- No cross-domain contamination

### 5. Evidence Assessment (GRADE-style)
- USPSTF Grade B (lung cancer screening)
- KDIGO moderate-quality evidence (BK monitoring)
- AUA/SUFU moderate-quality evidence (pelvic floor training)
- Naturally integrated, not forced on basic science

## Key Improvements Over v3

| Aspect | v3 | v4.1 |
|--------|----|----|
| Answer safety | Single mention | Triple redundancy (TASK + Guardrail 1 + Anti-pattern) |
| Longevity framing | Always applied | Conditional — domain-appropriate only |
| Domain awareness | None | Mode-matched (anatomy/pharm/clinical/basic science) |
| Evidence quality | Not mentioned | GRADE-style when citing guidelines |
| KG preservation | Implicit | Explicit instruction to preserve and build upon |
| Contradiction handling | None | Resolve by prioritizing most definitive finding |
| Differentials | "2-3 alternatives" (too many for word limit) | "1-2 highly relevant" (focused) |

## Observations

1. **Reasoning length**: Upgraded traces are 300-600 words (vs v3 target of <250). This is acceptable — deeper mechanism requires more words, and longer high-quality CoT is better for fine-tuning.

2. **Error correction**: The v4.1 prompt successfully identifies and fixes errors in GPT-4o reasoning without changing answers. This is a critical capability — we're not just making reasoning longer, we're making it more accurate.

3. **Population specificity**: Where relevant (61yo woman, 62yo smoker), age/sex/comorbidity context is naturally integrated into mechanistic reasoning rather than listed as afterthoughts.

## Recommendation

**V4.1 is ready for pilot deployment.** Run 100-example pilot with GPT-5.4:
```bash
python3 scripts/upgrade_medreason.py --count 100 --batch-size 1 --verify
```

Use `--batch-size 1` (not batched) for initial pilot to ensure clean parsing per example.
