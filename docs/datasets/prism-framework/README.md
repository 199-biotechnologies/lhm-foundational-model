# PRISM Framework v3.0

**Preventive Reasoning with Integrated Systemic Medicine**

A clinical reasoning framework for generating high-quality medical chain-of-thought (CoT) training data, designed to replace the "reasoning theater" found in existing medical SFT datasets.

## Purpose

This framework serves as a **system prompt** for LLMs performing medical reasoning dataset distillation. It enforces structured clinical reasoning, mechanistically-grounded pathophysiology, claim-specific evidence tiering, tool-augmented computation, and calibrated confidence — producing training data that teaches genuine medical reasoning rather than superficial pattern matching.

## Version History

### v3.0 (March 2026)
Major revision based on three-model review (Codex GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6). Key changes:
- **Route-specific reasoning templates** — Real clinical rules for ACUTE (ABC, time-sensitive triage), PREGNANCY (trimester physiology, fetal safety gates), PEDIATRIC (age-specific ranges, weight-based dosing), GERIATRIC (function-first, deprescribing), PSYCHIATRY (safety-first, organic rule-out), POST-OP (POD-based differentials)
- **Dynamic reasoning depth** — 3-level mechanistic budget (L1 physiologic, L2 pathway, L3 molecular) assigned by route
- **Tool-calling integration** — 16 clinical calculators + 5 reference lookups with ChatML-style format, tool reflection rule
- **Claim-specific evidence tiering** — Evidence per-claim, not per-molecule (e.g., SGLT2i: HF/CKD [A], senolysis [C])
- **Evidence corrections** — Fixed misclassified tiers (Lp(a), eGFR, Vitamin D, TSH, ApoB, telmisartan, LDN, SGLT2i senolysis, GLP-1 aging claims)
- **Acute/longevity firewall** — ACUTE, PREGNANCY, PEDIATRIC routes PROHIBITED from longevity overlay
- **Rigid output schema** — Machine-parseable structured block with single-value fields
- **Compressed reasoning** — 4-step core (Route+Problem → Hypotheses+Analysis → Tools+Labs → Convergence); lab interpretation abnormal-only
- **Refusal protocol** — "I need [X] before concluding [Y]" for insufficient data
- Reviews: `sources/codex-v3-review.md`, `sources/gemini-v3-review.md`, `sources/tool-usage-research.md`

### v2.1 (March 2026)
Mechanistic emphasis — 12 hallmarks of aging, molecular pathway tracing, SGLT2i immunosenolysis, GLP-1R aging, NAD+/CD38 axis, epigenetic reprogramming. 2026 integrations: OMICmAge, Life Biosciences FDA trial, updated metformin evidence. 70+ citations.

### v2.0 (March 2026)
Initial three-model synthesis (Claude + Gemini + Codex). Routing layer, dual-threshold biomarkers, pattern recognition, drug repurposing framework.

## Folder Structure

```
prism-framework/
├── README.md                          # This file
├── PRISM-v3-compact-system-prompt.md  # v3 production system prompt for distillation
├── PRISM-v2-system-prompt.md          # v2.1 full reference framework (920 lines)
├── PRISM-compact-system-prompt.md     # v2 compact system prompt (legacy)
├── sources/                           # Model reviews and research
│   ├── claude-draft-v1.md             # Initial Claude draft
│   ├── gemini-review.md               # Gemini v2 review
│   ├── codex-review.md                # Codex v2 review
│   ├── codex-v3-review.md             # Codex v3 review (GPT-5.4)
│   ├── gemini-v3-review.md            # Gemini v3 review (3.1 Pro)
│   └── tool-usage-research.md         # Tool-calling research synthesis
├── references/
│   ├── sources.md                     # All citations (180+)
│   ├── optimal-ranges.md              # Quick-reference biomarker ranges table
│   ├── drug-repurposing.md            # Drug repurposing evidence profiles
│   └── mechanisms.md                  # Molecular pathways reference
└── examples/
    ├── test-batch-a.md                # GPT-5.4 test output (Q1-5)
    ├── test-batch-b.md                # GPT-5.4 test output (Q6-10)
    └── test-mechanistic.md            # GPT-5.4 test output (mechanistic, v2.1)
```

## Key Features (v3)

1. **Route-Specific Reasoning** — Real clinical templates for 9 routes (ACUTE, PREGNANCY, PEDIATRIC, GERIATRIC, PSYCHIATRY, POST_OP, BASIC_SCIENCE, MCQ, DEFAULT) with actual reasoning rules, not just labels

2. **Dynamic Mechanistic Depth** — 3-level budget (L1 physiologic, L2 pathway, L3 molecular) auto-assigned by route with overrides for pharmacology and mechanism questions

3. **Tool-Augmented Reasoning** — 16 clinical calculators (HOMA-IR, eGFR, ASCVD, Wells, MELD-Na, etc.) + 5 reference lookups (drug interactions, pregnancy safety, renal dosing, PGx, guidelines) with ChatML-style tool-calling format

4. **Claim-Specific Evidence Tiering** — Evidence per-claim, not per-molecule. GLP-1 for obesity [A] vs GLP-1 for aging [C]. Separates clinical evidence tier from mechanistic confidence.

5. **Dual-Threshold Biomarkers** — Standard [A] and optimization targets with honest per-claim tiering. Corrected from v2: Vitamin D 40-60 [C] not 50-80 [B]; eGFR >90 [B] not [A]; Lp(a) optimization [B] not [A]

6. **Acute/Longevity Firewall** — ACUTE, PREGNANCY, PEDIATRIC routes PROHIBITED from longevity overlay, hallmark mapping, and optimization tables

7. **Compressed Reasoning** — 4-step core (Route+Problem → Hypotheses+Analysis → Tools+Labs → Convergence). Lab interpretation is abnormal-only and conditional.

8. **Machine-Parseable Output** — Rigid schema with single-value fields for route, depth, confidence, evidence tier, mechanistic confidence. No A/B slashes or compound labels.

9. **Tool Reflection** — Model must sanity-check calculator results against clinical expectations before proceeding (prevents blind trust of bad outputs)

10. **Refusal with Specificity** — "I need [X] before concluding [Y]" instead of silent fabrication

## Usage

Use `PRISM-v3-compact-system-prompt.md` as the system prompt when generating medical reasoning chains for distillation. The v2 files are preserved for reference.

For dataset generation at scale, stratify by route (60-70% diagnosis/management, 15-20% labs/preventive, 10% acute/emergency, 5-10% basic science/pharmacology). Include benign/no-pathology cases and near-miss distractor pairs.

## Evidence Base

See `references/sources.md` for the complete citation list. Evidence tiering follows:
- **[A]** Guideline-backed, RCT-supported (ACC/AHA, ESC/EAS, KDIGO, ACOG, IDSA)
- **[B]** Consensus, observational, Mendelian randomization
- **[C]** Preclinical, experimental, expert opinion
- Non-primary sources (interviews, news, social media) cannot support [A] or [B] tiering
