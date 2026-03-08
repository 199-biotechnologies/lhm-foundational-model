# Tool Usage Research Synthesis for PRISM v3

## Key Papers and Frameworks

### 1. AgentMD (NCBI/NLH, Nature Communications 2025, 19 citations)
- Curated 2,164 executable clinical calculators from PubMed literature
- Autonomous tool selection + slot filling from clinical text
- >85% accuracy, >90% pass rate on quality checks
- Key insight: LLMs are poor at arithmetic but excellent at recognizing WHEN a calculator is needed
- Tool format: Python functions with docstrings (e.g., compute_curb65())
- GitHub: ncbi-nlp/clinical-tool-learning

### 2. MeNTi (NAACL 2025, 17 citations)
- **Nested tool calling** — a tool can call other tools (e.g., BMI calculator calls unit converter)
- **Meta-tool mechanism** — LLM first calls a meta-tool to select the right calculator
- CalcQA benchmark for medical calculator tasks
- Key insight: Medical calculations require multi-step tool chains, not single calls
- Architecture: Toolkit → Meta-tool selector → Nested execution → Result synthesis
- Example flow: "Calculate CHA2DS2-VASc" → meta-tool identifies calculator → fills slots from clinical text → calls unit converter if needed → returns score with interpretation

### 3. ReflecTool (ACL 2025, 14 citations)
- **Reflection-aware** tool use — the agent can recognize when a tool call failed or returned unreliable results
- ClinicalAgent Benchmark (CAB): 18 diverse clinical tasks
- Two-stage process: (1) tool selection + execution, (2) reflection on results
- Long-term memory of successful tool usage patterns
- Outperforms pure LLMs by 10+ points on clinical tasks
- Key insight: Tool use without reflection leads to blindly trusting bad outputs

### 4. MedAgentGym (ICLR 2026 Oral, 18 citations)
- 72,000+ task instances across 129 categories
- **Code-based medical reasoning** — agents write Python code to analyze clinical data
- Training via SFT (+36.44%) then RL (+42.47%)
- Executable sandbox environments with deterministic verification
- Key insight: Code generation IS a form of tool use — models that can write calculation code outperform those that try to do math in-context

### 5. Nature Digital Medicine 2025 (29 citations)
- LLM agents CAN use tools to perform clinical calculations accurately
- Without tools: LLMs fail at clinical math (arithmetic errors, wrong formulas)
- With tools: Accuracy jumps dramatically
- Key finding: The bottleneck is not knowing the formula — it's extracting the right values from clinical text and routing to the right calculator

## Tool Categories for Medical Reasoning

### Tier 1: Clinical Calculators (highest impact, most validated)
| Calculator | Formula/Logic | When to Call |
|-----------|--------------|-------------|
| HOMA-IR | insulin × glucose / 405 | Any fasting insulin + glucose pair |
| eGFR (CKD-EPI 2021) | Creatinine-based, optional cystatin C | Any creatinine value |
| ASCVD Risk | Pooled Cohort Equations | Lipids + BP + age + diabetes status |
| CHA2DS2-VASc | Point-based | Atrial fibrillation present |
| MELD-Na | Bilirubin + INR + creatinine + sodium | Liver disease |
| Wells Score (PE) | Point-based clinical criteria | Suspected PE |
| Wells Score (DVT) | Point-based clinical criteria | Suspected DVT |
| CURB-65 | Point-based | Community-acquired pneumonia |
| qSOFA | Point-based | Suspected sepsis |
| Child-Pugh | Point-based | Liver disease severity |
| Corrected Calcium | Ca + 0.8 × (4.0 - albumin) | Low albumin states |
| Anion Gap | Na - (Cl + HCO3) | Metabolic acidosis |
| Osmolar Gap | Measured - Calculated osmolality | Toxic ingestion |
| Free Water Deficit | 0.6 × weight × (Na/140 - 1) | Hypernatremia |
| BMI | weight / height² | Any weight + height |
| BSA | Mosteller: √(height×weight/3600) | Drug dosing |
| Creatinine Clearance | Cockcroft-Gault | Drug dosing in renal impairment |
| Framingham Risk | Multi-variable | CVD risk estimation |
| FIB-4 | Age × AST / (Platelets × √ALT) | Liver fibrosis screening |
| NAFLD Fibrosis Score | Multi-variable | NAFLD staging |
| TG/HDL ratio | TG / HDL | Insulin resistance surrogate |
| ApoB/ApoA1 ratio | ApoB / ApoA1 | Atherogenic risk |
| NLR | Neutrophils / Lymphocytes | Inflammation marker |
| Free Testosterone | Vermeulen equation | Male hormone assessment |

### Tier 2: Reference Lookups (moderate impact)
- Drug interaction checker (given drug list → flag interactions)
- Biomarker reference range lookup (given marker → return standard + optimal range)
- Guideline retrieval (given condition → return current guideline summary)
- PGx lookup (given drug → return relevant pharmacogenes)
- Pregnancy safety lookup (given drug → return FDA category / lactation safety)

### Tier 3: Analytical Tools (advanced)
- Trend analyzer (given longitudinal values → compute rate of change, flag concerning trajectories)
- Pattern detector (given lab panel → check against multi-marker constellation criteria)
- Risk stratifier (given patient profile → compute composite risk scores)

## Training Data Format for Tool Use

### Option A: Inline Tool Calls (ChatML-style)
```
<|assistant|>
The patient's fasting insulin is 14 mIU/L and fasting glucose is 92 mg/dL. I need to calculate HOMA-IR.

<|tool_call|>
{"name": "calculate_homa_ir", "arguments": {"fasting_insulin": 14, "fasting_glucose": 92}}
<|tool_result|>
{"homa_ir": 3.18, "interpretation": "Insulin resistant (>2.5 standard, >1.0 longevity-optimal)"}
<|assistant|>
HOMA-IR = 3.18. This exceeds both the standard threshold (>2.5) and the longevity-optimal threshold (>1.0), confirming insulin resistance. Mechanistically, this reflects...
```

### Option B: Code Generation (MedAgentGym-style)
```python
# Calculate HOMA-IR
homa_ir = (fasting_insulin * fasting_glucose) / 405
# homa_ir = (14 * 92) / 405 = 3.18
# Interpretation: IR confirmed (standard: >2.5, longevity: >1.0)
```

### Option C: Structured Tool Annotation (AgentMD-style)
```
[TOOL: HOMA-IR(insulin=14, glucose=92) → 3.18]
HOMA-IR of 3.18 confirms insulin resistance...
```

### Recommendation for PRISM v3:
- Use **Option A** for the training data format — it's the most standard and compatible with modern function-calling fine-tuning pipelines (OpenAI, Anthropic, HuggingFace)
- Start with **Tier 1 calculators only** — these are deterministic, verifiable, and have the highest clinical impact
- Add tool calls as a new step between Step 4 (Lab Interpretation) and Step 5 (Convergence)
- The reasoning should show: (1) recognizing the need for a calculation, (2) extracting inputs from clinical text, (3) calling the tool, (4) interpreting the result in clinical context

## Integration into PRISM Reasoning Structure

### Proposed v3 Step 4b: Quantitative Analysis
```
### Step 4b: Quantitative Analysis (when calculable)
When clinical data supports computation:
1. Identify which calculators/scores are relevant
2. Extract input values from the clinical data
3. Call the calculator tool
4. Interpret the result against clinical thresholds
5. Integrate into the reasoning chain

Available tools:
- calculate_homa_ir(insulin, glucose) → HOMA-IR score
- calculate_egfr(creatinine, age, sex, [cystatin_c]) → eGFR
- calculate_ascvd_risk(age, sex, race, total_chol, hdl, sbp, bp_treated, diabetes, smoker) → 10-year risk %
- calculate_cha2ds2_vasc(chf, htn, age, diabetes, stroke_hx, vascular, sex) → score
- calculate_corrected_calcium(calcium, albumin) → corrected Ca
- calculate_anion_gap(sodium, chloride, bicarb) → AG
- calculate_fib4(age, ast, alt, platelets) → FIB-4 score
- lookup_drug_interaction(drug_list) → interaction flags
- lookup_reference_range(marker) → standard + optimal ranges
```

## Key Insights from Research

1. **LLMs + tools > LLMs alone** for any task involving arithmetic, scoring, or reference lookup
2. **Nested tool calling** is important — medical calculations often require intermediate steps (unit conversion, derived values feeding into composite scores)
3. **Reflection on tool results** prevents blindly trusting outputs — the model should sanity-check calculator results against clinical expectations
4. **Code generation** is a viable alternative to structured tool calls for training — MedAgentGym shows RL on code-based tasks yields +42% improvement
5. **For a 0.8B model**, simple tool-call format (Option A) is more realistic than code generation. The model needs to learn WHEN to call a tool and HOW to interpret results, not write complex code.
6. **Start small**: 10-15 core clinical calculators cover 80%+ of clinical calculation needs
