#!/usr/bin/env python3
"""Prepare 100 stratified questions into batches for PRISM v3 distillation."""
import json, random, re, os

random.seed(42)

# Load source dataset
with open('/Users/biobook/datasets/medical-o1-reasoning-SFT/medical_o1_sft.json') as f:
    data = json.load(f)

# Load v3 system prompt
with open('/Users/biobook/datasets/prism-framework/PRISM-v3-compact-system-prompt.md') as f:
    system_prompt = f.read()

# Simple route classifier based on keywords in question text
def classify_route(q):
    ql = q.lower()
    acute_kw = ['emergency', 'trauma', 'sepsis', 'shock', 'cardiac arrest', 'cpr', 'intubat',
                 'hemorrhag', 'anaphyla', 'dka', 'status epilepticus', 'stroke', 'stemi',
                 'unstable', 'resuscitat', 'icu', 'critical', 'acute abdomen', 'gi bleed']
    preg_kw = ['pregnan', 'trimester', 'obstetric', 'postpartum', 'eclampsia', 'ectopic',
               'gestational', 'labor', 'delivery', 'fetal', 'neonatal', 'newborn']
    peds_kw = ['child', 'infant', 'pediatric', 'neonate', 'boy', 'girl', 'year-old boy',
               'year-old girl', 'month-old', 'toddler', 'adolescent']
    psych_kw = ['psychiatr', 'suicid', 'psychosis', 'schizophren', 'bipolar', 'depression',
                'anxiety disorder', 'hallucination', 'delusion', 'substance abuse', 'overdose']
    geri_kw = ['elderly', 'geriatric', 'nursing home', '85-year', '90-year', 'dementia',
               'polypharmacy', 'fall risk', 'frail']
    mech_kw = ['mechanism of action', 'pathophysiology', 'molecular', 'signaling pathway',
               'receptor', 'pharmacology', 'biochem', 'how does', 'explain the mechanism']

    # Check age for pediatric
    age_match = re.search(r'(\d+)[\s-]*(year|month|week|day)[\s-]*old', ql)
    if age_match:
        age_val = int(age_match.group(1))
        unit = age_match.group(2)
        if unit in ('month', 'week', 'day') or (unit == 'year' and age_val < 18):
            if any(k in ql for k in peds_kw) or age_val < 18:
                return 'PEDIATRIC'

    if any(k in ql for k in acute_kw): return 'ACUTE'
    if any(k in ql for k in preg_kw): return 'PREGNANCY'
    if any(k in ql for k in psych_kw): return 'PSYCHIATRY'
    if any(k in ql for k in geri_kw): return 'GERIATRIC'
    if any(k in ql for k in mech_kw): return 'BASIC_SCIENCE'

    # MCQ detection
    has_options = bool(re.search(r'\b[A-E]\)', q) or re.search(r'\b[A-E]\.', q))
    if has_options: return 'MCQ'

    return 'DEFAULT'

# Classify all questions
classified = []
for i, entry in enumerate(data):
    route = classify_route(entry['Question'])
    classified.append((i, route, entry))

# Stratified sampling: target distribution
# 60% diagnosis/management (DEFAULT+MCQ), 15% labs/preventive, 10% acute, 5% peds, 5% other, 5% basic science
route_pools = {}
for idx, route, entry in classified:
    route_pools.setdefault(route, []).append((idx, entry))

print("Route distribution in source dataset:")
for route, items in sorted(route_pools.items(), key=lambda x: -len(x[1])):
    print(f"  {route}: {len(items)}")

# Sample targets
targets = {
    'DEFAULT': 25,
    'MCQ': 35,
    'ACUTE': 10,
    'PEDIATRIC': 8,
    'PREGNANCY': 5,
    'PSYCHIATRY': 5,
    'GERIATRIC': 4,
    'BASIC_SCIENCE': 8,
}

selected = []
for route, count in targets.items():
    pool = route_pools.get(route, [])
    if len(pool) < count:
        print(f"  WARNING: {route} has only {len(pool)}, taking all")
        chosen = pool
    else:
        chosen = random.sample(pool, count)
    for idx, entry in chosen:
        selected.append({'index': idx, 'route': route, 'question': entry['Question'],
                         'original_cot': entry.get('Complex_CoT', ''),
                         'original_response': entry.get('Response', '')})

random.shuffle(selected)
print(f"\nTotal selected: {len(selected)}")

# Split into batches of 5
BATCH_SIZE = 5
batches = [selected[i:i+BATCH_SIZE] for i in range(0, len(selected), BATCH_SIZE)]
print(f"Batches: {len(batches)}")

# Create output directory
out_dir = '/tmp/prism-v3-distill'
os.makedirs(out_dir, exist_ok=True)

# Save the selected questions for later parsing
with open(f'{out_dir}/selected_questions.json', 'w') as f:
    json.dump(selected, f, indent=2, ensure_ascii=False)

# Build prompt files
for batch_idx, batch in enumerate(batches):
    prompt = f"""SYSTEM PROMPT — follow this framework exactly:
---
{system_prompt}
---

Generate PRISM v3 reasoning for each question below. Follow the framework structure exactly:
1. Complete the reasoning chain (Steps 1-4)
2. End each question with the structured Output block
3. Use tool_call/tool_result format when calculations are needed
4. Assign the correct route and depth level
5. Do NOT skip the Output schema block — it is mandatory

"""
    for q_idx, item in enumerate(batch):
        prompt += f"### Question {batch_idx * BATCH_SIZE + q_idx + 1}\n{item['question']}\n\n"

    prompt_file = f'{out_dir}/batch_{batch_idx:02d}.md'
    with open(prompt_file, 'w') as f:
        f.write(prompt)

print(f"\nPrompt files written to {out_dir}/batch_00.md through batch_{len(batches)-1:02d}.md")

# Generate the runner script
runner = f"""#!/bin/bash
# PRISM v3 Distillation Runner — {len(batches)} batches, parallel execution
set -e

OUT_DIR="{out_dir}"
DONE=0
FAIL=0
TOTAL={len(batches)}

run_batch() {{
    local BATCH_NUM=$1
    local BATCH_FILE="$OUT_DIR/batch_$(printf '%02d' $BATCH_NUM).md"
    local OUTPUT_FILE="$OUT_DIR/output_$(printf '%02d' $BATCH_NUM).md"
    local ERROR_FILE="$OUT_DIR/error_$(printf '%02d' $BATCH_NUM).log"

    codex exec -m gpt-5.4 \\
        --skip-git-repo-check \\
        --sandbox read-only \\
        --ephemeral \\
        -c model_reasoning_effort="xhigh" \\
        -o "$OUTPUT_FILE" \\
        - < "$BATCH_FILE" \\
        2>"$ERROR_FILE"

    if [ -s "$OUTPUT_FILE" ]; then
        echo "DONE batch $BATCH_NUM ($(wc -l < "$OUTPUT_FILE") lines)"
    else
        echo "FAIL batch $BATCH_NUM — see $ERROR_FILE"
    fi
}}

echo "Starting {len(batches)} batches in parallel (10 workers)..."
echo ""

# Run in waves of 10
for WAVE_START in $(seq 0 10 $(({len(batches)} - 1))); do
    WAVE_END=$((WAVE_START + 9))
    if [ $WAVE_END -ge {len(batches)} ]; then
        WAVE_END=$(({len(batches)} - 1))
    fi

    echo "=== Wave: batches $WAVE_START-$WAVE_END ==="

    for i in $(seq $WAVE_START $WAVE_END); do
        run_batch $i &
    done

    wait
    echo "=== Wave complete ==="
    echo ""
done

echo "All batches complete. Outputs in $OUT_DIR/output_*.md"

# Quick summary
DONE=$(ls -1 "$OUT_DIR"/output_*.md 2>/dev/null | while read f; do [ -s "$f" ] && echo 1; done | wc -l | tr -d ' ')
echo "Successful: $DONE / $TOTAL"
"""

with open(f'{out_dir}/run_distill.sh', 'w') as f:
    f.write(runner)
os.chmod(f'{out_dir}/run_distill.sh', 0o755)

print(f"Runner script: {out_dir}/run_distill.sh")
