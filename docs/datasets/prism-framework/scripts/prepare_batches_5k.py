#!/usr/bin/env python3
"""Prepare 5000 stratified questions into batches for PRISM v3 distillation."""
import json, random, re, os

random.seed(2026)

# Load source dataset
with open('/Users/biobook/datasets/medical-o1-reasoning-SFT/medical_o1_sft.json') as f:
    data = json.load(f)

# Load v3 system prompt
with open('/Users/biobook/datasets/prism-framework/PRISM-v3-compact-system-prompt.md') as f:
    system_prompt = f.read()

# Exclude the 100 already distilled
with open('/tmp/prism-v3-distill/selected_questions.json') as f:
    already_done = {item['index'] for item in json.load(f)}
print(f"Excluding {len(already_done)} already-distilled questions")

def classify_route(q):
    ql = q.lower()
    acute_kw = ['emergency', 'trauma', 'sepsis', 'shock', 'cardiac arrest', 'cpr', 'intubat',
                 'hemorrhag', 'anaphyla', 'dka', 'status epilepticus', 'stroke', 'stemi',
                 'unstable', 'resuscitat', 'icu', 'critical', 'acute abdomen', 'gi bleed',
                 'code blue', 'massive transfusion', 'tension pneumo']
    preg_kw = ['pregnan', 'trimester', 'obstetric', 'postpartum', 'eclampsia', 'ectopic',
               'gestational', 'labor', 'delivery', 'fetal', 'neonatal', 'newborn',
               'gravida', 'para ', 'antepartum', 'intrapartum', 'amniotic']
    peds_kw = ['child', 'infant', 'pediatric', 'neonate', 'boy', 'girl', 'year-old boy',
               'year-old girl', 'month-old', 'toddler', 'adolescent', 'baby']
    psych_kw = ['psychiatr', 'suicid', 'psychosis', 'schizophren', 'bipolar', 'depression',
                'anxiety disorder', 'hallucination', 'delusion', 'substance abuse', 'overdose',
                'anorexia nervosa', 'bulimia', 'ptsd', 'ocd', 'panic disorder']
    geri_kw = ['elderly', 'geriatric', 'nursing home', '85-year', '90-year', 'dementia',
               'polypharmacy', 'fall risk', 'frail', 'alzheimer']
    mech_kw = ['mechanism of action', 'pathophysiology', 'molecular', 'signaling pathway',
               'receptor', 'pharmacology', 'biochem', 'how does', 'explain the mechanism',
               'enzyme', 'inhibitor', 'agonist', 'antagonist', 'gene expression']

    age_match = re.search(r'(\d+)[\s-]*(year|month|week|day)[\s-]*old', ql)
    if age_match:
        age_val = int(age_match.group(1))
        unit = age_match.group(2)
        if unit in ('month', 'week', 'day') or (unit == 'year' and age_val < 18):
            return 'PEDIATRIC'

    if any(k in ql for k in acute_kw): return 'ACUTE'
    if any(k in ql for k in preg_kw): return 'PREGNANCY'
    if any(k in ql for k in psych_kw): return 'PSYCHIATRY'
    if any(k in ql for k in geri_kw): return 'GERIATRIC'
    if any(k in ql for k in mech_kw): return 'BASIC_SCIENCE'

    has_options = bool(re.search(r'\b[A-E]\)', q) or re.search(r'\b[A-E]\.', q))
    if has_options: return 'MCQ'
    return 'DEFAULT'

# Classify all questions
route_pools = {}
for i, entry in enumerate(data):
    if i in already_done:
        continue
    route = classify_route(entry['Question'])
    route_pools.setdefault(route, []).append((i, entry))

print("Available by route (after exclusions):")
for route, items in sorted(route_pools.items(), key=lambda x: -len(x[1])):
    print(f"  {route}: {len(items)}")

# Target: 5000 total
# 60% diag/mgmt (DEFAULT+MCQ), 15% labs/preventive, 10% acute, 15% special routes
targets = {
    'DEFAULT': 1500,
    'MCQ': 1500,
    'ACUTE': 500,
    'PEDIATRIC': 400,
    'PREGNANCY': 350,
    'BASIC_SCIENCE': 350,
    'PSYCHIATRY': 250,
    'GERIATRIC': 102,  # take all available
}

selected = []
for route, count in targets.items():
    pool = route_pools.get(route, [])
    actual = min(count, len(pool))
    if actual < count:
        print(f"  NOTE: {route} capped at {actual} (wanted {count})")
    chosen = random.sample(pool, actual)
    for idx, entry in chosen:
        selected.append({'index': idx, 'route': route, 'question': entry['Question'],
                         'original_cot': entry.get('Complex_CoT', ''),
                         'original_response': entry.get('Response', '')})

# Fill remaining to reach 5000 from DEFAULT pool
remaining = 5000 - len(selected)
if remaining > 0:
    used_indices = {s['index'] for s in selected}
    extra_pool = [(i, e) for i, e in route_pools.get('DEFAULT', []) if i not in used_indices]
    extra = random.sample(extra_pool, min(remaining, len(extra_pool)))
    for idx, entry in extra:
        selected.append({'index': idx, 'route': 'DEFAULT', 'question': entry['Question'],
                         'original_cot': entry.get('Complex_CoT', ''),
                         'original_response': entry.get('Response', '')})

random.shuffle(selected)
print(f"\nTotal selected: {len(selected)}")

# Route breakdown
from collections import Counter
rc = Counter(s['route'] for s in selected)
for route, count in rc.most_common():
    print(f"  {route}: {count}")

# Split into batches of 10
BATCH_SIZE = 10
batches = [selected[i:i+BATCH_SIZE] for i in range(0, len(selected), BATCH_SIZE)]
print(f"Batches: {len(batches)} (size {BATCH_SIZE})")

# Create output directory
out_dir = '/tmp/prism-v3-distill-5k'
os.makedirs(out_dir, exist_ok=True)

# Save manifest
with open(f'{out_dir}/selected_questions.json', 'w') as f:
    json.dump(selected, f, ensure_ascii=False)

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
        q_num = batch_idx * BATCH_SIZE + q_idx + 1
        prompt += f"### Question {q_num}\n{item['question']}\n\n"

    prompt_file = f'{out_dir}/batch_{batch_idx:04d}.md'
    with open(prompt_file, 'w') as f:
        f.write(prompt)

print(f"Prompt files: {out_dir}/batch_0000.md to batch_{len(batches)-1:04d}.md")

# Generate runner script — 50 parallel workers
runner = f"""#!/bin/bash
# PRISM v3 Distillation — {len(batches)} batches, 50 parallel workers
OUT_DIR="{out_dir}"
TOTAL={len(batches)}
WORKERS=50
LOG="$OUT_DIR/progress.log"

echo "$(date): Starting $TOTAL batches with $WORKERS parallel workers" | tee "$LOG"

run_batch() {{
    local BATCH_NUM=$1
    local BATCH_FILE="$OUT_DIR/batch_$(printf '%04d' $BATCH_NUM).md"
    local OUTPUT_FILE="$OUT_DIR/output_$(printf '%04d' $BATCH_NUM).md"
    local ERROR_FILE="$OUT_DIR/error_$(printf '%04d' $BATCH_NUM).log"

    # Skip if already done
    if [ -s "$OUTPUT_FILE" ]; then
        echo "$(date): SKIP batch $BATCH_NUM (already done)" >> "$LOG"
        return 0
    fi

    codex exec -m gpt-5.4 \\
        --skip-git-repo-check \\
        --sandbox read-only \\
        --ephemeral \\
        -c model_reasoning_effort="xhigh" \\
        -o "$OUTPUT_FILE" \\
        - < "$BATCH_FILE" \\
        2>"$ERROR_FILE"

    if [ -s "$OUTPUT_FILE" ]; then
        echo "$(date): DONE batch $BATCH_NUM ($(wc -c < "$OUTPUT_FILE") bytes)" >> "$LOG"
    else
        echo "$(date): FAIL batch $BATCH_NUM" >> "$LOG"
    fi
}}

export -f run_batch
export OUT_DIR LOG

# Use seq + xargs for controlled parallelism
seq 0 $(($TOTAL - 1)) | xargs -P $WORKERS -I {{}} bash -c 'run_batch {{}}'

# Summary
DONE=$(find "$OUT_DIR" -name "output_*.md" -size +0c 2>/dev/null | wc -l | tr -d ' ')
FAILED=$(($TOTAL - $DONE))
echo "" | tee -a "$LOG"
echo "$(date): Complete. $DONE/$TOTAL successful, $FAILED failed." | tee -a "$LOG"

if [ $FAILED -gt 0 ]; then
    echo "Failed batches:" | tee -a "$LOG"
    for i in $(seq 0 $(($TOTAL - 1))); do
        f="$OUT_DIR/output_$(printf '%04d' $i).md"
        [ ! -s "$f" ] && echo "  batch $i" | tee -a "$LOG"
    done
fi
"""

with open(f'{out_dir}/run_distill.sh', 'w') as f:
    f.write(runner)
os.chmod(f'{out_dir}/run_distill.sh', 0o755)
print(f"Runner: {out_dir}/run_distill.sh")
print(f"\nEstimated time: ~2-3 hours with 50 parallel workers")
