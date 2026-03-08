#!/usr/bin/env python3
"""Collect and parse PRISM v3 5k distillation outputs into training JSONL."""
import json, re, os, glob

OUT_DIR = '/tmp/prism-v3-distill-5k'
PREV_DIR = '/Users/biobook/datasets/prism-framework/distilled'
DEST_DIR = '/Users/biobook/datasets/prism-framework/distilled'

os.makedirs(DEST_DIR, exist_ok=True)

# Load question manifest
with open(f'{OUT_DIR}/selected_questions.json') as f:
    selected = json.load(f)

q_meta = {i: item for i, item in enumerate(selected)}

# Parse output files
output_files = sorted(glob.glob(f'{OUT_DIR}/output_*.md'))
good_files = [f for f in output_files if os.path.getsize(f) > 0]
empty_files = [f for f in output_files if os.path.getsize(f) == 0]
missing = 500 - len(output_files)

print(f"Output files found: {len(output_files)}/500")
print(f"Non-empty: {len(good_files)}, Empty: {len(empty_files)}, Missing: {missing}")

def parse_output_schema(text):
    schema = {}
    output_match = re.search(r'## Output\s*\n(.*?)(?=\n## |\n### |\Z)', text, re.DOTALL)
    if not output_match:
        output_match = re.search(r'```\s*\n\s*## Output\s*\n(.*?)```', text, re.DOTALL)
    if not output_match:
        output_match = re.search(r'route:\s*(.*?)(?=\n\n\n|\Z)', text, re.DOTALL)
    if output_match:
        block = output_match.group(0) if not output_match.group(1) else output_match.group(1)
        for line in block.strip().split('\n'):
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, _, val = line.partition(':')
                key = key.strip().lower().replace(' ', '_')
                val = val.strip().strip('"').strip("'")
                if val:
                    schema[key] = val
    return schema if schema else None

def split_questions(text, batch_idx, batch_size=10):
    parts = re.split(r'(?=(?:###?\s*)?(?:\*\*)?Question\s+\d+(?:\*\*)?)', text)
    results = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        q_match = re.match(r'(?:###?\s*)?(?:\*\*)?Question\s+(\d+)(?:\*\*)?', part)
        if q_match:
            q_num = int(q_match.group(1))
            results.append((q_num, part))
    return results

all_examples = []
parse_failures = 0
total_parsed = 0
no_split = 0

for output_file in good_files:
    batch_idx = int(re.search(r'output_(\d+)', output_file).group(1))
    with open(output_file) as f:
        content = f.read()

    questions = split_questions(content, batch_idx)

    if not questions:
        no_split += 1
        q_global = batch_idx * 10
        meta = q_meta.get(q_global, {})
        schema = parse_output_schema(content)
        example = {
            'question': meta.get('question', ''),
            'route_classified': meta.get('route', 'UNKNOWN'),
            'route_assigned': schema.get('route', '') if schema else '',
            'prism_v3_cot': content,
            'structured_output': schema,
            'original_index': meta.get('index', -1),
            'batch': batch_idx,
            'parse_status': 'unsplit'
        }
        all_examples.append(example)
        if not schema:
            parse_failures += 1
        total_parsed += 1
        continue

    for q_num, response_text in questions:
        q_global = q_num - 1  # 1-indexed
        meta = q_meta.get(q_global, {})
        schema = parse_output_schema(response_text)
        example = {
            'question': meta.get('question', ''),
            'route_classified': meta.get('route', 'UNKNOWN'),
            'route_assigned': schema.get('route', '') if schema else '',
            'prism_v3_cot': response_text,
            'structured_output': schema,
            'original_index': meta.get('index', -1),
            'batch': batch_idx,
            'parse_status': 'ok' if schema else 'no_schema'
        }
        all_examples.append(example)
        if not schema:
            parse_failures += 1
        total_parsed += 1

print(f"\nTotal parsed: {total_parsed}")
print(f"With valid schema: {total_parsed - parse_failures}")
print(f"Missing schema: {parse_failures}")
print(f"Unsplit batches: {no_split}")

# Route distribution
from collections import Counter
rc = Counter(ex['route_classified'] for ex in all_examples)
print(f"\nRoute distribution:")
for route, count in rc.most_common():
    print(f"  {route}: {count}")

# Confidence distribution
conf_counts = Counter()
for ex in all_examples:
    if ex['structured_output']:
        conf_counts[ex['structured_output'].get('confidence', 'MISSING')] += 1
if conf_counts:
    print(f"\nConfidence distribution:")
    for conf, count in conf_counts.most_common():
        print(f"  {conf}: {count}")

# CoT length stats
cot_lengths = [len(ex['prism_v3_cot']) for ex in all_examples]
if cot_lengths:
    cot_lengths.sort()
    print(f"\nCoT length stats:")
    print(f"  Mean: {sum(cot_lengths)//len(cot_lengths)} chars")
    print(f"  Median: {cot_lengths[len(cot_lengths)//2]} chars")
    print(f"  Min: {min(cot_lengths)}, Max: {max(cot_lengths)}")

# Merge with previous 100
prev_path = f'{PREV_DIR}/prism_v3_distilled.jsonl'
prev_examples = []
if os.path.exists(prev_path):
    with open(prev_path) as f:
        prev_examples = [json.loads(line) for line in f]
    print(f"\nPrevious batch: {len(prev_examples)} examples")

merged = prev_examples + all_examples
print(f"Total merged: {len(merged)}")

# Save full dataset
full_path = f'{DEST_DIR}/prism_v3_full.jsonl'
with open(full_path, 'w') as f:
    for ex in merged:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
print(f"Saved to {full_path}")

# Save clean training format
train_path = f'{DEST_DIR}/prism_v3_train_full.jsonl'
valid = 0
with open(train_path, 'w') as f:
    for ex in merged:
        if ex.get('parse_status') in ('ok', 'unsplit') and ex.get('question'):
            train_ex = {
                'messages': [
                    {'role': 'user', 'content': ex['question']},
                    {'role': 'assistant', 'content': ex['prism_v3_cot']}
                ]
            }
            f.write(json.dumps(train_ex, ensure_ascii=False) + '\n')
            valid += 1
print(f"Saved {valid} training examples to {train_path}")

# List failed batches for retry
if empty_files or missing > 0:
    failed_batches = []
    for i in range(500):
        f = f'{OUT_DIR}/output_{i:04d}.md'
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            failed_batches.append(i)
    if failed_batches:
        with open(f'{OUT_DIR}/failed_batches.txt', 'w') as f:
            f.write('\n'.join(str(b) for b in failed_batches))
        print(f"\n{len(failed_batches)} failed batches written to {OUT_DIR}/failed_batches.txt")
