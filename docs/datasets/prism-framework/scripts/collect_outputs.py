#!/usr/bin/env python3
"""Collect and parse PRISM v3 distillation outputs into training JSONL."""
import json, re, os, glob, sys

OUT_DIR = '/tmp/prism-v3-distill'
DEST_DIR = '/Users/biobook/datasets/prism-framework/distilled'

os.makedirs(DEST_DIR, exist_ok=True)

# Load the question manifest
with open(f'{OUT_DIR}/selected_questions.json') as f:
    selected = json.load(f)

# Map question index to metadata
q_meta = {}
for i, item in enumerate(selected):
    q_meta[i] = item

# Parse output files
output_files = sorted(glob.glob(f'{OUT_DIR}/output_*.md'))
print(f"Found {len(output_files)} output files")

# Check which are non-empty
good_files = [f for f in output_files if os.path.getsize(f) > 0]
empty_files = [f for f in output_files if os.path.getsize(f) == 0]
print(f"Non-empty: {len(good_files)}, Empty: {len(empty_files)}")

if empty_files:
    print(f"Empty files: {[os.path.basename(f) for f in empty_files]}")

def parse_output_schema(text):
    """Extract the structured output block from a response."""
    schema = {}
    # Look for the ## Output block
    output_match = re.search(r'## Output\s*\n(.*?)(?=\n## |\n### |\Z)', text, re.DOTALL)
    if not output_match:
        output_match = re.search(r'```\s*\n\s*## Output\s*\n(.*?)```', text, re.DOTALL)
    if not output_match:
        # Try without header
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

def split_questions(text, batch_idx):
    """Split a batch output into individual question responses."""
    # Try splitting on "### Question N" or "**Question N**" or "Question N"
    parts = re.split(r'(?=(?:###?\s*)?(?:\*\*)?Question\s+\d+(?:\*\*)?)', text)
    results = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Extract question number
        q_match = re.match(r'(?:###?\s*)?(?:\*\*)?Question\s+(\d+)(?:\*\*)?', part)
        if q_match:
            q_num = int(q_match.group(1))
            results.append((q_num, part))
    return results

# Process all outputs
all_examples = []
parse_failures = 0
total_parsed = 0

for output_file in good_files:
    batch_idx = int(re.search(r'output_(\d+)', output_file).group(1))
    with open(output_file) as f:
        content = f.read()

    questions = split_questions(content, batch_idx)

    if not questions:
        print(f"  WARNING: Could not split {os.path.basename(output_file)} into questions")
        # Treat whole file as one response
        q_global = batch_idx * 5
        meta = q_meta.get(q_global, {})
        schema = parse_output_schema(content)
        example = {
            'question': meta.get('question', ''),
            'route_classified': meta.get('route', 'UNKNOWN'),
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
        q_global = q_num - 1  # Questions are 1-indexed
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

# Route distribution
route_counts = {}
for ex in all_examples:
    r = ex['route_classified']
    route_counts[r] = route_counts.get(r, 0) + 1
print(f"\nRoute distribution:")
for route, count in sorted(route_counts.items(), key=lambda x: -x[1]):
    print(f"  {route}: {count}")

# Confidence distribution
conf_counts = {}
for ex in all_examples:
    if ex['structured_output']:
        c = ex['structured_output'].get('confidence', 'MISSING')
        conf_counts[c] = conf_counts.get(c, 0) + 1
if conf_counts:
    print(f"\nConfidence distribution:")
    for conf, count in sorted(conf_counts.items(), key=lambda x: -x[1]):
        print(f"  {conf}: {count}")

# Average CoT length
cot_lengths = [len(ex['prism_v3_cot']) for ex in all_examples]
if cot_lengths:
    print(f"\nCoT length stats:")
    print(f"  Mean: {sum(cot_lengths)//len(cot_lengths)} chars")
    print(f"  Min: {min(cot_lengths)} chars")
    print(f"  Max: {max(cot_lengths)} chars")

# Save as JSONL
jsonl_path = f'{DEST_DIR}/prism_v3_distilled.jsonl'
with open(jsonl_path, 'w') as f:
    for ex in all_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
print(f"\nSaved {len(all_examples)} examples to {jsonl_path}")

# Also save a clean training format (question + CoT only, no metadata)
train_path = f'{DEST_DIR}/prism_v3_train.jsonl'
with open(train_path, 'w') as f:
    for ex in all_examples:
        if ex['parse_status'] == 'ok':
            train_ex = {
                'messages': [
                    {'role': 'user', 'content': ex['question']},
                    {'role': 'assistant', 'content': ex['prism_v3_cot']}
                ]
            }
            f.write(json.dumps(train_ex, ensure_ascii=False) + '\n')
valid_count = sum(1 for ex in all_examples if ex['parse_status'] == 'ok')
print(f"Saved {valid_count} valid training examples to {train_path}")
