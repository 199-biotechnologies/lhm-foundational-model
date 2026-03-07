
import json
import random
from pathlib import Path

DATA_PATH = Path("experiments/exp6_medical_qa/mlx_data/train.jsonl")
FILTERED_PATH = Path("experiments/exp6_medical_qa/mlx_data/train_filtered.jsonl")

def filter_explanations():
    print(f"Filtering {DATA_PATH} to only examples WITH explanations...")
    count = 0
    total = 0
    with open(DATA_PATH, "r") as f_in, open(FILTERED_PATH, "w") as f_out:
        for line in f_in:
            total += 1
            data = json.loads(line)
            assistant_content = data["messages"][1]["content"]
            if "Explanation:" in assistant_content or "Explanation:" in assistant_content.lower():
                f_out.write(json.dumps(data) + "
")
                count += 1
    
    print(f"Filter complete! Kept {count}/{total} examples. New file: {FILTERED_PATH}")

if __name__ == "__main__":
    filter_explanations()
