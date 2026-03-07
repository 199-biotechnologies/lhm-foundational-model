
import json
from pathlib import Path

# Paths
INPUT_PATH = Path("experiments/exp6_medical_qa/mlx_data/train.jsonl")
OUTPUT_PATH = Path("experiments/exp6_medical_qa/mlx_data/train_think.jsonl")

def reformat_messages(messages):
    user_msg = messages[0]["content"]
    assistant_msg = messages[1]["content"]
    
    # Check if explanation is present
    if "Explanation:" in assistant_msg:
        parts = assistant_msg.split("Explanation:", 1)
        answer = parts[0].strip()
        explanation = parts[1].strip()
        
        # Move explanation into <think> tags BEFORE the answer
        new_assistant_content = f"<think>
{explanation}
</think>
{answer}"
        messages[1]["content"] = new_assistant_content
    
    return messages

def process_file():
    print(f"Reformatting {INPUT_PATH} for reasoning...")
    with open(INPUT_PATH, "r") as f_in, open(OUTPUT_PATH, "w") as f_out:
        for line in f_in:
            data = json.loads(line)
            data["messages"] = reformat_messages(data["messages"])
            f_out.write(json.dumps(data) + "
")
    print(f"Reformat complete! New file: {OUTPUT_PATH}")

if __name__ == "__main__":
    process_file()
