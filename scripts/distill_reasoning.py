"""
Distill GPT-5.4 medical reasoning into Qwen 3.5 training format.

Sends medical questions to GPT-5.4 via Codex CLI, captures step-by-step
reasoning, and formats as <think>reasoning</think>answer for training.

Usage:
    python3 scripts/distill_reasoning.py --count 100 --batch-size 5
"""

import json
import subprocess
import sys
import time
from pathlib import Path

DATA_DIR = Path("data/raw/medical_qa")
OUTPUT_DIR = Path("experiments/exp6_medical_qa/distilled_data")

SYSTEM_PROMPT = """You are a medical expert taking a board exam. For each question, provide:
1. Your step-by-step clinical reasoning (be thorough but concise — cover key differentials, why you rule them out, and why your answer is correct)
2. Your final answer

Format your response EXACTLY as:
<reasoning>
[Your step-by-step thinking here — 3-8 sentences of real clinical logic, not filler]
</reasoning>
<answer>[Letter]. [Answer text]</answer>

Rules:
- Be clinically precise, not conversational. No "hmm", "let me think", "alright".
- Consider 2-3 differentials and explain why you rule them out.
- Reference specific pathophysiology, not just pattern matching.
- Keep reasoning under 200 words — dense and high-signal."""


def load_questions(count=100):
    """Load diverse medical questions."""
    questions = []

    # MedQA (USMLE) — gold standard
    with open(DATA_DIR / "medqa_train.json") as f:
        medqa = json.load(f)

    # MedMCQA — broader coverage
    with open(DATA_DIR / "medmcqa_train.json") as f:
        medmcqa = json.load(f)

    OPTION_LETTERS = ["A", "B", "C", "D"]

    # Take 60% MedQA, 40% MedMCQA for diversity
    import random
    random.seed(42)

    n_medqa = int(count * 0.6)
    n_medmcqa = count - n_medqa

    for item in random.sample(medqa, min(n_medqa, len(medqa))):
        options = item["options"]
        opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())
        questions.append({
            "question": f"Question: {item['question']}\n\n{opts_text}\n\nAnswer:",
            "correct_answer": item["answer_idx"],
            "correct_text": options.get(item["answer_idx"], ""),
            "source": "medqa",
        })

    for item in random.sample(medmcqa, min(n_medmcqa, len(medmcqa))):
        options = {
            "A": item["opa"], "B": item["opb"],
            "C": item["opc"], "D": item["opd"],
        }
        answer_idx = item.get("cop", -1)
        if not isinstance(answer_idx, int) or answer_idx < 0 or answer_idx > 3:
            continue
        answer_letter = OPTION_LETTERS[answer_idx]
        opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())
        subject = item.get("subject_name", "")
        topic = item.get("topic_name", "")
        prefix = f"[{subject}" + (f" — {topic}]" if topic else "]") if subject else ""
        questions.append({
            "question": f"{prefix} Question: {item['question']}\n\n{opts_text}\n\nAnswer:",
            "correct_answer": answer_letter,
            "correct_text": options[answer_letter],
            "source": "medmcqa",
        })

    return questions[:count]


def call_gpt54_batch(questions_batch):
    """Send a batch of questions to GPT-5.4 via Codex CLI."""
    batch_prompt = SYSTEM_PROMPT + "\n\n"
    batch_prompt += "Answer each question below. Number your responses to match.\n\n"

    for i, q in enumerate(questions_batch):
        batch_prompt += f"--- QUESTION {i+1} ---\n{q['question']}\n\n"

    batch_prompt += "--- END ---\nProvide all answers in order, each with <reasoning> and <answer> tags."

    # Write prompt to temp file
    prompt_file = Path(f"/tmp/distill-batch-{int(time.time())}.md")
    output_file = Path(f"/tmp/distill-output-{int(time.time())}.md")

    prompt_file.write_text(batch_prompt)

    result = subprocess.run(
        [
            "codex", "exec", "-m", "gpt-5.4",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
            "--ephemeral",
            "-c", "model_reasoning_effort=xhigh",
            "-o", str(output_file),
            "-",
        ],
        stdin=open(prompt_file),
        capture_output=True,
        text=True,
        timeout=300,
    )

    if output_file.exists():
        return output_file.read_text()
    return result.stdout


def parse_responses(raw_output, questions_batch):
    """Parse GPT-5.4 output into structured reasoning + answer pairs."""
    import re

    results = []

    # Find all reasoning/answer pairs
    reasoning_blocks = re.findall(
        r'<reasoning>(.*?)</reasoning>', raw_output, re.DOTALL
    )
    answer_blocks = re.findall(
        r'<answer>(.*?)</answer>', raw_output, re.DOTALL
    )

    for i, q in enumerate(questions_batch):
        reasoning = reasoning_blocks[i].strip() if i < len(reasoning_blocks) else ""
        answer = answer_blocks[i].strip() if i < len(answer_blocks) else ""

        if reasoning and answer:
            # Format for Qwen 3.5 training: <think> tags
            assistant_content = f"<think>\n{reasoning}\n</think>\n{answer}"

            results.append({
                "messages": [
                    {"role": "user", "content": q["question"]},
                    {"role": "assistant", "content": assistant_content},
                ],
                "source": q["source"],
                "correct_answer": q["correct_answer"],
                "gpt54_answer": answer[:1] if answer else "",
                "correct": answer.startswith(q["correct_answer"]) if answer else False,
            })

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading {args.count} medical questions...")
    questions = load_questions(args.count)
    print(f"Loaded {len(questions)} questions ({sum(1 for q in questions if q['source']=='medqa')} MedQA, {sum(1 for q in questions if q['source']=='medmcqa')} MedMCQA)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    correct_count = 0

    for batch_start in range(0, len(questions), args.batch_size):
        batch = questions[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (len(questions) + args.batch_size - 1) // args.batch_size

        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} questions)...")

        try:
            raw = call_gpt54_batch(batch)
            results = parse_responses(raw, batch)

            for r in results:
                all_results.append(r)
                if r["correct"]:
                    correct_count += 1

            print(f"  Parsed {len(results)}/{len(batch)} responses, "
                  f"correct so far: {correct_count}/{len(all_results)}")

            # Save intermediate results
            with open(OUTPUT_DIR / "gpt54_distilled_train.jsonl", "w") as f:
                for r in all_results:
                    # Write only the training format (messages)
                    f.write(json.dumps({"messages": r["messages"]}) + "\n")

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT on batch {batch_num}, skipping...")
            continue
        except Exception as e:
            print(f"  ERROR on batch {batch_num}: {e}")
            continue

    # Final stats
    print(f"\n{'='*60}")
    print(f"DONE: {len(all_results)} distilled examples")
    print(f"GPT-5.4 accuracy: {correct_count}/{len(all_results)} "
          f"({100*correct_count/max(1,len(all_results)):.1f}%)")
    print(f"Saved to: {OUTPUT_DIR / 'gpt54_distilled_train.jsonl'}")

    # Save full metadata
    with open(OUTPUT_DIR / "gpt54_distilled_metadata.json", "w") as f:
        json.dump({
            "total": len(all_results),
            "correct": correct_count,
            "accuracy": round(100 * correct_count / max(1, len(all_results)), 1),
            "sources": {
                "medqa": sum(1 for r in all_results if r["source"] == "medqa"),
                "medmcqa": sum(1 for r in all_results if r["source"] == "medmcqa"),
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()
