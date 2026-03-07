"""
Load MedQA, MedMCQA, and PubMedQA into a unified instruction-tuning format.

Each example becomes:
  input:  "Question: ... \nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:"
  output: "B. [explanation if available]"

MedMCQA has expert explanations — perfect for chain-of-thought training.
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "medical_qa"

OPTION_LETTERS = ["A", "B", "C", "D"]


def load_medqa(split="train"):
    """Load MedQA (USMLE 4-options). 10K train, 1.3K test."""
    path = DATA_DIR / f"medqa_{split}.json"
    if not path.exists():
        return []

    with open(path) as f:
        raw = json.load(f)

    examples = []
    for item in raw:
        options = item["options"]  # dict {"A": "...", "B": "...", ...}
        opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())
        answer_letter = item["answer_idx"]
        answer_text = options.get(answer_letter, item.get("answer", ""))

        examples.append({
            "input": f"Question: {item['question']}\n\n{opts_text}\n\nAnswer:",
            "output": f"{answer_letter}. {answer_text}",
            "source": "medqa",
            "has_explanation": False,
        })
    return examples


def load_medmcqa(split="train"):
    """Load MedMCQA. 183K train with explanations — chain-of-thought gold."""
    path = DATA_DIR / f"medmcqa_{split}.json"
    if not path.exists():
        return []

    with open(path) as f:
        raw = json.load(f)

    examples = []
    for item in raw:
        options = {
            "A": item["opa"],
            "B": item["opb"],
            "C": item["opc"],
            "D": item["opd"],
        }
        opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())
        answer_idx = item["cop"]  # 0-indexed integer
        if not isinstance(answer_idx, int) or answer_idx < 0 or answer_idx > 3:
            continue
        answer_letter = OPTION_LETTERS[answer_idx]
        answer_text = options[answer_letter]

        # Build output with explanation if available
        explanation = item.get("exp", "")
        if explanation and len(explanation.strip()) > 10:
            output = f"{answer_letter}. {answer_text}\n\nExplanation: {explanation.strip()}"
            has_exp = True
        else:
            output = f"{answer_letter}. {answer_text}"
            has_exp = False

        subject = item.get("subject_name", "")
        topic = item.get("topic_name", "")
        context = ""
        if subject:
            context = f"[{subject}"
            if topic:
                context += f" — {topic}"
            context += "] "

        examples.append({
            "input": f"{context}Question: {item['question']}\n\n{opts_text}\n\nAnswer:",
            "output": output,
            "source": "medmcqa",
            "has_explanation": has_exp,
        })
    return examples


def load_pubmedqa(split="train", include_artificial=True):
    """Load PubMedQA. 1K labeled + 211K artificial."""
    examples = []

    # Labeled
    path = DATA_DIR / f"pubmedqa_{split}.json"
    if path.exists():
        with open(path) as f:
            raw = json.load(f)
        for item in raw:
            context = item.get("context", {})
            if isinstance(context, dict):
                # context is {"contexts": [...], "labels": [...], "meshes": [...]}
                ctx_texts = context.get("contexts", [])
                context_str = " ".join(ctx_texts) if ctx_texts else ""
            elif isinstance(context, str):
                context_str = context
            else:
                context_str = str(context)

            decision = item.get("final_decision", "").lower()
            long_answer = item.get("long_answer", "")

            if decision and decision in ("yes", "no", "maybe"):
                output = f"{decision}"
                if long_answer:
                    output += f"\n\nExplanation: {long_answer}"

                inp = f"Context: {context_str[:500]}\n\nQuestion: {item['question']}\n\nAnswer (yes/no/maybe):"
                examples.append({
                    "input": inp,
                    "output": output,
                    "source": "pubmedqa_labeled",
                    "has_explanation": bool(long_answer),
                })

    # Artificial (larger, auto-generated)
    if include_artificial:
        art_path = DATA_DIR / "pubmedqa_artificial_train.json"
        if art_path.exists():
            with open(art_path) as f:
                raw = json.load(f)
            for item in raw:
                context = item.get("context", {})
                if isinstance(context, dict):
                    ctx_texts = context.get("contexts", [])
                    context_str = " ".join(ctx_texts) if ctx_texts else ""
                elif isinstance(context, str):
                    context_str = context
                else:
                    context_str = str(context)

                decision = item.get("final_decision", "").lower()
                long_answer = item.get("long_answer", "")

                if decision and decision in ("yes", "no", "maybe"):
                    output = f"{decision}"
                    if long_answer:
                        output += f"\n\nExplanation: {long_answer}"

                    inp = f"Context: {context_str[:500]}\n\nQuestion: {item['question']}\n\nAnswer (yes/no/maybe):"
                    examples.append({
                        "input": inp,
                        "output": output,
                        "source": "pubmedqa_artificial",
                        "has_explanation": bool(long_answer),
                    })

    return examples


def build_medical_qa_dataset(
    include_medqa=True,
    include_medmcqa=True,
    include_pubmedqa=True,
    include_pubmedqa_artificial=True,
    max_per_source=None,
    seed=42,
):
    """Build combined medical QA instruction-tuning dataset."""
    all_examples = []

    if include_medqa:
        ex = load_medqa("train")
        if max_per_source and len(ex) > max_per_source:
            random.seed(seed)
            ex = random.sample(ex, max_per_source)
        all_examples.extend(ex)
        print(f"  MedQA: {len(ex)} examples")

    if include_medmcqa:
        ex = load_medmcqa("train")
        if max_per_source and len(ex) > max_per_source:
            random.seed(seed)
            ex = random.sample(ex, max_per_source)
        all_examples.extend(ex)
        print(f"  MedMCQA: {len(ex)} examples")

    if include_pubmedqa:
        ex = load_pubmedqa("train", include_artificial=include_pubmedqa_artificial)
        if max_per_source and len(ex) > max_per_source:
            random.seed(seed)
            ex = random.sample(ex, max_per_source)
        all_examples.extend(ex)
        print(f"  PubMedQA: {len(ex)} examples")

    random.seed(seed)
    random.shuffle(all_examples)

    with_exp = sum(1 for e in all_examples if e["has_explanation"])
    print(f"  Total: {len(all_examples)} examples ({with_exp} with explanations)")

    return all_examples


if __name__ == "__main__":
    print("Loading medical QA datasets...")
    examples = build_medical_qa_dataset()
    print(f"\nSample MedQA:")
    for ex in examples:
        if ex["source"] == "medqa":
            print(f"  IN:  {ex['input'][:120]}...")
            print(f"  OUT: {ex['output'][:120]}...")
            break
    print(f"\nSample MedMCQA (with explanation):")
    for ex in examples:
        if ex["source"] == "medmcqa" and ex["has_explanation"]:
            print(f"  IN:  {ex['input'][:120]}...")
            print(f"  OUT: {ex['output'][:200]}...")
            break
