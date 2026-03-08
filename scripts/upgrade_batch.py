"""
Upgrade MedReason examples in batches — V4.3

Key changes from v4.2:
- FORWARD-ONLY REASONING: Stronger enforcement — must build from observations,
  explicitly bans naming the answer before the reasoning converges
- ANTI-TEMPLATE: Bans GPT-ism phrases ("Mechanistically", "clinical presentation
  is consistent with", "key finding here is") that create homogeneous training data
- KNOWLEDGE-CONSISTENT: Bans citing statistics, p-values, trial names, guideline
  numbers not in the question stem — prevents hallucination at inference when the
  0.8B model won't have access to these facts
- RETRY ON QUALITY FAILURE: If quality validators detect backward reasoning or
  template language, retries once with explicit anti-pattern feedback
- IMPROVED LONGEVITY CLASSIFIER: More exclusion keywords for surgical/procedural
  and health-services research questions

Carried from v4.2:
- Pre-classifies longevity relevance (two prompt variants, no self-gating)
- Independent cross-model verification (Gemini generates → Codex verifies)
- Fresh context per question (--ephemeral, stateless)
- Deduplication, parallel sharding, dataset-specific modes

Usage:
    python3 scripts/upgrade_batch.py --per-type 10
    python3 scripts/upgrade_batch.py --per-type 0 --datasets pubmedqa LastHumanity
    python3 scripts/upgrade_batch.py --per-type 100 --dry-run
"""

import json
import subprocess
import sys
import re
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

INPUT_FILE = Path("docs/datasets/medreason_32k.jsonl")
OUTPUT_DIR = Path("docs/datasets/upgraded")

# These get suffixed with shard ID when running parallel
LOG_FILE = OUTPUT_DIR / "upgrade_log.jsonl"
OUTPUT_FILE = OUTPUT_DIR / "v42_upgraded.jsonl"
FAILED_FILE = OUTPUT_DIR / "v42_failed.jsonl"

# Shard-aware globals (set in main)
SHARD_ID = None
SHARD_TOTAL = None

# ──────────────────────────────────────────────────────────────────────
# V4.2 SYSTEM PROMPTS
# ──────────────────────────────────────────────────────────────────────

# Base prompt — used for ALL examples
V43_BASE = """You are a physician-scientist upgrading medical reasoning traces.

TASK: Produce a chain-of-thought that reasons FORWARD from the question's clinical
findings to the answer. The ground-truth answer is provided — your reasoning must
arrive at it, but through forward discovery, not backward justification.

FORWARD REASONING (CRITICAL):
- Start from the most striking finding in the question stem.
- Build a pathophysiological chain: finding → mechanism → expected consequence.
- Introduce and eliminate differentials using SPECIFIC findings from the stem.
- ONLY name the answer/diagnosis AFTER your reasoning has converged on it.
- If your first sentence contains the answer word, you have failed.
- If you cannot construct a sound forward argument, output <flag>UNSUPPORTED</flag>.

KNOWLEDGE RULES:
- DO NOT cite specific statistics, p-values, sample sizes, or trial names that are
  not in the question stem. The student model will not have access to these at
  inference and will fabricate them.
- DO NOT use "Grade A evidence", "pathognomonic", or "High-certainty" unless naming
  the exact source. Use calibrated language: "consistent with", "strongly suggests".
- Each differential you eliminate MUST cite a specific stem finding, not a textbook
  generality.

STYLE RULES:
- DO NOT restate the question or use filler ("Let's analyze", "Interestingly",
  "This is a classic case of").
- DO NOT start sentences with "Mechanistically" or "The clinical presentation is
  consistent with".
- Vary sentence structure. Avoid repeating the same sentence pattern more than twice.
- PRESERVE accurate KG logic from the original reasoning as natural prose.

DOMAIN MATCHING:
- Anatomy/Histology: structural relationships, fascial planes, embryology.
- Pharmacology: MoA, receptor kinetics, side-effect mechanisms.
- Clinical: pivotal findings, pathophysiology, population nuances.
- Diagnostics: test characteristics, sensitivity/specificity, timing.
- Basic Science: metabolic pathways, molecular signaling.

FORMAT (STRICT):
<reasoning>
[5-15 sentences. Finding → mechanism → differential elimination → convergence.]
</reasoning>
<answer>[Letter]. [Answer text]</answer>"""

# Longevity extension — ONLY appended when pre-classified as relevant
LONGEVITY_EXTENSION = """

LONGEVITY & PREVENTIVE LENS:
After your core clinical reasoning, add 1-3 sentences connecting this clinical
scenario to upstream modifiable drivers, optimal-vs-normal ranges, or preventive
interventions. Frame this through a longevity medicine perspective:
- Identify which Hallmarks of Aging are relevant (if any)
- Note upstream metabolic or lifestyle drivers
- Mention screening, early detection, or intervention timing
This should feel like a natural extension of the reasoning, not a bolted-on paragraph."""

# Literature/PubMedQA variant — for questions about study conclusions
LITERATURE_PROMPT = """You are a physician-scientist evaluating a research finding.

TASK: Reason about whether the study's conclusion is supported. The ground-truth
answer is provided — arrive at it through forward analysis.

FORWARD REASONING:
- Start from the biological/clinical mechanism the study is investigating.
- Assess whether the study design can answer the question (plausibility analysis).
- Consider alternative explanations or confounders.
- Converge on the conclusion based on mechanistic plausibility, NOT by citing
  specific results you don't have access to.
- If the conclusion cannot be supported, output <flag>UNSUPPORTED</flag>.

KNOWLEDGE RULES:
- DO NOT fabricate statistics, p-values, sample sizes, effect sizes, or confidence
  intervals not present in the question stem.
- DO NOT name specific trials, meta-analyses, or guidelines not mentioned in the stem.
- Focus on biological plausibility and study design logic — this is what transfers
  to clinical reasoning at inference.
- Use calibrated language: "the data suggest", "consistent with", "mechanistically
  plausible because".

STYLE RULES:
- DO NOT start with "This study..." or "The research question...".
- DO NOT use "Mechanistically, this is plausible" — vary your language.
- Avoid stating the conclusion in your first sentence.

FORMAT (STRICT):
<reasoning>
[5-12 sentences. Mechanism → design assessment → plausibility → conclusion.]
</reasoning>
<answer>[Letter]. [Answer text]</answer>"""

VERIFY_SYSTEM = """You are a medical examiner. Given ONLY the reasoning below, determine the answer
to the question. You must pick from the provided options. Output ONLY the letter."""

# ──────────────────────────────────────────────────────────────────────
# LONGEVITY PRE-CLASSIFIER
# ──────────────────────────────────────────────────────────────────────

LONGEVITY_KEYWORDS = {
    # Chronic diseases
    "diabetes", "hypertension", "atherosclerosis", "heart failure", "copd",
    "chronic kidney", "cirrhosis", "fibrosis", "dementia", "alzheimer",
    "parkinson", "osteoporosis", "sarcopenia", "obesity", "metabolic syndrome",
    # Cancer / oncology
    "cancer", "carcinoma", "lymphoma", "leukemia", "melanoma", "tumor",
    "neoplasm", "malignancy", "oncology", "metastas", "screening",
    # Cardiovascular
    "coronary", "myocardial infarction", "stroke", "aneurysm", "dyslipidemia",
    "cholesterol", "triglyceride", "statin", "hypertensive",
    # Aging / longevity
    "aging", "ageing", "elderly", "geriatric", "senescence", "telomere",
    "longevity", "lifespan", "healthspan", "frailty",
    # Metabolic
    "insulin resistance", "hba1c", "glucose", "metabolic", "bmi",
    "visceral fat", "fatty liver", "nafld", "nash",
    # Prevention
    "screening", "prevention", "vaccination", "lifestyle", "smoking cessation",
    "exercise", "dietary", "modifiable risk",
    # Transplant / chronic management
    "transplant", "immunosuppress", "chronic rejection",
}

NOT_LONGEVITY_KEYWORDS = {
    # Anatomy / basic science
    "anatomy", "embryology", "histology", "fascial", "ligament", "tendon",
    "nerve root", "dermatome", "foramen", "brachial plexus",
    # Microbiology
    "plant", "bacterial", "viral replication", "parasite", "fungal",
    "gram stain", "culture", "agar",
    # Biochemistry
    "organic chemistry", "amino acid", "nucleotide", "enzyme kinetics",
    "mitochondria", "ribosome", "golgi", "endoplasmic reticulum",
    # Non-medical domains (LastHumanity, etc.)
    "camera trap", "efficientnet", "neural network", "deep learning",
    "machine learning", "classifier", "augmentation", "gaussian blur",
    "convolutional", "image classification", "pixel", "aperture",
    "drosophila", "tardigrade", "archaea", "bioinformatics",
    "sequence alignment", "phylogenetic", "metagenomics",
    "ravine", "habitat", "species distribution",
    # Surgical / procedural — longevity lens adds noise to acute surgical questions
    "surgical technique", "laparoscopic", "incision", "suture",
    "anastomosis", "tourniquet", "debridement", "amputation",
    "arthroscopy", "thoracotomy", "tracheostomy", "fasciotomy",
    # Acute emergency — not longevity-relevant
    "trauma", "fracture", "dislocation", "laceration", "hemorrhage",
    "cardiac arrest", "anaphylaxis", "status epilepticus",
    # Health services research — study design, not clinical reasoning
    "cost-effectiveness", "health services", "hospital readmission rate",
    "quality improvement", "patient satisfaction", "length of stay",
    "retrospective analysis", "survey", "questionnaire",
    # Pediatrics (acute) — child-specific presentations
    "neonatal", "newborn", "infant", "congenital",
    # Toxicology
    "poisoning", "overdose", "antidote", "toxidrome",
}


def is_longevity_relevant(question, options="", dataset_name=""):
    """Pre-classify whether longevity lens should be applied."""
    text = f"{question} {options}".lower()

    # Check exclusions first — any exclusion keyword blocks longevity
    for kw in NOT_LONGEVITY_KEYWORDS:
        if kw in text:
            return False

    # Check longevity relevance — need at least 1 keyword
    longevity_hits = sum(1 for kw in LONGEVITY_KEYWORDS if kw in text)
    return longevity_hits >= 1


def is_literature_question(dataset_name, question):
    """Detect literature/research questions that need different prompting."""
    if dataset_name in ("pubmedqa", "pubmedqa_artificial", "pubmedqa_unlabeled"):
        return True
    # Heuristic: questions about studies/trials
    q_lower = question.lower()
    return any(kw in q_lower for kw in [
        "study shows", "trial", "meta-analysis", "systematic review",
        "cohort study", "retrospective", "prospective",
    ])


# ──────────────────────────────────────────────────────────────────────
# GT LETTER EXTRACTION (expanded A-J, robust matching)
# ──────────────────────────────────────────────────────────────────────

def extract_gt_letter(example):
    """Extract ground truth letter from answer field, matching against options."""
    answer_text = example.get("answer", "")
    options_text = example.get("options", "")

    if not answer_text:
        return ""

    # 1. Direct letter prefix: "B. Small cell..." or "(E) Iliolumbar..."
    m = re.match(r'^\(?([A-J])\)?[\.\)]?\s', answer_text)
    if m:
        return m.group(1)

    # 2. Yes/No for pubmedqa variants
    lower = answer_text.lower().strip()
    if example.get("dataset_name", "") in ("pubmedqa", "pubmedqa_artificial", "pubmedqa_unlabeled"):
        if "yes" in lower[:20]:
            return "A"
        if "no" in lower[:20]:
            return "B"
        if "maybe" in lower[:20]:
            return "C"

    # Also handle generic yes/no
    if "the final decision is: yes" in lower:
        return "A"
    if "the final decision is: no" in lower:
        return "B"

    # 3. Match answer text against parsed options
    if options_text:
        # Parse options: "A. text\nB. text\n..." or "A) text\nB) text\n..."
        clean_opts = options_text.replace("Answer Choices:\n", "").replace("Answer Choices:", "")
        option_matches = re.findall(
            r'([A-J])[\.\)]\s*(.+?)(?=\n[A-J][\.\)]|\Z)',
            clean_opts, re.DOTALL
        )

        if option_matches:
            answer_lower = answer_text.lower().strip().rstrip('.')

            # Exact match first
            for letter, text in option_matches:
                opt_lower = text.strip().lower().rstrip('.')
                if answer_lower == opt_lower:
                    return letter

            # Substring match (30-char prefix)
            for letter, text in option_matches:
                opt_lower = text.strip().lower().rstrip('.')
                if (len(answer_lower) > 5 and len(opt_lower) > 5 and
                    (answer_lower.startswith(opt_lower[:30]) or
                     opt_lower.startswith(answer_lower[:30]))):
                    return letter

            # Containment match
            for letter, text in option_matches:
                opt_lower = text.strip().lower().rstrip('.')
                if (len(answer_lower) > 10 and len(opt_lower) > 10 and
                    (answer_lower in opt_lower or opt_lower in answer_lower)):
                    return letter

    # 4. REJECT rather than guess — don't extract random letters from prose
    #    Only accept if the answer IS a single letter
    stripped = answer_text.strip()
    if len(stripped) == 1 and stripped in "ABCDEFGHIJ":
        return stripped

    return ""


# ──────────────────────────────────────────────────────────────────────
# MODEL CALLS (fresh context per question)
# ──────────────────────────────────────────────────────────────────────

def call_gemini(prompt):
    """Call Gemini via CLI. Stateless — fresh context per call."""
    try:
        result = subprocess.run(
            ["gemini", "-y"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=180,
        )
        output = result.stdout.strip()
        if output:
            return output
    except subprocess.TimeoutExpired:
        print(f"    gemini timed out")
    except Exception as e:
        print(f"    gemini error: {e}")
    return None


def call_codex(prompt):
    """Call Codex GPT-5.4 via CLI. --ephemeral ensures fresh context."""
    import os
    ts = int(time.time())
    pid = os.getpid()
    output_file = Path(f"/tmp/upgrade-codex-{ts}-{pid}.md")
    prompt_file = Path(f"/tmp/upgrade-prompt-{ts}-{pid}.md")
    prompt_file.write_text(prompt)

    for model in ["gpt-5.4", "gpt-5.3-codex"]:
        try:
            result = subprocess.run(
                [
                    "codex", "exec", "-m", model,
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
                timeout=180,
            )
            if output_file.exists():
                content = output_file.read_text().strip()
                if content and "usage limit" not in content.lower():
                    return content, model
        except Exception:
            continue

    return None, None


def generate(prompt):
    """Generate upgrade. Primary: Gemini. Fallback: Codex."""
    output = call_gemini(prompt)
    if output:
        return output, "gemini"

    output, model = call_codex(prompt)
    if output:
        return output, model

    return None, None


def verify_blind(prompt):
    """Verify with DIFFERENT model than generator. Codex primary, Gemini fallback."""
    output, model = call_codex(prompt)
    if output:
        return output, f"codex-{model}"

    output = call_gemini(prompt)
    if output:
        return output, "gemini-verify"

    return None, None


# ──────────────────────────────────────────────────────────────────────
# DEDUPLICATION
# ──────────────────────────────────────────────────────────────────────

def load_existing_ids():
    """Load set of (source, source_id) already in output file."""
    existing = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    meta = entry.get("metadata", {})
                    key = (meta.get("source", ""), str(meta.get("source_id", "")))
                    existing.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
    return existing


def load_completed_ids():
    """Load set of already-processed example IDs from log."""
    completed = set()
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    completed.add(entry["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def make_example_id(ex):
    """Create unique ID for an example."""
    return f"{ex['dataset_name']}:{ex['id_in_dataset']}"


# ──────────────────────────────────────────────────────────────────────
# QUALITY VALIDATORS
# ──────────────────────────────────────────────────────────────────────

BANNED_UNGROUNDED = [
    "pathognomonic",
    "high-certainty evidence",
    "grade a",
    "grade b",
    "grade c",
    "[grade",
]

BANNED_TEMPLATE_PHRASES = [
    "mechanistically, this",
    "the clinical presentation is consistent with",
    "the key finding here is",
    "this is a classic case of",
    "this is a classic presentation of",
    "let's analyze",
    "interestingly,",
    "let me break",
    "let's break",
    "this clinical scenario",
    "putting it all together",
    "in summary,",
]


def validate_reasoning(reasoning, is_longevity, question_text, gt_letter=""):
    """Post-hoc quality checks on upgraded reasoning. Returns (issues, retryable)."""
    issues = []
    retryable = False
    lower = reasoning.lower()

    # Check for ungrounded confidence phrases
    for phrase in BANNED_UNGROUNDED:
        if phrase in lower:
            idx = lower.index(phrase)
            context = lower[idx:idx+100]
            has_citation = any(w in context for w in [
                "per ", "according to ", "guidelines", "trial",
                "study", "society", "association", "college",
            ])
            if not has_citation:
                issues.append(f"ungrounded_confidence:{phrase}")

    # Check for template phrases (retryable)
    for phrase in BANNED_TEMPLATE_PHRASES:
        if phrase in lower:
            issues.append(f"template_language:{phrase}")
            retryable = True

    # Check for backward reasoning: answer/diagnosis in first sentence
    sentences = [s.strip() for s in reasoning.split('.') if s.strip()]
    if sentences and gt_letter:
        first = sentences[0].lower()
        # Check if GT letter appears as "option [X]" or "[X]." in first sentence
        if f"option {gt_letter.lower()}" in first or f"answer is {gt_letter.lower()}" in first:
            issues.append("backward_reasoning:answer_in_first_sentence")
            retryable = True

    # Check for fabricated statistics not in question stem
    # Heuristic: numbers with % that aren't in the question
    q_lower = question_text.lower()
    stat_patterns = re.findall(r'\d+(?:\.\d+)?%', lower)
    for stat in stat_patterns:
        if stat not in q_lower:
            issues.append(f"fabricated_stat:{stat}")
            retryable = True
            break  # One is enough to flag

    # Check longevity contamination on non-longevity questions
    if not is_longevity:
        longevity_phrases = ["longevity", "healthspan", "hallmark of aging",
                            "metabolic perspective", "preventive perspective",
                            "upstream modifiable", "optimal vs normal"]
        for phrase in longevity_phrases:
            if phrase in lower:
                issues.append(f"longevity_contamination:{phrase}")

    return issues, retryable


# ──────────────────────────────────────────────────────────────────────
# CORE UPGRADE LOGIC
# ──────────────────────────────────────────────────────────────────────

def select_examples(per_type=10, datasets=None, shard_id=None, shard_total=None):
    """Select N examples from each dataset type, skipping already completed.

    When sharding, each shard takes every Nth example (by global index).
    """
    completed = load_completed_ids()
    print(f"Already completed: {len(completed)} examples")

    by_type = defaultdict(list)
    global_idx = 0

    with open(INPUT_FILE) as f:
        for line in f:
            ex = json.loads(line)
            eid = make_example_id(ex)
            if eid in completed:
                global_idx += 1
                continue
            ds = ex["dataset_name"]
            if datasets and ds not in datasets:
                global_idx += 1
                continue
            # Shard selection: this shard only takes its slice
            if shard_total and (global_idx % shard_total) != shard_id:
                global_idx += 1
                continue
            if per_type > 0 and len(by_type[ds]) >= per_type:
                global_idx += 1
                continue
            by_type[ds].append(ex)
            global_idx += 1

    selected = []
    for ds in sorted(by_type.keys()):
        n = len(by_type[ds])
        selected.extend(by_type[ds])
        print(f"  {ds:25s}: {n} selected")

    print(f"Total to process: {len(selected)}")
    return selected


RETRY_FEEDBACK = """Your previous attempt had these issues: {issues}

FIX THESE SPECIFICALLY:
- If "backward_reasoning": Your first sentence named the answer. Start from the
  most striking clinical FINDING instead. Build toward the answer.
- If "template_language": You used a cliché phrase. Vary your sentence openings.
  Start with the actual clinical finding, not meta-commentary.
- If "fabricated_stat": You cited a percentage not in the question stem. Remove it.
  Reason from mechanisms, not statistics you don't have.

Try again. Same rules apply."""


def build_prompt(example, gt_letter, retry_feedback=None):
    """Build the appropriate prompt based on question type and longevity relevance."""
    dataset = example["dataset_name"]
    question = example["question"]
    options = example.get("options", "")

    # Choose base prompt
    if is_literature_question(dataset, question):
        system = LITERATURE_PROMPT
    else:
        system = V43_BASE
        # Append longevity extension if relevant
        if is_longevity_relevant(question, options, dataset):
            system += LONGEVITY_EXTENSION

    # Build the full prompt
    prompt = f"""{system}

Upgrade the following example. Ground truth answer: {gt_letter}

Question: {question}
Options: {options}
Ground Truth Answer: {gt_letter}

Original Reasoning (GPT-4o + PrimeKG):
{example['reasoning'][:1500]}"""

    if retry_feedback:
        prompt += f"\n\n{retry_feedback}"

    prompt += "\n\nProduce your upgraded <reasoning> and <answer> now."

    return prompt


def _parse_and_check(raw, model, example, gt_letter):
    """Parse model output and run quality checks. Returns (result, model, status)."""
    if "<flag>UNSUPPORTED</flag>" in raw:
        return None, model, "unsupported_by_model"

    reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', raw, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', raw, re.DOTALL)

    if not reasoning_match or not answer_match:
        return None, model, "parse_failure"

    reasoning = reasoning_match.group(1).strip()
    answer = answer_match.group(1).strip()

    upgraded_letter = ""
    for char in answer:
        if char in "ABCDEFGHIJ":
            upgraded_letter = char
            break

    if upgraded_letter != gt_letter:
        return {
            "reasoning": reasoning, "answer": answer,
            "upgraded_letter": upgraded_letter, "gt_letter": gt_letter,
        }, model, "answer_changed"

    is_longevity = is_longevity_relevant(
        example["question"], example.get("options", ""), example["dataset_name"]
    )
    quality_issues, retryable = validate_reasoning(
        reasoning, is_longevity, example["question"], gt_letter
    )
    if quality_issues:
        print(f"    Quality issues: {quality_issues}")

    return {
        "reasoning": reasoning, "answer": answer,
        "upgraded_letter": upgraded_letter, "gt_letter": gt_letter,
        "generator": model, "quality_issues": quality_issues,
        "longevity_applied": is_longevity, "_retryable": retryable,
    }, model, "success"


def upgrade_single(example):
    """Upgrade a single example with v4.3 prompt. Retries once on quality failure."""
    gt_letter = extract_gt_letter(example)
    if not gt_letter:
        return None, None, "no_gt_letter"

    prompt = build_prompt(example, gt_letter)
    raw, model = generate(prompt)
    if not raw:
        return None, None, "api_failure"

    result, model, status = _parse_and_check(raw, model, example, gt_letter)

    # Retry once if quality issues are retryable
    if (status == "success" and result and result.get("_retryable")
            and result.get("quality_issues")):
        issues_str = ", ".join(result["quality_issues"])
        print(f"    Retrying due to: {issues_str}")
        feedback = RETRY_FEEDBACK.format(issues=issues_str)
        retry_prompt = build_prompt(example, gt_letter, retry_feedback=feedback)
        raw2, model2 = generate(retry_prompt)
        if raw2:
            result2, model2, status2 = _parse_and_check(
                raw2, model2, example, gt_letter
            )
            if status2 == "success":
                # Use retry if it has fewer issues
                qi1 = len(result.get("quality_issues", []))
                qi2 = len(result2.get("quality_issues", []))
                if qi2 < qi1:
                    print(f"    Retry improved: {qi1} → {qi2} issues")
                    result2["generator"] = f"{model2}-retry"
                    result2.pop("_retryable", None)
                    return result2, model2, status2
                else:
                    print(f"    Retry not better ({qi2} >= {qi1}), keeping original")

    if result:
        result.pop("_retryable", None)
    return result, model, status


def verify_reasoning(example, reasoning):
    """Independent verification: different model re-answers from reasoning alone."""
    prompt = f"""{VERIFY_SYSTEM}

Question: {example['question']}
Options: {example['options']}

Reasoning provided:
{reasoning}

Based ONLY on the reasoning above, the answer is:"""

    raw, verifier = verify_blind(prompt)
    if not raw:
        return None, None

    raw = raw.strip()
    for char in raw:
        if char in "ABCDEFGHIJ":
            return char, verifier
    return None, verifier


# ──────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────

def log_entry(example, status, result=None, model=None, verified=None, verifier=None):
    """Append to upgrade log."""
    entry = {
        "id": make_example_id(example),
        "dataset": example["dataset_name"],
        "id_in_dataset": example["id_in_dataset"],
        "status": status,
        "model": model,
        "verified": verified,
        "verifier": verifier,
        "prompt_version": "v4.3",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if result:
        entry["longevity_applied"] = result.get("longevity_applied")
        entry["quality_issues"] = result.get("quality_issues", [])
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_upgraded(example, result, verifier=None):
    """Append upgraded example to output JSONL (with dedup check)."""
    gt_letter = extract_gt_letter(example)
    source = example["dataset_name"]
    source_id = example["id_in_dataset"]

    # Normalize answer format: "[Letter]. [Text]"
    answer = result["answer"].strip()

    entry = {
        "messages": [
            {"role": "user", "content": f"{example['question']}\n\n{example['options']}\n\nAnswer:"},
            {"role": "assistant", "content": f"<think>\n{result['reasoning']}\n</think>\n{answer}"},
        ],
        "metadata": {
            "source": source,
            "source_id": source_id,
            "ground_truth": gt_letter,
            "original_reasoning_len": len(example["reasoning"]),
            "upgraded_reasoning_len": len(result["reasoning"]),
            "generator": result.get("generator"),
            "verifier": verifier,
            "longevity_applied": result.get("longevity_applied", False),
            "quality_issues": result.get("quality_issues", []),
            "prompt_version": "v4.3",
        },
    }
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_failed(example, result, reason):
    """Log failed upgrades for review."""
    entry = {
        "id": make_example_id(example),
        "dataset": example["dataset_name"],
        "question": example["question"][:200],
        "reason": reason,
        "gt_letter": extract_gt_letter(example),
        "upgraded_letter": result.get("upgraded_letter") if result else None,
        "prompt_version": "v4.3",
    }
    with open(FAILED_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MedReason V4.3 Upgrade Pipeline")
    parser.add_argument("--per-type", type=int, default=10,
                        help="Max examples per dataset type (0 = all remaining)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Only process these dataset types")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip independent verification step")
    parser.add_argument("--shard", type=str, default=None,
                        help="Shard ID/Total for parallel runs (e.g., '0/5' = shard 0 of 5)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse shard config
    global LOG_FILE, OUTPUT_FILE, FAILED_FILE, SHARD_ID, SHARD_TOTAL
    shard_id = None
    shard_total = None
    if args.shard:
        parts = args.shard.split("/")
        shard_id = int(parts[0])
        shard_total = int(parts[1])
        SHARD_ID = shard_id
        SHARD_TOTAL = shard_total
        # Each shard writes to its own files to avoid contention
        OUTPUT_FILE = OUTPUT_DIR / f"v42_upgraded_shard{shard_id}.jsonl"
        FAILED_FILE = OUTPUT_DIR / f"v42_failed_shard{shard_id}.jsonl"
        LOG_FILE = OUTPUT_DIR / f"upgrade_log_shard{shard_id}.jsonl"

    ds_label = ", ".join(args.datasets) if args.datasets else "all"
    per_label = "all remaining" if args.per_type == 0 else str(args.per_type)
    shard_label = f" — shard {shard_id}/{shard_total}" if shard_total else ""
    print("=" * 70)
    print(f"MedReason V4.3 Upgrade — {per_label} per type — datasets: {ds_label}{shard_label}")
    print("=" * 70)

    examples = select_examples(args.per_type, args.datasets, shard_id, shard_total)

    if args.dry_run:
        # Show longevity classification preview
        longevity_count = sum(
            1 for ex in examples
            if is_longevity_relevant(ex["question"], ex.get("options", ""), ex["dataset_name"])
        )
        lit_count = sum(
            1 for ex in examples
            if is_literature_question(ex["dataset_name"], ex["question"])
        )
        print(f"\n[DRY RUN]")
        print(f"  Longevity-relevant: {longevity_count}/{len(examples)}")
        print(f"  Literature mode:    {lit_count}/{len(examples)}")
        print(f"  Standard MCQ:       {len(examples) - lit_count}/{len(examples)}")

        # Show GT extraction stats
        no_gt = sum(1 for ex in examples if not extract_gt_letter(ex))
        print(f"  No GT extractable:  {no_gt}/{len(examples)}")
        return

    if not examples:
        print("\nNo examples to process (all already completed).")
        return

    # Load existing output IDs for dedup
    existing_ids = load_existing_ids()
    print(f"Existing output entries: {len(existing_ids)}")

    stats = {
        "success": 0, "answer_changed": 0, "parse_failure": 0,
        "api_failure": 0, "no_gt_letter": 0, "verify_fail": 0,
        "unsupported_by_model": 0, "duplicate": 0,
    }

    for i, ex in enumerate(examples):
        eid = make_example_id(ex)
        gt = extract_gt_letter(ex)

        # Skip if no GT
        if not gt:
            print(f"\n[{i+1}/{len(examples)}] {eid} — SKIP (no GT letter)")
            stats["no_gt_letter"] += 1
            log_entry(ex, "no_gt_letter")
            continue

        # Skip duplicates
        dedup_key = (ex["dataset_name"], str(ex["id_in_dataset"]))
        if dedup_key in existing_ids:
            print(f"\n[{i+1}/{len(examples)}] {eid} — SKIP (already in output)")
            stats["duplicate"] += 1
            continue

        is_longevity = is_longevity_relevant(
            ex["question"], ex.get("options", ""), ex["dataset_name"]
        )
        is_lit = is_literature_question(ex["dataset_name"], ex["question"])
        mode = "LIT" if is_lit else ("LONG" if is_longevity else "STD")

        print(f"\n[{i+1}/{len(examples)}] {eid} (GT={gt}, mode={mode})")
        print(f"  Q: {ex['question'][:80]}...")

        result, model, status = upgrade_single(ex)

        if status == "success":
            # Independent verification with DIFFERENT model
            verified = None
            verifier = None
            if not args.no_verify:
                print(f"  Upgraded ({model}). Cross-verifying...")
                v_letter, verifier = verify_reasoning(ex, result["reasoning"])
                verified = v_letter == gt if v_letter else None
                if verified:
                    print(f"  Verified ({verifier}): {v_letter} = {gt}")
                elif v_letter:
                    print(f"  VERIFY FAIL ({verifier}): {v_letter} != {gt} — discarding")
                    stats["verify_fail"] += 1
                    log_entry(ex, "verify_fail", result, model, False, verifier)
                    save_failed(ex, result, "verify_fail")
                    continue
                else:
                    print(f"  Verification inconclusive (API issue), keeping")

            stats["success"] += 1
            save_upgraded(ex, result, verifier)
            existing_ids.add(dedup_key)
            log_entry(ex, "success", result, model, verified, verifier)

            qi = result.get("quality_issues", [])
            qi_str = f" [QI: {', '.join(qi)}]" if qi else ""
            print(f"  SAVED ({len(result['reasoning'])} chars, longevity={is_longevity}){qi_str}")

        elif status == "answer_changed":
            print(f"  ANSWER CHANGED: {result['upgraded_letter']} != {gt} — discarding")
            stats["answer_changed"] += 1
            log_entry(ex, "answer_changed", result, model)
            save_failed(ex, result, "answer_changed")

        elif status == "unsupported_by_model":
            print(f"  Model flagged as UNSUPPORTED — rejecting")
            stats["unsupported_by_model"] += 1
            log_entry(ex, "unsupported_by_model", model=model)

        else:
            print(f"  FAILED: {status}")
            stats[status] = stats.get(status, 0) + 1
            log_entry(ex, status, model=model)

        # Brief pause to avoid rate limits
        time.sleep(1)

    # Summary
    print(f"\n{'=' * 70}")
    print("UPGRADE SUMMARY (V4.2)")
    print(f"{'=' * 70}")
    for k, v in stats.items():
        if v > 0:
            print(f"  {k:20s}: {v}")
    print(f"  {'TOTAL':20s}: {sum(stats.values())}")

    total_completed = len(load_completed_ids())
    print(f"\nCumulative completed: {total_completed}")
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            n_upgraded = sum(1 for _ in f)
        print(f"Total v4.2 upgraded examples: {n_upgraded}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
