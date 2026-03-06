"""
Comprehensive Medical Benchmark Suite for LHM / Improbability-0.8B.

Uses LOG-LIKELIHOOD evaluation (not generation) for reliable MCQ scoring.
For each question, computes P(answer_token | prompt + "The answer is: ")
and picks the option with highest probability.

Benchmarks:
1. MedQA (USMLE) — 20 medical licensing exam questions
2. MedMCQA — 15 Indian medical entrance exam questions
3. MMLU-Medical — 10 medical subset MMLU questions
4. Drug Interactions — 5 pharmacology interaction questions
5. Clinical Reasoning — 5 multi-step clinical reasoning questions
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OUTPUT_DIR = Path(__file__).parent / "outputs"

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
FINETUNED_PATH = "experiments/exp1_text_llm/outputs/model"


def load_model(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def score_options_loglikelihood(model, tokenizer, prompt, options, device):
    """
    Compute log-likelihood of each answer option given the prompt.
    Returns dict of {option_letter: log_prob}.

    For each option (A, B, C, D), we compute:
      P(" A" | prompt) using the model's next-token logits.
    """
    # Tokenize prompt
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    prompt_ids = {k: v.to(device) for k, v in prompt_ids.items()}

    with torch.no_grad():
        outputs = model(**prompt_ids)
        # logits shape: (1, seq_len, vocab_size)
        # We want the logits at the last token position (prediction for next token)
        last_logits = outputs.logits[0, -1, :]  # (vocab_size,)
        log_probs = F.log_softmax(last_logits, dim=-1)

    scores = {}
    for letter in options:
        # Try multiple token representations of the answer
        candidates = [f" {letter}", letter, f" {letter}.", f" ({letter})"]
        best_score = float("-inf")
        for candidate in candidates:
            token_ids = tokenizer.encode(candidate, add_special_tokens=False)
            if token_ids:
                # Use the first token's log-prob
                score = log_probs[token_ids[0]].item()
                best_score = max(best_score, score)
        scores[letter] = best_score

    return scores


def run_benchmark_loglik(name, questions, model, tokenizer, device):
    """Run benchmark using log-likelihood scoring."""
    correct = 0
    total = len(questions)
    details = []

    for q_data in questions:
        opts_text = "\n".join(f"{k}. {v}" for k, v in q_data["options"].items())
        prompt = f"Question: {q_data['q']}\n\n{opts_text}\n\nThe answer is:"

        scores = score_options_loglikelihood(
            model, tokenizer, prompt, list(q_data["options"].keys()), device
        )
        predicted = max(scores, key=scores.get)
        is_correct = predicted == q_data["answer"]
        if is_correct:
            correct += 1

        details.append({
            "question": q_data["q"][:80],
            "correct_answer": q_data["answer"],
            "predicted": predicted,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "is_correct": is_correct,
        })

    accuracy = correct / total
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 4),
        "details": details,
    }


# ── Benchmark Questions ──────────────────────────────────────────────────

MEDQA_QUESTIONS = [
    {"q": "A 65-year-old man presents with progressive dyspnea and bilateral lower extremity edema. Chest X-ray shows cardiomegaly and bilateral pleural effusions. Which of the following is the most likely diagnosis?",
     "options": {"A": "Pneumonia", "B": "Congestive heart failure", "C": "Pulmonary embolism", "D": "COPD exacerbation"}, "answer": "B"},
    {"q": "A 45-year-old woman presents with fatigue, weight gain, and cold intolerance. Lab results show elevated TSH and low free T4. What is the most appropriate initial treatment?",
     "options": {"A": "Propylthiouracil", "B": "Levothyroxine", "C": "Methimazole", "D": "Radioactive iodine"}, "answer": "B"},
    {"q": "A 30-year-old man presents with acute onset severe headache described as 'the worst headache of my life.' What is the most important next step?",
     "options": {"A": "Prescribe acetaminophen", "B": "CT head without contrast", "C": "MRI brain", "D": "Lumbar puncture"}, "answer": "B"},
    {"q": "A 55-year-old diabetic patient presents with HbA1c of 9.2% on metformin monotherapy. What is the most appropriate next step?",
     "options": {"A": "Increase metformin dose", "B": "Add a second oral agent or GLP-1 agonist", "C": "Start insulin immediately", "D": "Dietary counseling only"}, "answer": "B"},
    {"q": "A patient with atrial fibrillation has a CHA2DS2-VASc score of 3. What is the recommended anticoagulation strategy?",
     "options": {"A": "No anticoagulation needed", "B": "Aspirin only", "C": "Direct oral anticoagulant (DOAC)", "D": "Clopidogrel"}, "answer": "C"},
    {"q": "Which electrolyte abnormality is most commonly associated with prolonged QT interval on ECG?",
     "options": {"A": "Hyperkalemia", "B": "Hypomagnesemia", "C": "Hypernatremia", "D": "Hypercalcemia"}, "answer": "B"},
    {"q": "A 70-year-old man presents with painless jaundice and weight loss. CT shows a mass in the head of the pancreas. What is the most likely diagnosis?",
     "options": {"A": "Acute pancreatitis", "B": "Cholecystitis", "C": "Pancreatic adenocarcinoma", "D": "Chronic pancreatitis"}, "answer": "C"},
    {"q": "Which class of medications is first-line for reducing mortality in patients with heart failure with reduced ejection fraction (HFrEF)?",
     "options": {"A": "Calcium channel blockers", "B": "ACE inhibitors / ARBs / ARNI", "C": "Digoxin", "D": "Antiarrhythmics"}, "answer": "B"},
    {"q": "A 25-year-old woman presents with sudden onset pleuritic chest pain and dyspnea. She takes oral contraceptives. D-dimer is elevated. What is the next best diagnostic step?",
     "options": {"A": "Chest X-ray", "B": "CT pulmonary angiography", "C": "Echocardiogram", "D": "Ventilation-perfusion scan"}, "answer": "B"},
    {"q": "In a patient with suspected acute myocardial infarction, which cardiac biomarker rises earliest?",
     "options": {"A": "Troponin I", "B": "CK-MB", "C": "Myoglobin", "D": "LDH"}, "answer": "C"},
    {"q": "A 60-year-old man with COPD presents with acute worsening dyspnea, fever, and increased sputum production. ABG shows pH 7.32, pCO2 55, pO2 58. What is the most appropriate initial management?",
     "options": {"A": "Intubation and mechanical ventilation", "B": "Non-invasive positive pressure ventilation (NIPPV)", "C": "High-flow nasal cannula only", "D": "Nebulized albuterol only"}, "answer": "B"},
    {"q": "Which medication should be avoided in patients with a history of angioedema from ACE inhibitors?",
     "options": {"A": "Amlodipine", "B": "Losartan", "C": "Lisinopril", "D": "Metoprolol"}, "answer": "C"},
    {"q": "A patient presents with polyuria, polydipsia, and a random glucose of 280 mg/dL. What is the diagnostic threshold for diabetes mellitus using fasting plasma glucose?",
     "options": {"A": "100 mg/dL", "B": "110 mg/dL", "C": "126 mg/dL", "D": "140 mg/dL"}, "answer": "C"},
    {"q": "What is the most common cause of community-acquired pneumonia in adults?",
     "options": {"A": "Haemophilus influenzae", "B": "Streptococcus pneumoniae", "C": "Klebsiella pneumoniae", "D": "Staphylococcus aureus"}, "answer": "B"},
    {"q": "A 50-year-old patient is found to have a serum sodium of 118 mEq/L. What is the maximum safe rate of sodium correction to avoid osmotic demyelination syndrome?",
     "options": {"A": "4-6 mEq/L per 24 hours", "B": "8-10 mEq/L per 24 hours", "C": "12-14 mEq/L per 24 hours", "D": "16-18 mEq/L per 24 hours"}, "answer": "B"},
    {"q": "Which vitamin deficiency causes scurvy?",
     "options": {"A": "Vitamin A", "B": "Vitamin B12", "C": "Vitamin C", "D": "Vitamin D"}, "answer": "C"},
    {"q": "A 40-year-old woman presents with bilateral hand joint pain and morning stiffness lasting >1 hour. RF and anti-CCP are positive. What is the most likely diagnosis?",
     "options": {"A": "Osteoarthritis", "B": "Rheumatoid arthritis", "C": "Gout", "D": "Systemic lupus erythematosus"}, "answer": "B"},
    {"q": "What is the first-line treatment for H. pylori eradication?",
     "options": {"A": "PPI + amoxicillin + metronidazole", "B": "PPI + amoxicillin + clarithromycin", "C": "H2 blocker + bismuth", "D": "PPI monotherapy"}, "answer": "B"},
    {"q": "A patient with cirrhosis develops confusion and asterixis. Serum ammonia is elevated. What is the most appropriate treatment?",
     "options": {"A": "Neomycin only", "B": "Lactulose", "C": "Flumazenil", "D": "Naloxone"}, "answer": "B"},
    {"q": "Which screening test is recommended for colorectal cancer starting at age 45?",
     "options": {"A": "CEA levels annually", "B": "Colonoscopy every 10 years", "C": "Barium enema every 5 years", "D": "CT abdomen annually"}, "answer": "B"},
]

MEDMCQA_QUESTIONS = [
    {"q": "The enzyme deficient in Gaucher's disease is:",
     "options": {"A": "Sphingomyelinase", "B": "Glucocerebrosidase", "C": "Hexosaminidase A", "D": "Alpha-galactosidase"}, "answer": "B"},
    {"q": "Ghon complex is seen in:",
     "options": {"A": "Primary tuberculosis", "B": "Secondary tuberculosis", "C": "Miliary tuberculosis", "D": "Tuberculoma"}, "answer": "A"},
    {"q": "Corkscrew esophagus is seen in:",
     "options": {"A": "Achalasia", "B": "Diffuse esophageal spasm", "C": "GERD", "D": "Esophageal web"}, "answer": "B"},
    {"q": "The most common site of hypertensive intracerebral hemorrhage is:",
     "options": {"A": "Putamen", "B": "Thalamus", "C": "Pons", "D": "Cerebellum"}, "answer": "A"},
    {"q": "Reed-Sternberg cells are pathognomonic of:",
     "options": {"A": "Non-Hodgkin lymphoma", "B": "Hodgkin lymphoma", "C": "Multiple myeloma", "D": "Chronic lymphocytic leukemia"}, "answer": "B"},
    {"q": "The most common type of renal stone is:",
     "options": {"A": "Uric acid", "B": "Calcium oxalate", "C": "Struvite", "D": "Cystine"}, "answer": "B"},
    {"q": "Trousseau sign is positive in:",
     "options": {"A": "Hypercalcemia", "B": "Hypocalcemia", "C": "Hyperkalemia", "D": "Hyponatremia"}, "answer": "B"},
    {"q": "The most common cause of Cushing syndrome is:",
     "options": {"A": "Adrenal adenoma", "B": "Pituitary adenoma", "C": "Exogenous corticosteroids", "D": "Ectopic ACTH"}, "answer": "C"},
    {"q": "Port wine stain is characteristic of:",
     "options": {"A": "Turner syndrome", "B": "Sturge-Weber syndrome", "C": "Tuberous sclerosis", "D": "Neurofibromatosis"}, "answer": "B"},
    {"q": "Kayser-Fleischer ring is seen in:",
     "options": {"A": "Hemochromatosis", "B": "Wilson disease", "C": "Primary biliary cholangitis", "D": "Alpha-1 antitrypsin deficiency"}, "answer": "B"},
    {"q": "The most sensitive marker for acute pancreatitis is:",
     "options": {"A": "Serum amylase", "B": "Serum lipase", "C": "CRP", "D": "LDH"}, "answer": "B"},
    {"q": "Philadelphia chromosome is associated with:",
     "options": {"A": "AML", "B": "CML", "C": "ALL", "D": "CLL"}, "answer": "B"},
    {"q": "Curling ulcer is associated with:",
     "options": {"A": "Head injury", "B": "Burns", "C": "Sepsis", "D": "Uremia"}, "answer": "B"},
    {"q": "Mallory-Weiss tear involves:",
     "options": {"A": "Esophageal perforation", "B": "Gastroesophageal junction mucosal tear", "C": "Duodenal ulcer", "D": "Gastric varices"}, "answer": "B"},
    {"q": "The triad of Horner syndrome includes all EXCEPT:",
     "options": {"A": "Miosis", "B": "Ptosis", "C": "Anhidrosis", "D": "Mydriasis"}, "answer": "D"},
]

MMLU_MEDICAL = [
    {"q": "Which of the following is NOT a function of the liver?",
     "options": {"A": "Gluconeogenesis", "B": "Production of erythropoietin", "C": "Synthesis of albumin", "D": "Bile production"}, "answer": "B", "subject": "anatomy"},
    {"q": "The sinoatrial node is located in the:",
     "options": {"A": "Left atrium", "B": "Right atrium", "C": "Left ventricle", "D": "Interventricular septum"}, "answer": "B", "subject": "anatomy"},
    {"q": "Which immunoglobulin crosses the placenta?",
     "options": {"A": "IgA", "B": "IgM", "C": "IgG", "D": "IgE"}, "answer": "C", "subject": "clinical_knowledge"},
    {"q": "The most common chromosomal abnormality in spontaneous abortions is:",
     "options": {"A": "Monosomy X", "B": "Trisomy 16", "C": "Trisomy 21", "D": "Triploidy"}, "answer": "B", "subject": "medical_genetics"},
    {"q": "Hemophilia A is caused by deficiency of factor:",
     "options": {"A": "VII", "B": "VIII", "C": "IX", "D": "X"}, "answer": "B", "subject": "clinical_knowledge"},
    {"q": "The normal range for serum potassium is:",
     "options": {"A": "2.5-3.5 mEq/L", "B": "3.5-5.0 mEq/L", "C": "5.0-6.5 mEq/L", "D": "6.5-8.0 mEq/L"}, "answer": "B", "subject": "clinical_knowledge"},
    {"q": "Which nerve is most commonly injured in fracture of the surgical neck of humerus?",
     "options": {"A": "Radial nerve", "B": "Axillary nerve", "C": "Ulnar nerve", "D": "Musculocutaneous nerve"}, "answer": "B", "subject": "anatomy"},
    {"q": "The Glasgow Coma Scale ranges from:",
     "options": {"A": "0 to 15", "B": "3 to 15", "C": "1 to 15", "D": "3 to 12"}, "answer": "B", "subject": "clinical_knowledge"},
    {"q": "Wernicke encephalopathy is caused by deficiency of:",
     "options": {"A": "Vitamin B12", "B": "Thiamine (B1)", "C": "Niacin (B3)", "D": "Folate"}, "answer": "B", "subject": "clinical_knowledge"},
    {"q": "Which type of hypersensitivity reaction is involved in contact dermatitis?",
     "options": {"A": "Type I", "B": "Type II", "C": "Type III", "D": "Type IV"}, "answer": "D", "subject": "clinical_knowledge"},
]

DRUG_INTERACTION_QUESTIONS = [
    {"q": "A patient on warfarin is prescribed amiodarone. What is the expected effect on INR?",
     "options": {"A": "Decreased INR", "B": "Increased INR (dangerous)", "C": "No change", "D": "Unpredictable"}, "answer": "B"},
    {"q": "Which of these medications should NOT be combined with MAO inhibitors?",
     "options": {"A": "Acetaminophen", "B": "SSRIs (serotonin syndrome risk)", "C": "Aspirin", "D": "Famotidine"}, "answer": "B"},
    {"q": "A patient on metformin is scheduled for a CT with IV contrast. What should be done?",
     "options": {"A": "Continue metformin normally", "B": "Hold metformin before and 48h after contrast", "C": "Double the metformin dose", "D": "Switch to insulin permanently"}, "answer": "B"},
    {"q": "Combining an ACE inhibitor with potassium-sparing diuretics increases the risk of:",
     "options": {"A": "Hypokalemia", "B": "Hyperkalemia", "C": "Hyponatremia", "D": "Metabolic alkalosis"}, "answer": "B"},
    {"q": "Grapefruit juice significantly affects the metabolism of which drug?",
     "options": {"A": "Metformin", "B": "Simvastatin", "C": "Aspirin", "D": "Metoprolol"}, "answer": "B"},
]

CLINICAL_REASONING = [
    {"q": "A 68-year-old man with diabetes, hypertension, and CKD stage 3 presents with acute chest pain. Troponin is 0.8 ng/mL. ECG shows ST depression in V4-V6. What is the TIMI risk score component present?",
     "options": {"A": "Age < 65", "B": "Known CAD with >=50% stenosis", "C": "At least 3 cardiac risk factors", "D": "Prior aspirin use"}, "answer": "C"},
    {"q": "A patient with severe sepsis has MAP of 55 mmHg despite 30 mL/kg crystalloid bolus. What is the next step?",
     "options": {"A": "Additional fluid bolus", "B": "Start norepinephrine", "C": "Start dobutamine", "D": "Packed red blood cell transfusion"}, "answer": "B"},
    {"q": "An 80-year-old nursing home resident develops fever, productive cough, and right lower lobe consolidation. What is the CURB-65 score consideration?",
     "options": {"A": "Age alone does not contribute", "B": "Age >=65 contributes 1 point", "C": "Nursing home status adds 2 points", "D": "Consolidation adds 1 point"}, "answer": "B"},
    {"q": "A patient with acute kidney injury has K+ of 7.2 mEq/L with peaked T waves on ECG. What is the FIRST intervention?",
     "options": {"A": "Oral kayexalate", "B": "IV calcium gluconate", "C": "IV insulin + glucose", "D": "Emergent hemodialysis"}, "answer": "B"},
    {"q": "A cirrhotic patient develops spontaneous bacterial peritonitis. What is the empiric antibiotic of choice?",
     "options": {"A": "Metronidazole", "B": "Cefotaxime (third-gen cephalosporin)", "C": "Vancomycin", "D": "Ciprofloxacin"}, "answer": "B"},
]


BENCHMARKS = {
    "MedQA (USMLE)": MEDQA_QUESTIONS,
    "MedMCQA": MEDMCQA_QUESTIONS,
    "MMLU-Medical": MMLU_MEDICAL,
    "Drug Interactions": DRUG_INTERACTION_QUESTIONS,
    "Clinical Reasoning": CLINICAL_REASONING,
}

PUBLISHED_BASELINES = {
    "MedQA (USMLE)": {"Random": 25.0, "BioBERT": 36.7, "PubMedBERT": 38.3, "BioGPT": 44.1, "GPT-4": 86.7},
    "MedMCQA": {"Random": 25.0, "PubMedBERT": 32.1, "BioGPT": 37.0, "GPT-4": 72.0},
    "MMLU-Medical": {"Random": 25.0, "Llama-2-7B": 35.0, "GPT-4": 87.0},
    "Drug Interactions": {"Random": 25.0},
    "Clinical Reasoning": {"Random": 25.0},
}


def main():
    print("=" * 70)
    print(" MEDICAL BENCHMARK SUITE (Log-Likelihood Evaluation)")
    print(" Models: Base Qwen3.5-0.8B vs Improbability-0.8B")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()

    # Load models
    print("\n1. Loading models...")
    base_model, base_tok = load_model(MODEL_NAME, device)
    print("   Base Qwen3.5-0.8B loaded.")

    ft_path = Path(__file__).parent.parent.parent / FINETUNED_PATH
    if ft_path.exists():
        ft_model, ft_tok = load_model(str(ft_path), device)
        print("   Improbability-0.8B loaded.")
        has_ft = True
    else:
        print("   Improbability-0.8B not found — running base only.")
        has_ft = False

    all_results = {}

    for bench_name, questions in BENCHMARKS.items():
        print(f"\n{'─' * 70}")
        print(f" {bench_name} ({len(questions)} questions)")
        print(f"{'─' * 70}")

        # Base model
        print("   Running Base Qwen3.5-0.8B...")
        base_result = run_benchmark_loglik(bench_name, questions, base_model, base_tok, device)
        print(f"   Base: {base_result['correct']}/{base_result['total']} = {base_result['accuracy']:.1%}")

        result = {"base": base_result}

        # Fine-tuned model
        if has_ft:
            print("   Running Improbability-0.8B...")
            ft_result = run_benchmark_loglik(bench_name, questions, ft_model, ft_tok, device)
            print(f"   Improbability: {ft_result['correct']}/{ft_result['total']} = {ft_result['accuracy']:.1%}")
            result["improbability"] = ft_result

        if bench_name in PUBLISHED_BASELINES:
            result["published_baselines"] = PUBLISHED_BASELINES[bench_name]

        all_results[bench_name] = result

    elapsed = time.time() - t0

    # Save (strip per-question details from main results, save separately)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Summary results (without per-question details)
    summary = {}
    for name, r in all_results.items():
        summary[name] = {}
        for key, val in r.items():
            if isinstance(val, dict) and "details" in val:
                summary[name][key] = {k: v for k, v in val.items() if k != "details"}
            else:
                summary[name][key] = val
    summary["_metadata"] = {
        "method": "log-likelihood",
        "elapsed_seconds": round(elapsed, 1),
        "device": str(device),
    }

    with open(OUTPUT_DIR / "benchmark_suite.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Detailed results (with per-question breakdown)
    with open(OUTPUT_DIR / "benchmark_details.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'=' * 70}")
    print(" BENCHMARK RESULTS SUMMARY (Log-Likelihood Method)")
    print(f"{'=' * 70}")
    print(f" {'Benchmark':<25} {'Base':>8} {'Improbability':>14} {'Random':>8} {'GPT-4':>8}")
    print(f" {'-'*25} {'-'*8} {'-'*14} {'-'*8} {'-'*8}")

    for name, r in all_results.items():
        base_pct = f"{r['base']['accuracy']:.1%}"
        ft_pct = f"{r.get('improbability', {}).get('accuracy', 0):.1%}" if has_ft else "—"
        baselines = r.get("published_baselines", {})
        rand_pct = "25.0%"
        gpt4 = f"{baselines.get('GPT-4', 0):.1f}%" if "GPT-4" in baselines else "—"
        print(f" {name:<25} {base_pct:>8} {ft_pct:>14} {rand_pct:>8} {gpt4:>8}")

    print(f"{'=' * 70}")
    print(f" Evaluation method: Log-likelihood (standard for small LMs)")
    print(f" Total time: {elapsed:.1f}s")
    print(f"\nResults saved to {OUTPUT_DIR}")

    # Cleanup
    del base_model
    if has_ft:
        del ft_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
