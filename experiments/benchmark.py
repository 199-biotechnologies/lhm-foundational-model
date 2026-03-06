"""
LHM Benchmark Suite — Test and demonstrate all models.

Generates a structured transcript showing:
1. Patient trajectory predictions (Qwen3.5 generative model)
2. Risk score predictions (all classification models)
3. Model comparison on held-out patients
4. Synthetic clinical scenarios
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.mimic_loader import build_patient_records
from src.data.ehr_to_text import patient_to_text, build_training_pairs
from src.data.medical_tokenizer import tokenize_all_patients, VOCAB_SIZE, PAD_TOKEN
from src.data.feature_builder import build_tabular_features, get_feature_columns, split_data
from src.evaluation.metrics import compute_binary_metrics


OUTPUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# 1. Generative Trajectory Predictions (Qwen3.5-0.8B)
# ---------------------------------------------------------------------------

def test_generative_model(records, test_pairs, n_examples=5):
    """Generate trajectory predictions using the fine-tuned Qwen3.5 model."""
    print("\n" + "=" * 80)
    print("TEST 1: GENERATIVE TRAJECTORY PREDICTION (Qwen3.5-0.8B)")
    print("=" * 80)
    print("Task: Given patient history, predict future visits (diagnoses + labs)")
    print("-" * 80)

    model_path = Path("experiments/exp1_text_llm/outputs/model")
    if not model_path.exists():
        print("  [SKIP] Qwen3.5 model not found. Run experiment 1 first.")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float32,
        trust_remote_code=True,
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    for i, pair in enumerate(test_pairs[:n_examples]):
        print(f"\n{'─' * 80}")
        print(f"PATIENT {i+1} (subject_id: {pair['subject_id']})")
        print(f"History: {pair['n_history']} visits | Future to predict: {pair['n_future']} visits")
        print(f"{'─' * 80}")

        # Show input (truncated)
        input_text = pair["input"]
        print(f"\n  INPUT (patient history):")
        for line in input_text.split("\n"):
            print(f"    {line}")

        # Generate prediction
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                )
                predicted = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
            except RuntimeError:
                predicted = "[generation failed]"

        print(f"\n  PREDICTED (model output):")
        for line in predicted.strip().split("\n")[:5]:
            print(f"    {line}")

        print(f"\n  ACTUAL (ground truth):")
        actual = pair["output"]
        for line in actual.split("\n")[:5]:
            print(f"    {line}")

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# 2. Classification Risk Scores (All models)
# ---------------------------------------------------------------------------

def test_classification_models(records):
    """Run risk prediction on test patients across all classification models."""
    print("\n" + "=" * 80)
    print("TEST 2: CLINICAL RISK SCORES (All Classification Models)")
    print("=" * 80)
    print("Task: Predict 30-day readmission and in-hospital mortality risk")
    print("-" * 80)

    # Prepare data
    tokenized, code_to_idx, lab_quantiles = tokenize_all_patients(records, max_tokens=512)
    np.random.seed(42)
    np.random.shuffle(tokenized)
    n_test = max(1, int(len(tokenized) * 0.15))
    test_patients = tokenized[:n_test]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load each model and get predictions
    model_results = {}

    # XGBoost
    try:
        import xgboost as xgb
        df = build_tabular_features(records)
        feature_cols = get_feature_columns(df)
        _, _, test_df = split_data(df)
        X_test = test_df[feature_cols].values

        xgb_readm = xgb.XGBClassifier()
        # Re-train quickly for predictions (XGBoost is fast)
        train_df, val_df, _ = split_data(df)
        X_train = train_df[feature_cols].values
        y_train_r = train_df["readmission_30d"].values
        X_val = val_df[feature_cols].values
        y_val_r = val_df["readmission_30d"].values

        if len(np.unique(y_train_r)) > 1:
            xgb_model = xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                eval_metric="logloss", early_stopping_rounds=50, random_state=42,
            )
            xgb_model.fit(X_train, y_train_r, eval_set=[(X_val, y_val_r)], verbose=False)
            xgb_preds = xgb_model.predict_proba(X_test)[:, 1]
            model_results["XGBoost"] = {
                "readmission_probs": xgb_preds,
                "readmission_true": test_df["readmission_30d"].values,
            }
    except Exception as e:
        print(f"  XGBoost: {e}")

    # Neural models
    neural_models = [
        ("EHRMamba", "experiments/exp2_mamba/outputs/best_model.pt", "mamba"),
        ("Continuous-Time", "experiments/exp3_continuous_time/outputs/best_model.pt", "continuous"),
        ("Medical Decoder", "experiments/exp4_medical_tokens/outputs/best_model.pt", "decoder"),
        ("Hybrid LHM", "experiments/exp5_hybrid/outputs/best_model.pt", "hybrid"),
    ]

    for name, path, model_type in neural_models:
        if not Path(path).exists():
            continue
        try:
            if model_type == "mamba":
                from experiments.exp2_mamba.run import EHRMamba, EHRTokenDataset
                model = EHRMamba(vocab_size=VOCAB_SIZE, d_model=128, n_layers=4, d_state=16).to(device)
            elif model_type == "continuous":
                from experiments.exp3_continuous_time.run import ContinuousTimeEHR, ContinuousTimeDataset
                model = ContinuousTimeEHR(vocab_size=VOCAB_SIZE, d_model=128, n_layers=4, n_heads=4).to(device)
            elif model_type == "decoder":
                from experiments.exp4_medical_tokens.run import MedicalTokenDecoder
                model = MedicalTokenDecoder(vocab_size=VOCAB_SIZE, d_model=128, n_layers=4, n_heads=4).to(device)
            elif model_type == "hybrid":
                from experiments.exp5_hybrid.run import HybridLHM
                model = HybridLHM(vocab_size=VOCAB_SIZE, d_model=128, n_layers=8, n_heads=4, d_state=16).to(device)

            model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
            model.eval()

            readm_preds = []
            mort_preds = []
            readm_true = []
            mort_true = []

            for tp in test_patients:
                tokens = tp.token_ids[:256]
                types = tp.token_types[:256]
                times = tp.timestamps[:256]
                pad = 256 - len(tokens)

                t_ids = torch.tensor(tokens + [0]*pad, dtype=torch.long).unsqueeze(0).to(device)
                t_types = torch.tensor(types + [0]*pad, dtype=torch.long).unsqueeze(0).to(device)
                t_times = torch.tensor(times + [0.0]*pad, dtype=torch.float32).unsqueeze(0).to(device)
                t_mask = torch.tensor([1.0]*len(tokens) + [0.0]*pad, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    if model_type in ("continuous", "hybrid"):
                        out = model(t_ids, t_types, t_times, t_mask)
                    else:
                        out = model(t_ids, t_types, t_mask)

                readm_preds.append(torch.sigmoid(out["readmission_logit"]).item())
                mort_preds.append(torch.sigmoid(out["mortality_logit"]).item())
                readm_true.append(tp.readmission_30d)
                mort_true.append(tp.mortality)

            model_results[name] = {
                "readmission_probs": np.array(readm_preds),
                "readmission_true": np.array(readm_true),
                "mortality_probs": np.array(mort_preds),
                "mortality_true": np.array(mort_true),
            }
            del model

        except Exception as e:
            print(f"  {name}: Error — {e}")

    # Print results table
    print(f"\n  {'Model':<25} {'Readm Risk':>12} {'Mort Risk':>12} {'Actual Readm':>14} {'Actual Mort':>12}")
    print(f"  {'─'*75}")

    for patient_idx in range(min(len(test_patients), 10)):
        tp = test_patients[patient_idx]
        print(f"\n  Patient {tp.subject_id} (tokens: {len(tp.token_ids)})")

        for model_name, results in model_results.items():
            if patient_idx < len(results["readmission_probs"]):
                r_prob = results["readmission_probs"][patient_idx]
                r_true = results["readmission_true"][patient_idx]
                m_prob = results.get("mortality_probs", [0]*100)[patient_idx] if "mortality_probs" in results else 0
                m_true = results.get("mortality_true", [0]*100)[patient_idx] if "mortality_true" in results else 0
                print(f"    {model_name:<23} {r_prob:>10.1%}   {m_prob:>10.1%}   {int(r_true):>12}   {int(m_true):>10}")

    # Aggregate metrics
    print(f"\n\n  AGGREGATE METRICS (test set)")
    print(f"  {'Model':<25} {'Readm AUROC':>12} {'Readm AUPRC':>12} {'Mort AUROC':>12}")
    print(f"  {'─'*63}")

    for model_name, results in model_results.items():
        r_metrics = compute_binary_metrics(results["readmission_true"], results["readmission_probs"])
        m_metrics = {"auroc": 0.0, "auprc": 0.0}
        if "mortality_true" in results:
            m_metrics = compute_binary_metrics(results["mortality_true"], results["mortality_probs"])
        print(f"    {model_name:<23} {r_metrics['auroc']:>10.4f}   {r_metrics['auprc']:>10.4f}   {m_metrics['auroc']:>10.4f}")


# ---------------------------------------------------------------------------
# 3. Clinical Scenario Tests
# ---------------------------------------------------------------------------

def test_clinical_scenarios(records):
    """Test models on interpretable clinical scenarios."""
    print("\n" + "=" * 80)
    print("TEST 3: CLINICAL SCENARIO ANALYSIS")
    print("=" * 80)
    print("Comparing model behavior on specific patient profiles")
    print("-" * 80)

    # Find interesting patients from our data
    scenarios = []

    for subject_id in records["subject_id"].unique():
        patient = records[records["subject_id"] == subject_id].sort_values("admittime")
        n_visits = len(patient)
        last = patient.iloc[-1]
        mortality = int(last.get("hospital_expire_flag", 0))

        if n_visits >= 3:
            # Multi-visit patient
            diags = last.get("diagnoses", [])
            labs = last.get("labs", {})
            age = last.get("anchor_age", 0)

            scenarios.append({
                "subject_id": subject_id,
                "age": age,
                "gender": last.get("gender", "?"),
                "n_visits": n_visits,
                "mortality": mortality,
                "n_diagnoses": len(diags) if isinstance(diags, list) else 0,
                "profile": "frequent_readmitter" if n_visits >= 4 else "moderate",
            })

    # Sort by number of visits (most complex patients first)
    scenarios.sort(key=lambda x: x["n_visits"], reverse=True)

    print(f"\n  Found {len(scenarios)} patients with 3+ visits\n")

    print(f"  {'Subject ID':>12} {'Age':>5} {'Sex':>5} {'Visits':>7} {'Diagnoses':>10} {'Mortality':>10} {'Profile':<20}")
    print(f"  {'─'*75}")

    for s in scenarios[:15]:
        print(f"  {s['subject_id']:>12} {s['age']:>5} {s['gender']:>5} {s['n_visits']:>7} "
              f"{s['n_diagnoses']:>10} {s['mortality']:>10} {s['profile']:<20}")

    # Show detailed trajectory for top patients
    print(f"\n\n  DETAILED TRAJECTORIES")
    print(f"  {'─'*75}")

    for s in scenarios[:3]:
        text = patient_to_text(records, s["subject_id"])
        print(f"\n  Patient {s['subject_id']} ({s['age']}{s['gender']}, {s['n_visits']} visits):")
        for line in text.split("\n"):
            print(f"    {line}")
        print()


# ---------------------------------------------------------------------------
# 4. Known Medical Benchmarks Info
# ---------------------------------------------------------------------------

def print_benchmark_info():
    """Print information about standard medical benchmarks we could test."""
    print("\n" + "=" * 80)
    print("AVAILABLE MEDICAL BENCHMARKS (for Phase 2)")
    print("=" * 80)

    benchmarks = [
        {
            "name": "EHRSHOT",
            "source": "Stanford, 2023",
            "description": "Few-shot EHR benchmark with 15 clinical tasks",
            "tasks": "Lab abnormality, mortality, readmission, LOS, diagnoses",
            "data": "Requires STARR-OMOP (Stanford) or mapping to MIMIC",
            "url": "github.com/som-shahlab/ehrshot-benchmark",
            "feasibility": "HIGH — can adapt our models",
        },
        {
            "name": "MIMIC-III Benchmarks",
            "source": "MIT/Harvard, 2019",
            "description": "Standardized clinical prediction tasks on MIMIC-III",
            "tasks": "Mortality (48h), decompensation, LOS, phenotyping",
            "data": "MIMIC-III (requires PhysioNet credentialing)",
            "url": "github.com/YerevaNN/mimic3-benchmarks",
            "feasibility": "HIGH — same data source, easy to adapt",
        },
        {
            "name": "MedQA (USMLE)",
            "source": "Various",
            "description": "Medical question answering (USMLE-style)",
            "tasks": "Multiple choice medical knowledge questions",
            "data": "Public",
            "url": "github.com/jind11/MedQA",
            "feasibility": "LOW — our model predicts trajectories, not Q&A",
        },
        {
            "name": "Clinical Trial Outcome Prediction",
            "source": "Various",
            "description": "Predict trial success from patient characteristics",
            "tasks": "Binary outcome prediction",
            "data": "ClinicalTrials.gov + MIMIC",
            "feasibility": "MEDIUM — aligns with trajectory prediction thesis",
        },
        {
            "name": "eICU Collaborative Research Database",
            "source": "MIT/Philips",
            "description": "Multi-center ICU data for benchmarking",
            "tasks": "Mortality, LOS, ventilation prediction",
            "data": "Requires PhysioNet credentialing",
            "feasibility": "HIGH — validates generalization across hospitals",
        },
        {
            "name": "OMOP CDM Tasks",
            "source": "OHDSI",
            "description": "Standardized observational health data tasks",
            "tasks": "Drug safety, disease onset, treatment effect",
            "data": "Any OMOP-mapped database",
            "feasibility": "HIGH — standard format, many hospitals use it",
        },
    ]

    for b in benchmarks:
        print(f"\n  {b['name']} ({b['source']})")
        print(f"    {b['description']}")
        print(f"    Tasks: {b['tasks']}")
        print(f"    Data: {b['data']}")
        print(f"    Feasibility for LHM: {b['feasibility']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("╔" + "═" * 78 + "╗")
    print("║" + " LHM BENCHMARK SUITE — Architecture Shootout Results ".center(78) + "║")
    print("║" + " 199 Biotechnologies | March 2026 ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    records = build_patient_records()
    pairs = build_training_pairs(records, min_visits=2)

    # Split test pairs
    subject_ids = list(set(p["subject_id"] for p in pairs))
    np.random.seed(42)
    np.random.shuffle(subject_ids)
    n_test = max(1, int(len(subject_ids) * 0.15))
    test_subj = set(subject_ids[:n_test])
    test_pairs = [p for p in pairs if p["subject_id"] in test_subj]

    # Run all tests
    test_clinical_scenarios(records)
    test_classification_models(records)
    test_generative_model(records, test_pairs, n_examples=3)
    print_benchmark_info()

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
