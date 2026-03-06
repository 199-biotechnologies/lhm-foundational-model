"""
Experiment 0: XGBoost Baseline

Standard gradient boosting on tabular EHR features.
This sets the performance bar that all architecture experiments must beat.
"""

import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.mimic_loader import build_patient_records
from src.data.feature_builder import build_tabular_features, get_feature_columns, split_data
from src.evaluation.metrics import ExperimentResults, compute_binary_metrics, print_comparison_table


OUTPUT_DIR = Path(__file__).parent / "outputs"


def train_binary_classifier(
    X_train, y_train, X_val, y_val, task_name: str
) -> tuple[xgb.XGBClassifier, dict]:
    """Train XGBoost binary classifier with early stopping."""
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def run_experiment():
    """Run the full XGBoost baseline experiment."""
    print("=" * 60)
    print("EXPERIMENT 0: XGBoost Baseline")
    print("=" * 60)

    start_time = time.time()

    # Load and prepare data
    print("\n1. Loading MIMIC-IV data...")
    records = build_patient_records()
    df = build_tabular_features(records)
    feature_cols = get_feature_columns(df)

    print(f"   Patients: {df['subject_id'].nunique()}")
    print(f"   Admissions: {len(df)}")
    print(f"   Features: {len(feature_cols)}")

    # Split data (patient-level)
    print("\n2. Splitting data (patient-level)...")
    train, val, test = split_data(df)
    print(f"   Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    X_train = train[feature_cols].values
    X_val = val[feature_cols].values
    X_test = test[feature_cols].values

    results = ExperimentResults(
        experiment_name="XGBoost Baseline",
        model_params=0,  # Will set after training
    )

    # Task 1: 30-day Readmission
    print("\n3. Training: 30-day readmission...")
    y_train_r = train["readmission_30d"].values
    y_val_r = val["readmission_30d"].values
    y_test_r = test["readmission_30d"].values

    if len(np.unique(y_train_r)) > 1:
        model_r = train_binary_classifier(X_train, y_train_r, X_val, y_val_r, "readmission")
        y_pred_r = model_r.predict_proba(X_test)[:, 1]
        metrics_r = compute_binary_metrics(y_test_r, y_pred_r)
        results.readmission_auroc = metrics_r["auroc"]
        results.readmission_auprc = metrics_r["auprc"]
        print(f"   AUROC: {metrics_r['auroc']:.4f} | AUPRC: {metrics_r['auprc']:.4f}")

        # Feature importance
        importance = dict(zip(feature_cols, model_r.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print("   Top features:")
        for feat, imp in top_features:
            print(f"     {feat}: {imp:.4f}")
    else:
        print("   Skipped — single class in training data")

    # Task 2: In-hospital Mortality
    print("\n4. Training: mortality prediction...")
    y_train_m = train["mortality"].values
    y_val_m = val["mortality"].values
    y_test_m = test["mortality"].values

    if len(np.unique(y_train_m)) > 1:
        model_m = train_binary_classifier(X_train, y_train_m, X_val, y_val_m, "mortality")
        y_pred_m = model_m.predict_proba(X_test)[:, 1]
        metrics_m = compute_binary_metrics(y_test_m, y_pred_m)
        results.mortality_auroc = metrics_m["auroc"]
        results.mortality_auprc = metrics_m["auprc"]
        print(f"   AUROC: {metrics_m['auroc']:.4f} | AUPRC: {metrics_m['auprc']:.4f}")
    else:
        print("   Skipped — single class in training data")

    # Timing
    results.training_time_seconds = time.time() - start_time

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\n5. Results saved to {results_path}")

    # Print summary
    print_comparison_table([results])

    return results


if __name__ == "__main__":
    run_experiment()
