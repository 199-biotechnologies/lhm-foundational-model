"""
Phase 2a: XGBoost on combined MIMIC-IV + Synthea dataset.

Tests whether 12x more data improves classification performance.
"""

import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.combined_loader import build_combined_records
from src.data.feature_builder import build_tabular_features, get_feature_columns, split_data


OUTPUT_DIR = Path(__file__).parent / "outputs"


def main():
    print("=" * 70)
    print(" Phase 2a: XGBoost on Combined Dataset (MIMIC + Synthea)")
    print("=" * 70)

    # Load combined data
    print("\n1. Loading combined data...")
    records = build_combined_records()

    # Build features
    print("\n2. Building features...")
    df = build_tabular_features(records)
    print(f"   Feature matrix: {df.shape}")
    print(f"   Readmission rate: {df['readmission_30d'].mean():.3f}")
    print(f"   Mortality rate: {df['mortality'].mean():.3f}")

    # Split
    train, val, test = split_data(df)
    feat_cols = get_feature_columns(df)
    print(f"   Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"   Features: {len(feat_cols)}")

    results = {}
    start = time.time()

    # 30-day readmission
    print("\n3. Training readmission model...")
    X_train = train[feat_cols].values
    y_train = train["readmission_30d"].values

    X_val = val[feat_cols].values
    y_val = val["readmission_30d"].values

    X_test = test[feat_cols].values
    y_test = test["readmission_30d"].values

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
        early_stopping_rounds=50, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict_proba(X_test)[:, 1]

    if len(np.unique(y_test)) > 1:
        readm_auroc = roc_auc_score(y_test, y_pred)
        readm_auprc = average_precision_score(y_test, y_pred)
    else:
        readm_auroc = 0.5
        readm_auprc = 0.0
    print(f"   Readmission AUROC: {readm_auroc:.4f}")
    print(f"   Readmission AUPRC: {readm_auprc:.4f}")

    # Mortality
    print("\n4. Training mortality model...")
    y_train_m = train["mortality"].values
    y_val_m = val["mortality"].values
    y_test_m = test["mortality"].values

    if y_train_m.sum() > 0:
        model_m = xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            scale_pos_weight=max(1, (1 - y_train_m.mean()) / max(y_train_m.mean(), 1e-6)),
            reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
            early_stopping_rounds=50, random_state=42,
        )
        model_m.fit(X_train, y_train_m, eval_set=[(X_val, y_val_m)], verbose=False)
        y_pred_m = model_m.predict_proba(X_test)[:, 1]

        if len(np.unique(y_test_m)) > 1:
            mort_auroc = roc_auc_score(y_test_m, y_pred_m)
            mort_auprc = average_precision_score(y_test_m, y_pred_m)
        else:
            mort_auroc = 0.5
            mort_auprc = 0.0
    else:
        mort_auroc = 0.5
        mort_auprc = 0.0
        print("   No mortality events in training set")

    print(f"   Mortality AUROC: {mort_auroc:.4f}")
    print(f"   Mortality AUPRC: {mort_auprc:.4f}")

    elapsed = time.time() - start

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "Phase 2a: XGBoost (MIMIC + Synthea)",
        "n_patients": int(records["subject_id"].nunique()),
        "n_records": len(df),
        "readmission_auroc": f"{readm_auroc:.4f}",
        "readmission_auprc": f"{readm_auprc:.4f}",
        "mortality_auroc": f"{mort_auroc:.4f}",
        "mortality_auprc": f"{mort_auprc:.4f}",
        "training_time_s": f"{elapsed:.1f}",
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f" Results saved to {OUTPUT_DIR / 'results.json'}")
    print(f" Training time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
