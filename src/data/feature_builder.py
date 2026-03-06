"""
Tabular feature builder for XGBoost baseline (Experiment 0).

Converts MIMIC-IV patient records into flat feature vectors for
traditional ML models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.mimic_loader import build_patient_records, load_table


# Top labs by frequency — used as feature columns
TOP_LAB_IDS = [
    50971, 50983, 50912, 50902, 51006, 51221, 50882, 50868,
    51265, 51222, 51277, 51279, 51301, 51249, 51250, 51248,
    50931, 50960, 50893, 50970, 50878, 50885, 50861,
]

LAB_ID_TO_NAME = {
    50861: "alt", 50868: "anion_gap", 50878: "ast", 50882: "bicarbonate",
    50885: "bilirubin", 50893: "calcium", 50902: "chloride", 50912: "creatinine",
    50931: "glucose", 50960: "magnesium", 50970: "phosphate", 50971: "potassium",
    50983: "sodium", 51006: "bun", 51221: "hematocrit", 51222: "hemoglobin",
    51248: "mch", 51249: "mchc", 51250: "mcv", 51265: "platelet",
    51277: "rdw", 51279: "rbc", 51301: "wbc",
}


def build_tabular_features(records: pd.DataFrame) -> pd.DataFrame:
    """
    Build flat feature matrix from patient admission records.

    For each admission, extracts:
    - Demographics: age, gender (binary)
    - Lab values: most recent values for top labs
    - Admission info: admission type, length of stay
    - History: number of prior admissions, days since last admission
    - Diagnosis counts: number of unique ICD codes
    """
    features = []

    for subject_id in records["subject_id"].unique():
        patient = records[records["subject_id"] == subject_id].sort_values("admittime")

        for idx, (_, row) in enumerate(patient.iterrows()):
            feat = {
                "subject_id": subject_id,
                "hadm_id": row["hadm_id"],
            }

            # Demographics
            feat["age"] = row.get("anchor_age", 0)
            feat["gender_m"] = 1 if row.get("gender") == "M" else 0

            # Lab values
            labs = row.get("labs", {})
            if isinstance(labs, dict):
                for lab_id in TOP_LAB_IDS:
                    col_name = f"lab_{LAB_ID_TO_NAME.get(lab_id, str(lab_id))}"
                    feat[col_name] = labs.get(lab_id, labs.get(str(lab_id), np.nan))
            else:
                for lab_id in TOP_LAB_IDS:
                    col_name = f"lab_{LAB_ID_TO_NAME.get(lab_id, str(lab_id))}"
                    feat[col_name] = np.nan

            # Admission info
            try:
                admit = pd.to_datetime(row["admittime"])
                disch = pd.to_datetime(row["dischtime"])
                feat["los_days"] = (disch - admit).total_seconds() / 86400
            except Exception:
                feat["los_days"] = np.nan

            # History features
            feat["n_prior_admissions"] = idx
            if idx > 0:
                prev_row = patient.iloc[idx - 1]
                try:
                    prev_disch = pd.to_datetime(prev_row["dischtime"])
                    curr_admit = pd.to_datetime(row["admittime"])
                    feat["days_since_last"] = (curr_admit - prev_disch).total_seconds() / 86400
                except Exception:
                    feat["days_since_last"] = np.nan
            else:
                feat["days_since_last"] = np.nan

            # Diagnosis count
            diags = row.get("diagnoses", [])
            feat["n_diagnoses"] = len(diags) if isinstance(diags, list) else 0

            # Target: hospital mortality
            feat["mortality"] = int(row.get("hospital_expire_flag", 0))

            # Target: 30-day readmission
            if idx < len(patient) - 1:
                next_row = patient.iloc[idx + 1]
                try:
                    next_admit = pd.to_datetime(next_row["admittime"])
                    curr_disch = pd.to_datetime(row["dischtime"])
                    days_to_readmit = (next_admit - curr_disch).total_seconds() / 86400
                    feat["readmission_30d"] = 1 if days_to_readmit <= 30 else 0
                except Exception:
                    feat["readmission_30d"] = 0
            else:
                feat["readmission_30d"] = 0

            features.append(feat)

    df = pd.DataFrame(features)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature columns (exclude IDs and targets)."""
    exclude = {"subject_id", "hadm_id", "mortality", "readmission_30d"}
    return [c for c in df.columns if c not in exclude]


def split_data(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """Patient-level train/val/test split (no data leakage)."""
    subjects = df["subject_id"].unique()
    train_subj, test_subj = train_test_split(subjects, test_size=test_size, random_state=random_state)
    train_subj, val_subj = train_test_split(train_subj, test_size=val_size / (1 - test_size), random_state=random_state)

    train = df[df["subject_id"].isin(train_subj)]
    val = df[df["subject_id"].isin(val_subj)]
    test = df[df["subject_id"].isin(test_subj)]

    return train, val, test


if __name__ == "__main__":
    records = build_patient_records()
    df = build_tabular_features(records)
    print(f"Feature matrix: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nTarget distribution:")
    print(f"  Mortality: {df['mortality'].mean():.3f}")
    print(f"  30-day readmission: {df['readmission_30d'].mean():.3f}")
    print(f"\nMissing values (top 10):")
    missing = df.isnull().mean().sort_values(ascending=False).head(10)
    for col, pct in missing.items():
        print(f"  {col}: {pct:.1%}")
