"""
Combined data loader for LHM Phase 2a.

Merges MIMIC-IV demo + Synthea into a single longitudinal dataset.
All downstream experiments can use this for larger-scale training.
"""

from pathlib import Path

import pandas as pd

from src.data.mimic_loader import build_patient_records as build_mimic_records
from src.data.synthea_loader import build_patient_records as build_synthea_records


def build_combined_records() -> pd.DataFrame:
    """
    Build combined patient records from all available data sources.

    Returns DataFrame with unified schema:
    - subject_id, gender, anchor_age, admittime, dischtime, diagnoses, labs, source
    """
    frames = []

    # MIMIC-IV demo
    try:
        mimic = build_mimic_records()
        mimic["source"] = "mimic"
        frames.append(mimic)
        print(f"  MIMIC-IV: {mimic['subject_id'].nunique()} patients, {len(mimic)} admissions")
    except Exception as e:
        print(f"  MIMIC-IV: skipped ({e})")

    # Synthea
    try:
        synthea = build_synthea_records()
        synthea["source"] = "synthea"
        frames.append(synthea)
        print(f"  Synthea:  {synthea['subject_id'].nunique()} patients, {len(synthea)} encounters")
    except Exception as e:
        print(f"  Synthea:  skipped ({e})")

    if not frames:
        raise RuntimeError("No data sources available")

    # Unify columns
    common_cols = ["subject_id", "hadm_id", "admittime", "dischtime",
                   "gender", "anchor_age", "diagnoses", "labs", "source"]

    for df in frames:
        for col in common_cols:
            if col not in df.columns:
                df[col] = None

    combined = pd.concat([df[common_cols] for df in frames], ignore_index=True)
    combined = combined.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    return combined


if __name__ == "__main__":
    print("Building combined dataset...")
    records = build_combined_records()
    print(f"\nCombined: {records['subject_id'].nunique()} patients, {len(records)} records")
    print(f"Sources: {records['source'].value_counts().to_dict()}")
    print(f"Columns: {list(records.columns)}")

    # Stats per source
    for source in records["source"].unique():
        sub = records[records["source"] == source]
        visits_per_patient = sub.groupby("subject_id").size()
        print(f"\n{source}:")
        print(f"  Patients: {sub['subject_id'].nunique()}")
        print(f"  Records: {len(sub)}")
        print(f"  Visits/patient: mean={visits_per_patient.mean():.1f}, "
              f"median={visits_per_patient.median():.0f}, max={visits_per_patient.max()}")
