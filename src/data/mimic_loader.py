"""
MIMIC-IV data loader and patient record builder.

Loads MIMIC-IV (demo or full) and constructs longitudinal patient records
from admissions, diagnoses, lab events, prescriptions, and vitals.
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd


DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "mimic-iv-demo"
PROCESSED_DIR = DATA_DIR / "processed"


def load_table(name: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load a MIMIC-IV CSV table."""
    data_dir = data_dir or RAW_DIR
    # MIMIC-IV organizes tables under hosp/ and icu/ subdirectories
    for subdir in ["hosp", "icu", ""]:
        path = data_dir / subdir / f"{name}.csv"
        if path.exists():
            return pd.read_csv(path)
        # Try .csv.gz
        gz_path = data_dir / subdir / f"{name}.csv.gz"
        if gz_path.exists():
            return pd.read_csv(gz_path, compression="gzip")
    raise FileNotFoundError(f"Table {name} not found in {data_dir}")


def build_patient_records(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Build longitudinal patient records from MIMIC-IV tables.

    Returns a DataFrame with one row per patient containing:
    - subject_id, gender, anchor_age
    - admissions: list of admission dicts with diagnoses, labs, meds
    """
    data_dir = data_dir or RAW_DIR

    # Load core tables
    patients = load_table("patients", data_dir)
    admissions = load_table("admissions", data_dir)
    diagnoses = load_table("diagnoses_icd", data_dir)
    labevents = load_table("labevents", data_dir)

    # Try to load prescriptions (may not be in demo)
    try:
        prescriptions = load_table("prescriptions", data_dir)
    except FileNotFoundError:
        prescriptions = pd.DataFrame(columns=["subject_id", "hadm_id", "drug", "starttime"])

    # Merge admissions with patient demographics
    records = admissions.merge(
        patients[["subject_id", "gender", "anchor_age"]],
        on="subject_id",
        how="left",
    )

    # Attach diagnoses per admission
    diag_grouped = (
        diagnoses.groupby("hadm_id")["icd_code"]
        .apply(list)
        .reset_index()
        .rename(columns={"icd_code": "diagnoses"})
    )
    records = records.merge(diag_grouped, on="hadm_id", how="left")

    # Attach top labs per admission (most recent per lab item)
    if not labevents.empty and "hadm_id" in labevents.columns:
        lab_summary = (
            labevents.groupby(["hadm_id", "itemid"])["valuenum"]
            .last()
            .reset_index()
            .groupby("hadm_id")
            .apply(lambda x: dict(zip(x["itemid"], x["valuenum"])), include_groups=False)
            .reset_index()
            .rename(columns={0: "labs"})
        )
        records = records.merge(lab_summary, on="hadm_id", how="left")

    # Sort by patient and admission time
    records = records.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    return records


def get_patient_timeline(records: pd.DataFrame, subject_id: int) -> list[dict]:
    """Get chronological list of admissions for a single patient."""
    patient = records[records["subject_id"] == subject_id].sort_values("admittime")
    timeline = []
    for _, row in patient.iterrows():
        visit = {
            "admittime": row.get("admittime"),
            "dischtime": row.get("dischtime"),
            "diagnoses": row.get("diagnoses", []),
            "labs": row.get("labs", {}),
        }
        timeline.append(visit)
    return timeline


if __name__ == "__main__":
    print(f"Looking for MIMIC-IV data in: {RAW_DIR}")
    if RAW_DIR.exists():
        records = build_patient_records()
        print(f"Built records for {records['subject_id'].nunique()} patients")
        print(f"Total admissions: {len(records)}")
        print(f"Columns: {list(records.columns)}")
    else:
        print("MIMIC-IV data not found. Run: python -m src.data.download_mimic_demo")
