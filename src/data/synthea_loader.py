"""
Synthea synthetic patient data loader.

Loads Synthea-generated CSV files and constructs longitudinal patient records
in the same format as mimic_loader.build_patient_records().
"""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "synthea" / "output" / "csv"


# Map Synthea LOINC observation codes to MIMIC-style lab item IDs
LOINC_TO_ITEMID = {
    "2160-0": 50912,   # Creatinine
    "2345-7": 50931,   # Glucose
    "6299-2": 50820,   # BUN
    "2951-2": 50983,   # Sodium
    "2823-3": 50971,   # Potassium
    "2075-0": 50861,   # Chloride
    "17861-6": 50863,  # Calcium
    "2085-9": 50904,   # HDL Cholesterol
    "2089-1": 50905,   # LDL Cholesterol
    "2571-8": 50907,   # Triglycerides
    "4548-4": 50852,   # HbA1c
    "718-7": 51222,    # Hemoglobin
    "6690-2": 51301,   # WBC
    "789-8": 51265,    # RBC
    "785-6": 51237,    # MCH
    "786-4": 51250,    # MCHC
    "787-2": 51249,    # MCV
    "777-3": 51244,    # Platelets
}

# Map Synthea SNOMED codes to ICD-10 for common conditions
SNOMED_TO_ICD = {
    "44054006": "I10",       # Hypertension
    "73211009": "E11.9",     # Diabetes type 2
    "15777000": "E78.5",     # Hyperlipidemia
    "53741008": "I25.10",    # Coronary artery disease
    "185086009": "Z00.00",   # General exam
    "40055000": "J06.9",     # Upper respiratory infection
    "195662009": "J20.9",    # Acute bronchitis
    "59621000": "I10",       # Essential hypertension
    "431855005": "J06.9",    # Acute viral pharyngitis
    "162864005": "R05.9",    # Cough
    "68496003": "N39.0",     # UTI
    "10509002": "J18.9",     # Pneumonia
    "87433001": "M54.5",     # Low back pain
    "271737000": "R10.9",    # Abdominal pain
    "36971009": "J30.9",     # Allergic rhinitis
}


def build_patient_records(data_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Build longitudinal patient records from Synthea CSV files.

    Returns DataFrame matching mimic_loader format:
    - subject_id, gender, anchor_age, admittime, dischtime, diagnoses, labs
    """
    patients = pd.read_csv(data_dir / "patients.csv")
    encounters = pd.read_csv(data_dir / "encounters.csv")
    conditions = pd.read_csv(data_dir / "conditions.csv")
    observations = pd.read_csv(data_dir / "observations.csv")

    # Map patient IDs to integer subject_ids
    patient_ids = patients["Id"].unique()
    id_map = {pid: i + 100000 for i, pid in enumerate(patient_ids)}

    # Build patient demographics
    patients["subject_id"] = patients["Id"].map(id_map)
    patients["gender"] = patients["GENDER"].map({"M": "M", "F": "F"})
    patients["anchor_age"] = (
        pd.to_datetime("2023-01-01") - pd.to_datetime(patients["BIRTHDATE"])
    ).dt.days // 365

    # Build encounters (admissions)
    encounters["subject_id"] = encounters["PATIENT"].map(id_map)
    encounters["admittime"] = pd.to_datetime(encounters["START"])
    encounters["dischtime"] = pd.to_datetime(encounters["STOP"])
    encounters["hadm_id"] = range(200000, 200000 + len(encounters))

    # Filter to meaningful encounter types (skip wellness visits for training)
    encounter_types = ["inpatient", "outpatient", "emergency", "urgentcare"]
    mask = encounters["ENCOUNTERCLASS"].str.lower().isin(encounter_types)
    encounters = encounters[mask].copy()

    # Map conditions to ICD codes per encounter
    conditions["hadm_id"] = conditions["ENCOUNTER"].map(
        dict(zip(encounters["Id"], encounters["hadm_id"]))
    )
    conditions = conditions.dropna(subset=["hadm_id"])
    conditions["hadm_id"] = conditions["hadm_id"].astype(int)
    conditions["icd_code"] = conditions["CODE"].astype(str).map(SNOMED_TO_ICD)
    conditions.loc[conditions["icd_code"].isna(), "icd_code"] = (
        "Z99.9"  # Unspecified dependence
    )

    diag_grouped = (
        conditions.groupby("hadm_id")["icd_code"]
        .apply(list)
        .reset_index()
        .rename(columns={"icd_code": "diagnoses"})
    )

    # Map observations to lab values per encounter
    observations["hadm_id"] = observations["ENCOUNTER"].map(
        dict(zip(encounters["Id"], encounters["hadm_id"]))
    )
    observations = observations.dropna(subset=["hadm_id"])
    observations["hadm_id"] = observations["hadm_id"].astype(int)
    observations["itemid"] = observations["CODE"].map(LOINC_TO_ITEMID)
    obs_labs = observations.dropna(subset=["itemid", "VALUE"])
    obs_labs = obs_labs.copy()
    obs_labs["valuenum"] = pd.to_numeric(obs_labs["VALUE"], errors="coerce")
    obs_labs = obs_labs.dropna(subset=["valuenum"])

    if not obs_labs.empty:
        lab_summary = (
            obs_labs.groupby(["hadm_id", "itemid"])["valuenum"]
            .last()
            .reset_index()
            .groupby("hadm_id")
            .apply(
                lambda x: dict(zip(x["itemid"].astype(int), x["valuenum"])),
                include_groups=False,
            )
            .reset_index()
            .rename(columns={0: "labs"})
        )
    else:
        lab_summary = pd.DataFrame(columns=["hadm_id", "labs"])

    # Assemble records
    records = encounters[["subject_id", "hadm_id", "admittime", "dischtime"]].copy()
    records = records.merge(
        patients[["subject_id", "gender", "anchor_age"]].drop_duplicates(),
        on="subject_id",
        how="left",
    )
    records = records.merge(diag_grouped, on="hadm_id", how="left")
    records = records.merge(lab_summary, on="hadm_id", how="left")

    # Sort by patient and time
    records = records.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    return records


if __name__ == "__main__":
    print(f"Looking for Synthea data in: {RAW_DIR}")
    if RAW_DIR.exists():
        records = build_patient_records()
        print(f"Built records for {records['subject_id'].nunique()} patients")
        print(f"Total encounters: {len(records)}")
        print(f"Columns: {list(records.columns)}")
        print(f"\nSample patient timeline:")
        sid = records["subject_id"].iloc[0]
        patient = records[records["subject_id"] == sid]
        for _, row in patient.head(3).iterrows():
            print(f"  {row['admittime']} | diags: {row.get('diagnoses', [])}")
    else:
        print("Synthea data not found. Run Synthea first.")
