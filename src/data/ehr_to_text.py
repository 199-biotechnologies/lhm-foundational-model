"""
EHR-to-Text converter (DT-GPT style).

Converts structured MIMIC-IV patient records into natural language
narratives suitable for LLM fine-tuning.
"""

from typing import Optional

import pandas as pd

# Common MIMIC-IV lab item IDs and their names
LAB_NAMES = {
    50862: "Albumin",
    50868: "Alk Phos",
    50882: "Bicarbonate",
    50885: "Bilirubin",
    50893: "Calcium",
    50902: "Chloride",
    50912: "Creatinine",
    50931: "Glucose",
    50960: "Magnesium",
    50970: "Phosphate",
    50971: "Potassium",
    50983: "Sodium",
    51006: "BUN",
    51222: "Hemoglobin",
    51248: "MCH",
    51249: "MCHC",
    51250: "MCV",
    51265: "Platelet",
    51277: "RDW",
    51279: "RBC",
    51301: "WBC",
    51221: "Hematocrit",
}


def format_labs(labs: dict) -> str:
    """Convert lab dict {itemid: value} to readable string."""
    if not labs or not isinstance(labs, dict):
        return ""
    parts = []
    for item_id, value in labs.items():
        item_id = int(float(item_id)) if isinstance(item_id, str) else int(item_id)
        name = LAB_NAMES.get(item_id, f"Lab{item_id}")
        if pd.notna(value):
            parts.append(f"{name}={value:.1f}")
    return ", ".join(parts[:15])  # Cap at 15 labs to keep token count manageable


def format_diagnoses(diagnoses: list) -> str:
    """Convert ICD code list to string."""
    if not diagnoses or not isinstance(diagnoses, list):
        return "none"
    # Keep first 10 diagnoses
    codes = [str(c) for c in diagnoses[:10]]
    return ", ".join(codes)


def patient_to_text(
    records: pd.DataFrame,
    subject_id: int,
    max_visits: int = 20,
) -> str:
    """
    Convert a patient's full history to a text narrative.

    Format:
    Patient: {age}{gender}.
    Visit {date}: Diagnoses: {ICD codes}. Labs: {lab values}. Meds: {medications}.
    Visit {date}: ...
    """
    patient = records[records["subject_id"] == subject_id].sort_values("admittime")

    if patient.empty:
        return ""

    row0 = patient.iloc[0]
    age = row0.get("anchor_age", "?")
    gender = "M" if row0.get("gender") == "M" else "F"

    lines = [f"Patient: {age}{gender}."]

    for _, row in patient.tail(max_visits).iterrows():
        date = str(row.get("admittime", "unknown"))[:10]
        diag = format_diagnoses(row.get("diagnoses"))
        labs = format_labs(row.get("labs"))

        visit_parts = [f"Visit {date}:"]
        visit_parts.append(f"Diagnoses: {diag}.")
        if labs:
            visit_parts.append(f"Labs: {labs}.")

        lines.append(" ".join(visit_parts))

    return "\n".join(lines)


def build_training_pairs(
    records: pd.DataFrame,
    min_visits: int = 2,
) -> list[dict]:
    """
    Build input/output training pairs by splitting patient timelines at landmark timepoints.

    For each patient with N visits (N >= min_visits):
    - Input: visits 1..k (history)
    - Output: visits k+1..N (future to predict)

    We create multiple splits per patient for more training data.
    """
    pairs = []
    for subject_id in records["subject_id"].unique():
        patient = records[records["subject_id"] == subject_id].sort_values("admittime")
        n_visits = len(patient)

        if n_visits < min_visits:
            continue

        # Create splits at each possible landmark
        for k in range(1, n_visits):
            history_records = patient.iloc[:k]
            future_records = patient.iloc[k:]

            # Build text for history
            history_lines = []
            row0 = patient.iloc[0]
            age = row0.get("anchor_age", "?")
            gender = "M" if row0.get("gender") == "M" else "F"
            history_lines.append(f"Patient: {age}{gender}.")

            for _, row in history_records.iterrows():
                date = str(row.get("admittime", "unknown"))[:10]
                diag = format_diagnoses(row.get("diagnoses"))
                labs = format_labs(row.get("labs"))
                visit_parts = [f"Visit {date}: Diagnoses: {diag}."]
                if labs:
                    visit_parts.append(f"Labs: {labs}.")
                history_lines.append(" ".join(visit_parts))

            history_lines.append("Predict next visits:")
            input_text = "\n".join(history_lines)

            # Build text for future
            future_lines = []
            for _, row in future_records.iterrows():
                date = str(row.get("admittime", "unknown"))[:10]
                diag = format_diagnoses(row.get("diagnoses"))
                labs = format_labs(row.get("labs"))
                visit_parts = [f"Visit {date}: Diagnoses: {diag}."]
                if labs:
                    visit_parts.append(f"Labs: {labs}.")
                future_lines.append(" ".join(visit_parts))

            output_text = "\n".join(future_lines)

            pairs.append({
                "subject_id": subject_id,
                "split_at": k,
                "n_history": k,
                "n_future": n_visits - k,
                "input": input_text,
                "output": output_text,
            })

    return pairs


if __name__ == "__main__":
    from src.data.mimic_loader import build_patient_records, RAW_DIR

    if RAW_DIR.exists():
        records = build_patient_records()
        # Show example for first patient
        first_patient = records["subject_id"].iloc[0]
        print(patient_to_text(records, first_patient))
        print("\n---\n")

        pairs = build_training_pairs(records)
        print(f"Built {len(pairs)} training pairs from {records['subject_id'].nunique()} patients")
        if pairs:
            print(f"\nExample input:\n{pairs[0]['input'][:500]}")
            print(f"\nExample output:\n{pairs[0]['output'][:300]}")
    else:
        print("MIMIC-IV data not found. Run: python -m src.data.download_mimic_demo")
