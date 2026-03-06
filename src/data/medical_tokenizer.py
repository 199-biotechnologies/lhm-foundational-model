"""
Medical tokenizer for EHR event sequences.

Converts structured MIMIC-IV patient records into sequences of medical tokens,
where each token represents a clinical concept (diagnosis, lab result, time gap,
visit boundary, demographic) rather than a text subword.

Used by Experiments 2 (EHRMamba) and 4 (Medical Token model).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# Special tokens
PAD_TOKEN = 0
CLS_TOKEN = 1
SEP_TOKEN = 2
VISIT_START = 3
VISIT_END = 4
PREDICT_TOKEN = 5

# Token type offsets
DEMO_OFFSET = 10       # 10-29: demographics
TIME_OFFSET = 30       # 30-49: time gap bins
DIAG_OFFSET = 50       # 50-5049: diagnosis codes (up to 5000 unique)
LAB_OFFSET = 5050      # 5050-5749: lab tokens (lab_id * 10 + quantile_bin)
MORTALITY_POS = 5750
READMISSION_POS = 5751

VOCAB_SIZE = 6000

# Time gap bins (in days)
TIME_BINS = [0, 1, 3, 7, 14, 30, 90, 180, 365, 1825, float("inf")]
TIME_BIN_LABELS = ["same_day", "1d", "3d", "1w", "2w", "1m", "3m", "6m", "1y", "5y", "5y+"]

# Lab quantile bins
N_LAB_BINS = 10

# Top lab item IDs we track
TOP_LAB_IDS = [
    50971, 50983, 50912, 50902, 51006, 51221, 50882, 50868,
    51265, 51222, 51277, 51279, 51301, 51249, 51250, 51248,
    50931, 50960, 50893, 50970, 50878, 50885, 50861,
]
LAB_ID_TO_IDX = {lab_id: i for i, lab_id in enumerate(TOP_LAB_IDS)}


@dataclass
class TokenizedPatient:
    """A patient's EHR history as a token sequence."""
    subject_id: int
    token_ids: list[int] = field(default_factory=list)
    token_types: list[int] = field(default_factory=list)  # 0=special, 1=demo, 2=time, 3=diag, 4=lab
    timestamps: list[float] = field(default_factory=list)  # hours since first admission
    mortality: int = 0
    readmission_30d: int = 0


def _time_gap_token(days: float) -> int:
    """Convert a time gap in days to a binned token."""
    for i, upper in enumerate(TIME_BINS[1:]):
        if days < upper:
            return TIME_OFFSET + i
    return TIME_OFFSET + len(TIME_BINS) - 2


def _age_token(age: int) -> int:
    """Bin age into decade tokens."""
    decade = min(age // 10, 9)
    return DEMO_OFFSET + decade


def _gender_token(gender: str) -> int:
    """Gender token."""
    return DEMO_OFFSET + 10 + (1 if gender == "M" else 0)


def _lab_token(lab_idx: int, value: float, quantile_boundaries: dict) -> int:
    """Convert a lab value to a quantile-binned token."""
    boundaries = quantile_boundaries.get(lab_idx)
    if boundaries is None or np.isnan(value):
        return None
    bin_idx = int(np.searchsorted(boundaries, value))
    bin_idx = min(bin_idx, N_LAB_BINS - 1)
    return LAB_OFFSET + lab_idx * N_LAB_BINS + bin_idx


def _diagnosis_token(icd_code: str, code_to_idx: dict) -> int:
    """Convert ICD code to token."""
    idx = code_to_idx.get(str(icd_code))
    if idx is None:
        return None
    return DIAG_OFFSET + idx


def compute_lab_quantiles(records: pd.DataFrame) -> dict:
    """Compute quantile boundaries for each lab from training data."""
    quantiles = {}
    for subject_id in records["subject_id"].unique():
        patient = records[records["subject_id"] == subject_id]
        for _, row in patient.iterrows():
            labs = row.get("labs", {})
            if not isinstance(labs, dict):
                continue
            for lab_id, value in labs.items():
                lab_id_int = int(float(lab_id)) if isinstance(lab_id, str) else int(lab_id)
                if lab_id_int in LAB_ID_TO_IDX and pd.notna(value):
                    idx = LAB_ID_TO_IDX[lab_id_int]
                    if idx not in quantiles:
                        quantiles[idx] = []
                    quantiles[idx].append(float(value))

    boundaries = {}
    for idx, values in quantiles.items():
        if len(values) >= N_LAB_BINS:
            percentiles = np.linspace(0, 100, N_LAB_BINS + 1)[1:-1]
            boundaries[idx] = np.percentile(values, percentiles)
        else:
            boundaries[idx] = np.array(sorted(set(values)))
    return boundaries


def build_diagnosis_vocab(records: pd.DataFrame, max_codes: int = 4999) -> dict:
    """Build ICD code -> index mapping from training data."""
    code_counts = {}
    for _, row in records.iterrows():
        diags = row.get("diagnoses", [])
        if isinstance(diags, list):
            for code in diags:
                code_str = str(code)
                code_counts[code_str] = code_counts.get(code_str, 0) + 1

    sorted_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:max_codes]
    return {code: idx for idx, (code, _) in enumerate(sorted_codes)}


def tokenize_patient(
    records: pd.DataFrame,
    subject_id: int,
    code_to_idx: dict,
    lab_quantiles: dict,
    max_tokens: int = 2048,
) -> TokenizedPatient:
    """Convert a patient's full history into a medical token sequence."""
    patient = records[records["subject_id"] == subject_id].sort_values("admittime")
    if patient.empty:
        return TokenizedPatient(subject_id=subject_id)

    result = TokenizedPatient(subject_id=subject_id)
    row0 = patient.iloc[0]

    # CLS + demographics
    result.token_ids.append(CLS_TOKEN)
    result.token_types.append(0)
    result.timestamps.append(0.0)

    age = row0.get("anchor_age", 50)
    result.token_ids.append(_age_token(int(age)))
    result.token_types.append(1)
    result.timestamps.append(0.0)

    gender = row0.get("gender", "F")
    result.token_ids.append(_gender_token(gender))
    result.token_types.append(1)
    result.timestamps.append(0.0)

    first_admit = pd.to_datetime(row0["admittime"])
    prev_discharge = None

    for idx, (_, row) in enumerate(patient.iterrows()):
        if len(result.token_ids) >= max_tokens - 1:
            break

        admit_time = pd.to_datetime(row["admittime"])
        hours_since_first = (admit_time - first_admit).total_seconds() / 3600

        # Time gap token (from previous discharge)
        if prev_discharge is not None:
            gap_days = (admit_time - prev_discharge).total_seconds() / 86400
            result.token_ids.append(_time_gap_token(max(0, gap_days)))
            result.token_types.append(2)
            result.timestamps.append(hours_since_first)

        # Visit start
        result.token_ids.append(VISIT_START)
        result.token_types.append(0)
        result.timestamps.append(hours_since_first)

        # Diagnoses
        diags = row.get("diagnoses", [])
        if isinstance(diags, list):
            for code in diags[:15]:  # Cap at 15 diagnoses per visit
                tok = _diagnosis_token(code, code_to_idx)
                if tok is not None and len(result.token_ids) < max_tokens - 1:
                    result.token_ids.append(tok)
                    result.token_types.append(3)
                    result.timestamps.append(hours_since_first)

        # Labs
        labs = row.get("labs", {})
        if isinstance(labs, dict):
            for lab_id, value in labs.items():
                lab_id_int = int(float(lab_id)) if isinstance(lab_id, str) else int(lab_id)
                if lab_id_int in LAB_ID_TO_IDX and pd.notna(value):
                    lab_idx = LAB_ID_TO_IDX[lab_id_int]
                    tok = _lab_token(lab_idx, float(value), lab_quantiles)
                    if tok is not None and len(result.token_ids) < max_tokens - 1:
                        result.token_ids.append(tok)
                        result.token_types.append(4)
                        result.timestamps.append(hours_since_first)

        # Visit end
        result.token_ids.append(VISIT_END)
        result.token_types.append(0)
        result.timestamps.append(hours_since_first)

        try:
            prev_discharge = pd.to_datetime(row["dischtime"])
        except Exception:
            prev_discharge = admit_time

    # Targets
    last_row = patient.iloc[-1]
    result.mortality = int(last_row.get("hospital_expire_flag", 0))

    if len(patient) >= 2:
        second_last = patient.iloc[-2]
        try:
            last_admit = pd.to_datetime(last_row["admittime"])
            prev_disch = pd.to_datetime(second_last["dischtime"])
            days = (last_admit - prev_disch).total_seconds() / 86400
            result.readmission_30d = 1 if days <= 30 else 0
        except Exception:
            result.readmission_30d = 0

    return result


def tokenize_all_patients(
    records: pd.DataFrame,
    max_tokens: int = 2048,
) -> tuple[list[TokenizedPatient], dict, dict]:
    """Tokenize all patients. Returns tokens, code_to_idx, lab_quantiles."""
    code_to_idx = build_diagnosis_vocab(records)
    lab_quantiles = compute_lab_quantiles(records)

    tokenized = []
    for subject_id in records["subject_id"].unique():
        tp = tokenize_patient(records, subject_id, code_to_idx, lab_quantiles, max_tokens)
        if len(tp.token_ids) > 5:
            tokenized.append(tp)

    return tokenized, code_to_idx, lab_quantiles


if __name__ == "__main__":
    from src.data.mimic_loader import build_patient_records

    records = build_patient_records()
    tokenized, code_to_idx, lab_quantiles = tokenize_all_patients(records)

    print(f"Tokenized {len(tokenized)} patients")
    print(f"Diagnosis vocab size: {len(code_to_idx)}")
    print(f"Lab quantiles computed for {len(lab_quantiles)} labs")

    if tokenized:
        tp = tokenized[0]
        print(f"\nExample patient {tp.subject_id}:")
        print(f"  Tokens: {len(tp.token_ids)}")
        print(f"  First 30 tokens: {tp.token_ids[:30]}")
        print(f"  Mortality: {tp.mortality}")
        print(f"  Readmission 30d: {tp.readmission_30d}")
