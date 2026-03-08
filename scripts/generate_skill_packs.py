#!/usr/bin/env python3
"""
PRISM Skill Packs — 6 Modular Training Datasets

Generates small, targeted training datasets that each teach a distinct PRISM
capability. Each pack produces SFT-format JSONL with <think> tags.

Packs:
  P1  prism-biomarker    (300)  Synthetic lab panel interpretation
  P2  prism-mechanism    (300)  Molecular cascade reasoning (filtered from 32K)
  P3  prism-metabolic    (200)  Multi-marker constellation recognition (filtered)
  P4  prism-repurposing  (200)  Geroprotective drug reasoning (synthetic)
  P5  prism-trajectory   (200)  Rate-of-change interpretation (synthetic)
  P6  prism-routing      (200)  Clinical route classification + firewall (filtered)

Usage:
    python3 scripts/generate_skill_packs.py --pack P1 --count 5 --dry-run
    python3 scripts/generate_skill_packs.py --pack P1 --count 300 --shard 0/5
    python3 scripts/generate_skill_packs.py --merge P1
    python3 scripts/generate_skill_packs.py --combine-all
    python3 scripts/generate_skill_packs.py --stats P1
"""

import json
import subprocess
import sys
import re
import os
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path("docs/datasets/prism-packs")
MEDREASON_FILE = Path("docs/datasets/medreason_32k.jsonl")
PRISM_PROMPT_FILE = Path("docs/datasets/prism-framework/PRISM-v3-compact-system-prompt.md")
OPTIMAL_RANGES_FILE = Path("docs/datasets/prism-framework/references/optimal-ranges.md")
DRUG_REPURPOSING_FILE = Path("docs/datasets/prism-framework/references/drug-repurposing.md")

PACK_CONFIG = {
    "P1": {"name": "prism-biomarker",   "dir": "P1_biomarker",   "default_count": 300},
    "P2": {"name": "prism-mechanism",   "dir": "P2_mechanism",   "default_count": 300},
    "P3": {"name": "prism-metabolic",   "dir": "P3_metabolic",   "default_count": 200},
    "P4": {"name": "prism-repurposing", "dir": "P4_repurposing", "default_count": 200},
    "P5": {"name": "prism-trajectory",  "dir": "P5_trajectory",  "default_count": 200},
    "P6": {"name": "prism-routing",     "dir": "P6_routing",     "default_count": 200},
}

# ──────────────────────────────────────────────────────────────────────
# SHARED INFRASTRUCTURE
# ──────────────────────────────────────────────────────────────────────

def load_prism_prompt():
    """Load PRISM v3 system prompt."""
    return PRISM_PROMPT_FILE.read_text()

PRISM_PROMPT = None  # lazy-loaded


def call_codex(prompt):
    """Call Codex GPT-5.4 via CLI. Fresh context per call."""
    ts = int(time.time())
    pid = os.getpid()
    output_file = Path(f"/tmp/skillpack-codex-{ts}-{pid}.md")
    prompt_file = Path(f"/tmp/skillpack-prompt-{ts}-{pid}.md")
    prompt_file.write_text(prompt)

    try:
        for model in ["gpt-5.4", "gpt-5.3-codex"]:
            try:
                result = subprocess.run(
                    [
                        "codex", "exec", "-m", model,
                        "--skip-git-repo-check",
                        "--sandbox", "read-only",
                        "--ephemeral",
                        "-c", "model_reasoning_effort=xhigh",
                        "-o", str(output_file),
                        "-",
                    ],
                    stdin=open(prompt_file),
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if output_file.exists():
                    content = output_file.read_text().strip()
                    if content and "usage limit" not in content.lower():
                        return content, model
                    # Clean up output file for retry with next model
                    output_file.unlink(missing_ok=True)
            except subprocess.TimeoutExpired:
                print(f"    codex ({model}) timed out")
            except Exception as e:
                print(f"    codex ({model}) error: {e}")
    finally:
        # Cleanup temp files only after all models tried
        for f in [output_file, prompt_file]:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

    return None, None


def call_gemini_api(prompt):
    """Call Gemini API directly via API key (bypasses CLI rate limits)."""
    import urllib.request
    import urllib.error

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None, None

    # Try models in preference order
    for model in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.7},
        })

        req = urllib.request.Request(url, data=payload.encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                if text and len(text) > 100:
                    return text.strip(), f"gemini-api-{model}"
        except urllib.error.HTTPError as e:
            print(f"    gemini-api ({model}) HTTP {e.code}")
        except Exception as e:
            print(f"    gemini-api ({model}) error: {e}")

    return None, None


def call_model(prompt):
    """Try Codex first, fall back to Gemini API."""
    content, model = call_codex(prompt)
    if content:
        return content, model
    return call_gemini_api(prompt)


def parse_generation(raw):
    """Extract <reasoning>, <answer>, <question> tags from model output."""
    result = {}
    for tag in ["reasoning", "answer", "question", "options"]:
        m = re.search(rf'<{tag}>(.*?)</{tag}>', raw, re.DOTALL)
        if m:
            result[tag] = m.group(1).strip()
    return result


def make_sft_entry(question, answer_text, reasoning, metadata):
    """Create standard SFT training entry."""
    return {
        "messages": [
            {"role": "user", "content": f"{question}\n\nAnswer:"},
            {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n{answer_text}"},
        ],
        "metadata": metadata,
    }


def make_example_hash(text):
    """Content-based dedup hash."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_existing_hashes(output_file):
    """Load hashes from existing output for dedup."""
    hashes = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    h = entry.get("metadata", {}).get("content_hash", "")
                    if h:
                        hashes.add(h)
                except (json.JSONDecodeError, KeyError):
                    continue
    return hashes


def save_example(entry, output_file):
    """Append single example to JSONL output."""
    with open(output_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_entry(entry, log_file):
    """Append to generation log."""
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_pack_paths(pack_id, shard_id=None):
    """Get output paths for a pack, optionally shard-specific."""
    cfg = PACK_CONFIG[pack_id]
    pack_dir = BASE_DIR / cfg["dir"]
    pack_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_shard{shard_id}" if shard_id is not None else ""
    return {
        "dir": pack_dir,
        "train": pack_dir / f"train{suffix}.jsonl",
        "log": pack_dir / f"generation_log{suffix}.jsonl",
        "metadata": pack_dir / "metadata.json",
    }


# ──────────────────────────────────────────────────────────────────────
# MEDREASON 32K LOADING + FILTERING
# ──────────────────────────────────────────────────────────────────────

def load_medreason():
    """Load full MedReason 32K dataset."""
    examples = []
    with open(MEDREASON_FILE) as f:
        for line in f:
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def filter_by_keywords(examples, include_kw, exclude_kw=None):
    """Filter examples by keyword presence in question + options."""
    results = []
    for ex in examples:
        text = f"{ex.get('question', '')} {ex.get('options', '')}".lower()
        if any(kw in text for kw in include_kw):
            if exclude_kw and any(kw in text for kw in exclude_kw):
                continue
            results.append(ex)
    return results


# ──────────────────────────────────────────────────────────────────────
# ROUTE CLASSIFIER (reused from prepare_batches.py)
# ──────────────────────────────────────────────────────────────────────

def classify_route(q):
    """Classify clinical route from question text."""
    ql = q.lower()
    acute_kw = ['emergency', 'trauma', 'sepsis', 'shock', 'cardiac arrest', 'cpr', 'intubat',
                'hemorrhag', 'anaphyla', 'dka', 'status epilepticus', 'stroke', 'stemi',
                'unstable', 'resuscitat', 'icu', 'critical', 'acute abdomen', 'gi bleed']
    preg_kw = ['pregnan', 'trimester', 'obstetric', 'postpartum', 'eclampsia', 'ectopic',
               'gestational', 'labor', 'delivery', 'fetal', 'neonatal', 'newborn']
    peds_kw = ['child', 'infant', 'pediatric', 'neonate', 'boy', 'girl', 'year-old boy',
               'year-old girl', 'month-old', 'toddler', 'adolescent']
    psych_kw = ['psychiatr', 'suicid', 'psychosis', 'schizophren', 'bipolar', 'depression',
                'anxiety disorder', 'hallucination', 'delusion', 'substance abuse', 'overdose']
    geri_kw = ['elderly', 'geriatric', 'nursing home', '85-year', '90-year', 'dementia',
               'polypharmacy', 'fall risk', 'frail']
    postop_kw = ['postoperative', 'post-operative', 'pod ', 'post-op', 'day after surgery']

    age_match = re.search(r'(\d+)[\s-]*(year|month|week|day)[\s-]*old', ql)
    if age_match:
        age_val = int(age_match.group(1))
        unit = age_match.group(2)
        if unit in ('month', 'week', 'day') or (unit == 'year' and age_val < 18):
            return 'PEDIATRIC'

    if any(k in ql for k in acute_kw):
        return 'ACUTE'
    if any(k in ql for k in preg_kw):
        return 'PREGNANCY'
    if any(k in ql for k in postop_kw):
        return 'POST_OP'
    if any(k in ql for k in psych_kw):
        return 'PSYCHIATRY'
    if any(k in ql for k in geri_kw):
        return 'GERIATRIC'

    return 'DEFAULT'


# ──────────────────────────────────────────────────────────────────────
# QUALITY VALIDATORS
# ──────────────────────────────────────────────────────────────────────

def validate_common(parsed):
    """Shared quality checks. Returns list of issues."""
    issues = []
    reasoning = parsed.get("reasoning", "")
    answer = parsed.get("answer", "")

    if not reasoning or len(reasoning) < 100:
        issues.append("reasoning_too_short")
    if not answer:
        issues.append("no_answer")
    if len(reasoning) > 5000:
        issues.append("reasoning_too_long")

    # Template language
    lower = reasoning.lower()
    banned = ["let me think", "let's analyze", "hmm", "oh wait", "interestingly,",
              "putting it all together", "in summary,"]
    for phrase in banned:
        if phrase in lower:
            issues.append(f"template_language:{phrase}")

    return issues


def validate_p1(parsed):
    """P1 biomarker: must reference standard AND optimal thresholds + hallmarks."""
    issues = validate_common(parsed)
    reasoning = parsed.get("reasoning", "").lower()

    has_standard = any(w in reasoning for w in ["standard range", "normal range", "reference range",
                                                 "standard threshold", "conventionally normal",
                                                 "standard primary", "standard risk", "standard criteria",
                                                 "standard limit", "within standard"])
    has_optimal = any(w in reasoning for w in ["optimal", "longevity", "optimal range",
                                                "longevity-optimal", "optimization target"])
    has_hallmark = any(w in reasoning for w in ["hallmark", "aging", "senescence", "inflammaging",
                                                 "nutrient sensing", "autophagy", "proteostasis"])

    if not has_standard:
        issues.append("missing_standard_threshold")
    if not has_optimal:
        issues.append("missing_optimal_threshold")
    if not has_hallmark:
        issues.append("missing_hallmark_reference")

    return issues


def validate_p2(parsed):
    """P2 mechanism: ≥3 molecular entities + directional arrows."""
    issues = validate_common(parsed)
    reasoning = parsed.get("reasoning", "")

    # Count molecular entities (capitalized terms that look like proteins/pathways)
    mol_entities = re.findall(r'\b[A-Z][A-Za-z0-9]{2,}(?:-[A-Za-z0-9]+)*\b', reasoning)
    unique_entities = set(mol_entities)
    if len(unique_entities) < 3:
        issues.append(f"too_few_molecular_entities:{len(unique_entities)}")

    if '→' not in reasoning and '->' not in reasoning:
        issues.append("missing_directional_arrows")

    return issues


def validate_p3(parsed):
    """P3 metabolic: must name at least one PRISM pattern constellation."""
    issues = validate_common(parsed)
    reasoning = parsed.get("reasoning", "").lower()

    constellations = ["insulin resistance", "chronic inflammation", "atherogenic",
                      "thyroid spectrum", "metabolic syndrome", "ir cascade"]
    if not any(c in reasoning for c in constellations):
        issues.append("missing_constellation_reference")

    return issues


def validate_p4(parsed):
    """P4 repurposing: must include evidence tiers + monitoring."""
    issues = validate_common(parsed)
    reasoning = parsed.get("reasoning", "").lower()

    has_tier = any(w in reasoning for w in ["[a]", "[b]", "[c]", "tier a", "tier b", "tier c",
                                             "evidence tier"])
    has_monitoring = any(w in reasoning for w in ["monitor", "monitoring", "check", "assess",
                                                   "follow-up", "surveillance"])
    has_contraindication = any(w in reasoning for w in ["contraindic", "avoid", "do not use",
                                                         "caution", "risk"])

    if not has_tier:
        issues.append("missing_evidence_tier")
    if not has_monitoring:
        issues.append("missing_monitoring")
    if not has_contraindication:
        issues.append("missing_contraindication_discussion")

    return issues


def validate_p5(parsed):
    """P5 trajectory: must contain rate calculations + trajectory vs absolute comparison."""
    issues = validate_common(parsed)
    reasoning = parsed.get("reasoning", "").lower()

    has_rate = any(w in reasoning for w in ["rate of change", "per month", "per year",
                                             "trajectory", "trend", "slope", "velocity",
                                             "increasing", "decreasing", "rising", "declining"])
    has_comparison = any(w in reasoning for w in ["absolute value", "static value", "single point",
                                                   "vs absolute", "compared to a stable",
                                                   "trajectory vs", "trend vs", "static elevation",
                                                   "static glucose", "static measurement",
                                                   "single timepoint", "point-in-time",
                                                   "not worsening", "no directional trend"])

    if not has_rate:
        issues.append("missing_rate_calculation")
    if not has_comparison:
        issues.append("missing_trajectory_vs_absolute")

    return issues


def validate_p6(parsed, route):
    """P6 routing: ACUTE/PEDS/PREGNANCY must NOT contain longevity keywords."""
    issues = validate_common(parsed)
    reasoning = parsed.get("reasoning", "").lower()

    firewall_routes = {"ACUTE", "PEDIATRIC", "PREGNANCY"}
    longevity_kw = ["longevity", "healthspan", "hallmark of aging", "geroprotect",
                    "optimal range", "longevity-optimal", "optimization target",
                    "biological aging"]

    if route in firewall_routes:
        for kw in longevity_kw:
            if kw in reasoning:
                issues.append(f"firewall_violation:{kw}")

    # Check route is correctly identified
    has_route = any(w in reasoning for w in ["route:", "route classification", route.lower()])
    if not has_route:
        issues.append("missing_route_classification")

    return issues


VALIDATORS = {
    "P1": validate_p1,
    "P2": validate_p2,
    "P3": validate_p3,
    "P4": validate_p4,
    "P5": validate_p5,
    # P6 handled separately (needs route arg)
}


# ──────────────────────────────────────────────────────────────────────
# P1: BIOMARKER GENERATOR
# ──────────────────────────────────────────────────────────────────────

# Parsed from optimal-ranges.md — representative subset for generation
BIOMARKER_DB = {
    "HbA1c":          {"unit": "%",      "standard": (None, 5.7),   "optimal": (None, 5.3),   "tier": "B", "category": "metabolic"},
    "Fasting Glucose": {"unit": "mg/dL",  "standard": (70, 99),     "optimal": (72, 85),      "tier": "B", "category": "metabolic"},
    "Fasting Insulin": {"unit": "mIU/L",  "standard": (None, 25),   "optimal": (None, 7),     "tier": "B", "category": "metabolic"},
    "HOMA-IR":         {"unit": "",       "standard": (None, 2.5),  "optimal": (None, 1.0),   "tier": "B", "category": "metabolic"},
    "Uric Acid":       {"unit": "mg/dL",  "standard": (3.5, 7.2),   "optimal": (4.0, 5.5),    "tier": "B", "category": "metabolic"},
    "ApoB":            {"unit": "mg/dL",  "standard": (None, 130),  "optimal": (None, 70),    "tier": "A/B", "category": "lipid"},
    "LDL-C":           {"unit": "mg/dL",  "standard": (None, 100),  "optimal": (None, 70),    "tier": "A", "category": "lipid"},
    "Lp(a)":           {"unit": "nmol/L", "standard": (None, 75),   "optimal": (None, 30),    "tier": "A", "category": "lipid"},
    "TG/HDL ratio":    {"unit": "",       "standard": (None, 3.5),  "optimal": (None, 1.0),   "tier": "B", "category": "lipid"},
    "hsCRP":           {"unit": "mg/L",   "standard": (None, 3.0),  "optimal": (None, 0.5),   "tier": "B", "category": "inflammatory"},
    "Homocysteine":    {"unit": "µmol/L", "standard": (None, 15),   "optimal": (None, 8),     "tier": "B", "category": "inflammatory"},
    "NLR":             {"unit": "",       "standard": (None, 3.0),  "optimal": (None, 2.0),   "tier": "B", "category": "inflammatory"},
    "RDW":             {"unit": "%",      "standard": (11.5, 14.5), "optimal": (None, 13.0),  "tier": "B", "category": "inflammatory"},
    "Fibrinogen":      {"unit": "mg/dL",  "standard": (200, 400),   "optimal": (None, 300),   "tier": "B", "category": "inflammatory"},
    "Ferritin_M":      {"unit": "ng/mL",  "standard": (20, 300),    "optimal": (40, 150),     "tier": "B", "category": "iron"},
    "Ferritin_F":      {"unit": "ng/mL",  "standard": (12, 150),    "optimal": (30, 80),      "tier": "B", "category": "iron"},
    "Transferrin Sat": {"unit": "%",      "standard": (20, 50),     "optimal": (25, 40),      "tier": "B", "category": "iron"},
    "TSH":             {"unit": "mIU/L",  "standard": (0.4, 4.0),   "optimal": (0.5, 2.5),    "tier": "B", "category": "thyroid"},
    "Free T4":         {"unit": "ng/dL",  "standard": (0.8, 1.8),   "optimal": (1.1, 1.5),    "tier": "B", "category": "thyroid"},
    "Free T3":         {"unit": "pg/mL",  "standard": (2.0, 4.4),   "optimal": (3.0, 3.8),    "tier": "C", "category": "thyroid"},
    "Total T_M":       {"unit": "ng/dL",  "standard": (300, 1000),  "optimal": (500, 900),    "tier": "B", "category": "hormone_m"},
    "Free T_M":        {"unit": "ng/dL",  "standard": (5, 21),      "optimal": (10, 18),      "tier": "B", "category": "hormone_m"},
    "SHBG":            {"unit": "nmol/L", "standard": (16, 55),     "optimal": (20, 40),      "tier": "B", "category": "hormone_m"},
    "eGFR":            {"unit": "mL/min", "standard": (60, None),   "optimal": (90, None),    "tier": "A", "category": "renal"},
    "Cystatin C":      {"unit": "mg/L",   "standard": (0.5, 1.0),   "optimal": (None, 0.8),   "tier": "B", "category": "renal"},
    "ALT":             {"unit": "U/L",    "standard": (None, 40),   "optimal": (None, 20),    "tier": "B", "category": "hepatic"},
    "GGT":             {"unit": "U/L",    "standard": (None, 60),   "optimal": (None, 20),    "tier": "B", "category": "hepatic"},
    "Vitamin D":       {"unit": "ng/mL",  "standard": (30, 100),    "optimal": (40, 60),      "tier": "C", "category": "micronutrient"},
    "Vitamin B12":     {"unit": "pg/mL",  "standard": (200, 900),   "optimal": (400, 800),    "tier": "B", "category": "micronutrient"},
    "Omega-3 Index":   {"unit": "%",      "standard": (4, None),    "optimal": (8, None),     "tier": "A", "category": "micronutrient"},
    "RBC Magnesium":   {"unit": "mg/dL",  "standard": (4.2, 6.8),   "optimal": (6.0, 6.5),    "tier": "B", "category": "micronutrient"},
    "Folate":          {"unit": "ng/mL",  "standard": (3, None),    "optimal": (15, None),    "tier": "B", "category": "micronutrient"},
    "VO2 Max":         {"unit": "mL/kg/min", "standard": (None, None), "optimal": (None, None), "tier": "A", "category": "functional"},
    "Grip Strength_M": {"unit": "kg",     "standard": (None, None), "optimal": (40, None),    "tier": "A", "category": "functional"},
    "IGF-1":           {"unit": "ng/mL",  "standard": (None, None), "optimal": (None, None),  "tier": "B", "category": "growth"},
}

# Categories that make sense together on a lab panel
LAB_PANEL_GROUPS = {
    "comprehensive_metabolic": ["HbA1c", "Fasting Glucose", "Fasting Insulin", "HOMA-IR",
                                 "Uric Acid", "ALT", "GGT", "eGFR"],
    "lipid_cardiovascular": ["ApoB", "LDL-C", "Lp(a)", "TG/HDL ratio", "hsCRP", "Homocysteine"],
    "inflammatory": ["hsCRP", "NLR", "RDW", "Fibrinogen", "Ferritin_M", "Ferritin_F"],
    "thyroid": ["TSH", "Free T4", "Free T3"],
    "male_hormone": ["Total T_M", "Free T_M", "SHBG", "Fasting Insulin"],
    "micronutrient": ["Vitamin D", "Vitamin B12", "Omega-3 Index", "RBC Magnesium", "Folate"],
    "renal_hepatic": ["eGFR", "Cystatin C", "ALT", "GGT", "Uric Acid"],
}


def generate_lab_value(marker_name, marker_info):
    """Generate a realistic lab value — sometimes normal, sometimes concerning."""
    lo, hi = marker_info["standard"]
    opt_lo, opt_hi = marker_info["optimal"]

    # Decide zone: 30% clearly abnormal, 40% suboptimal (between standard and optimal), 30% optimal
    zone = random.choices(["abnormal", "suboptimal", "optimal"], weights=[30, 40, 30])[0]

    if zone == "abnormal":
        # Outside standard range
        if hi is not None:
            return round(hi * random.uniform(1.05, 1.6), 1)
        elif lo is not None:
            return round(lo * random.uniform(0.4, 0.9), 1)
        else:
            return round(random.uniform(20, 80), 1)
    elif zone == "suboptimal":
        # Within standard but outside optimal
        if opt_hi is not None and hi is not None:
            return round(random.uniform(opt_hi, hi), 1)
        elif opt_lo is not None and lo is not None:
            return round(random.uniform(lo, opt_lo), 1)
        elif hi is not None and opt_hi is not None:
            return round(random.uniform(opt_hi * 0.8, hi), 1)
        else:
            return round(random.uniform(30, 60), 1)
    else:
        # Optimal range
        if opt_lo is not None and opt_hi is not None:
            return round(random.uniform(opt_lo, opt_hi), 1)
        elif opt_hi is not None:
            lo_bound = opt_hi * 0.5 if opt_lo is None else opt_lo
            return round(random.uniform(lo_bound, opt_hi), 1)
        elif opt_lo is not None:
            return round(random.uniform(opt_lo, opt_lo * 1.3), 1)
        else:
            return round(random.uniform(30, 60), 1)


def generate_p1_example(idx):
    """Generate a synthetic biomarker panel interpretation example."""
    global PRISM_PROMPT
    if PRISM_PROMPT is None:
        PRISM_PROMPT = load_prism_prompt()

    # Pick a panel type and select 5-10 markers
    panel_type = random.choice(list(LAB_PANEL_GROUPS.keys()))
    base_markers = LAB_PANEL_GROUPS[panel_type]

    # Add 1-3 markers from other panels for cross-system interpretation
    other_markers = [m for m in BIOMARKER_DB.keys()
                     if m not in base_markers and BIOMARKER_DB[m]["category"] != "functional"]
    extra = random.sample(other_markers, min(3, len(other_markers)))
    markers = base_markers[:] + extra
    random.shuffle(markers)
    markers = markers[:random.randint(5, 10)]

    # Filter sex-specific markers
    sex = random.choice(["male", "female"])
    age = random.randint(28, 72)
    if sex == "female":
        markers = [m for m in markers if "_M" not in m]
    else:
        markers = [m for m in markers if "_F" not in m]

    # Generate values
    lab_lines = []
    for m in markers:
        if m not in BIOMARKER_DB:
            continue
        info = BIOMARKER_DB[m]
        val = generate_lab_value(m, info)
        display_name = m.replace("_M", "").replace("_F", "")
        lab_lines.append(f"  {display_name}: {val} {info['unit']}")

    lab_panel = "\n".join(lab_lines)

    # Create options — one correct (most concerning) + 3 distractors
    options = ["A", "B", "C", "D"]
    option_markers = random.sample(markers[:4] if len(markers) >= 4 else markers, min(4, len(markers)))
    while len(option_markers) < 4:
        filler = random.choice([m for m in markers if m not in option_markers])
        option_markers.append(filler)

    options_text = "\n".join(
        f"{letter}. {m.replace('_M', '').replace('_F', '')}"
        for letter, m in zip(options, option_markers)
    )

    question = f"""A {age}-year-old {sex} presents for a longevity-focused health assessment. The following lab results are obtained:

{lab_panel}

Which finding is most concerning from a longevity perspective?

{options_text}"""

    prompt = f"""{PRISM_PROMPT}

---

BIOMARKER INTERPRETATION TASK:

You are generating training data for a medical reasoning model. Given the lab panel below, produce a structured reasoning trace that:

1. Applies DUAL-THRESHOLD interpretation (standard range vs longevity-optimal range) for EACH abnormal or suboptimal value
2. Identifies which Hallmark(s) of Aging are implicated
3. Checks for multi-marker CONSTELLATIONS (Insulin Resistance Cascade, Chronic Inflammation, Atherogenic Risk)
4. Explains WHY the most concerning finding matters from a longevity perspective
5. Provides the correct answer

FORMAT:
<reasoning>
[Your chain-of-thought reasoning. 8-15 sentences. Must reference both standard AND optimal thresholds. Must mention at least one hallmark of aging. Must check for pattern constellations.]
</reasoning>
<answer>[Letter]. [Marker name]</answer>

Question:
{question}

Ground truth: The most concerning finding should be identified based on the gap between the patient's value, the standard range, and the longevity-optimal target — prioritizing findings that indicate active pathological processes or multiple hallmark involvement."""

    return question, options_text, prompt


# ──────────────────────────────────────────────────────────────────────
# P2: MECHANISM FILTER + GENERATOR
# ──────────────────────────────────────────────────────────────────────

P2_INCLUDE_KW = [
    "mechanism of action", "pharmacology", "pathophysiology", "receptor",
    "signaling", "pathway", "enzyme", "kinase", "phosphorylat", "cascade",
    "molecular", "cellular", "transduction", "inhibit", "antagonist",
    "agonist", "channel", "transport", "synthesis", "degradation",
    "metabolism", "cytochrome", "substrate", "mediator",
]

P2_EXCLUDE_KW = [
    "camera trap", "neural network", "deep learning", "machine learning",
    "image classification", "drosophila", "bioinformatics",
]


def build_p2_prompt(example):
    """Build prompt for P2 mechanism trace."""
    global PRISM_PROMPT
    if PRISM_PROMPT is None:
        PRISM_PROMPT = load_prism_prompt()

    question = example["question"]
    options = example.get("options", "")
    answer = example.get("answer", "")

    return f"""{PRISM_PROMPT}

---

L3-DEPTH MOLECULAR CASCADE TASK:

You are generating training data for a medical reasoning model. Given the question below, produce an L3-depth molecular cascade reasoning trace that:

1. Traces the FULL molecular cascade from initial trigger to clinical manifestation
2. Uses directional arrows (→) to show causality: Target → Pathway → Downstream → Clinical Effect
3. Names ≥3 specific molecular entities (proteins, enzymes, receptors, transcription factors)
4. Links the mechanism to relevant Hallmarks of Aging where applicable
5. Eliminates incorrect options with mechanism-based reasoning

FORMAT:
<reasoning>
[Your L3-depth reasoning. 8-15 sentences. Must contain ≥3 molecular entities. Must use → arrows. Must link to hallmarks if clinically relevant.]
</reasoning>
<answer>[Letter]. [Answer text]</answer>

Question: {question}
Options: {options}
Ground Truth: {answer}

Produce your reasoning now. Remember: L3 depth means FULL molecular cascade, not just organ-level description."""


# ──────────────────────────────────────────────────────────────────────
# P3: METABOLIC CONSTELLATION FILTER + GENERATOR
# ──────────────────────────────────────────────────────────────────────

P3_INCLUDE_KW = [
    "diabetes", "insulin", "glucose", "hba1c", "metabolic", "obesity",
    "dyslipidemia", "cholesterol", "triglyceride", "hypertension",
    "cardiovascular", "coronary", "atherosclerosis", "heart failure",
    "endocrine", "thyroid", "adrenal", "pituitary", "cortisol",
    "aldosterone", "cushing", "addison", "pheochromocytoma",
    "renal", "kidney", "nephropathy", "proteinuria",
    "liver", "hepatic", "steatosis", "nafld", "cirrhosis",
    "metabolic syndrome", "bmi", "waist circumference",
]


def build_p3_prompt(example):
    """Build prompt for P3 metabolic constellation recognition."""
    global PRISM_PROMPT
    if PRISM_PROMPT is None:
        PRISM_PROMPT = load_prism_prompt()

    question = example["question"]
    options = example.get("options", "")
    answer = example.get("answer", "")

    return f"""{PRISM_PROMPT}

---

METABOLIC CONSTELLATION RECOGNITION TASK:

You are generating training data for a medical reasoning model. Given the metabolic/endocrine/cardiovascular question below, produce a reasoning trace that:

1. Identifies multi-marker CONSTELLATIONS from the PRISM framework:
   - Insulin Resistance Cascade: TG/HDL>2, insulin>7, HbA1c>5.3%, glucose>85, uric acid>5.5, ALT>25+GGT>25
   - Chronic Inflammation: hsCRP>1, ferritin>150, NLR>2.5, albumin<4.0
   - Atherogenic Risk Enhancement: ApoB>80+hsCRP>1, Lp(a)≥50, ApoB/ApoA1>0.7
   - Thyroid Spectrum: TSH>4.0 or TSH 2.5-4.0+symptoms+TPO Ab positive
2. Traces the metabolic pathway at L2 depth
3. Names the specific constellation pattern if present
4. Connects to longevity implications where relevant

FORMAT:
<reasoning>
[Your reasoning. 8-15 sentences. Must name at least one PRISM pattern constellation by name. Must trace metabolic pathway.]
</reasoning>
<answer>[Letter]. [Answer text]</answer>

Question: {question}
Options: {options}
Ground Truth: {answer}

Produce your constellation-aware reasoning now."""


# ──────────────────────────────────────────────────────────────────────
# P4: DRUG REPURPOSING GENERATOR
# ──────────────────────────────────────────────────────────────────────

P4_DRUGS = [
    "Rapamycin", "Metformin", "Semaglutide", "Empagliflozin", "Telmisartan",
    "Rosuvastatin", "Acarbose", "Low-Dose Naltrexone", "Dasatinib+Quercetin",
    "NMN", "Taurine", "17α-Estradiol", "Spermidine",
]

P4_SCENARIOS = [
    "55M with early insulin resistance (HOMA-IR 2.1), no diabetes",
    "62F with hsCRP 2.8, ApoB 95, moderate ASCVD risk",
    "48M with metabolic syndrome, BMI 34, ALT 45",
    "70F with osteoporosis, sarcopenia, declining grip strength",
    "45M with family history of early MI, Lp(a) 85 nmol/L",
    "58F with Hashimoto's, fatigue, hsCRP 4.2",
    "52M with prediabetes (HbA1c 5.9%), considering prevention",
    "65M post-MI, on statin, seeking longevity optimization",
    "42F with PCOS, insulin resistance, trying to conceive",
    "73M with HFpEF, eGFR 45, uric acid 8.1",
    "50F with breast cancer history, now in remission, seeking longevity",
    "60M with early CKD (eGFR 55), hypertension, gout",
    "38M athlete with VO2max >90th percentile, seeking further optimization",
    "68F with multiple autoimmune conditions, high inflammatory markers",
    "55M with elevated ferritin (450), normal TSAT, chronic fatigue",
]


def generate_p4_example(idx):
    """Generate a drug repurposing reasoning example."""
    global PRISM_PROMPT
    if PRISM_PROMPT is None:
        PRISM_PROMPT = load_prism_prompt()

    drug = P4_DRUGS[idx % len(P4_DRUGS)]
    scenario = P4_SCENARIOS[idx % len(P4_SCENARIOS)]

    # Create distractors from remaining drugs
    other_drugs = [d for d in P4_DRUGS if d != drug]
    distractors = random.sample(other_drugs, 3)
    all_options = [drug] + distractors
    random.shuffle(all_options)
    correct_idx = all_options.index(drug)
    correct_letter = "ABCD"[correct_idx]

    options_text = "\n".join(f"{letter}. {d}" for letter, d in zip("ABCD", all_options))

    question = f"""A {scenario}. From a geroprotective/longevity medicine perspective, which drug would be most appropriate to consider as an adjunct intervention?

{options_text}"""

    drug_ref = DRUG_REPURPOSING_FILE.read_text()

    prompt = f"""{PRISM_PROMPT}

---

DRUG REFERENCE (for this task):
{drug_ref[:4000]}

---

GEROPROTECTIVE DRUG REASONING TASK:

You are generating training data for a medical reasoning model. Given the clinical scenario below, produce a reasoning trace that:

1. Identifies the STANDARD treatment approach for this patient
2. Evaluates the geroprotective ALTERNATIVE: mechanism of action → relevant pathway → expected benefit
3. Cites the EVIDENCE TIER: [A] RCT-backed, [B] observational/consensus, [C] preclinical
4. Lists CONTRAINDICATIONS and monitoring requirements
5. Addresses competing risks if applicable

FORMAT:
<reasoning>
[Your reasoning. 8-15 sentences. Must include evidence tiers [A/B/C]. Must mention monitoring. Must discuss contraindications or cautions.]
</reasoning>
<answer>{correct_letter}. {drug}</answer>

Question: {question}
Ground Truth: {correct_letter}. {drug}

Produce your geroprotective reasoning now. Remember: evidence tier is for the SPECIFIC CLAIM being made, not the drug in general."""

    return question, options_text, prompt, correct_letter, drug


# ──────────────────────────────────────────────────────────────────────
# P5: TRAJECTORY GENERATOR
# ──────────────────────────────────────────────────────────────────────

# Markers suitable for trajectory analysis
TRAJECTORY_MARKERS = [
    ("HbA1c", "%", 4.5, 7.5),
    ("ApoB", "mg/dL", 40, 160),
    ("hsCRP", "mg/L", 0.1, 8.0),
    ("eGFR", "mL/min", 30, 120),
    ("ALT", "U/L", 10, 80),
    ("TSH", "mIU/L", 0.3, 8.0),
    ("Fasting Insulin", "mIU/L", 2, 30),
    ("Fasting Glucose", "mg/dL", 65, 130),
    ("Ferritin", "ng/mL", 15, 500),
    ("Vitamin D", "ng/mL", 10, 80),
]

TRAJECTORY_PATTERNS = ["improving", "deteriorating", "stable_suboptimal", "crossing_threshold"]


def generate_trajectory_data(marker_name, unit, val_min, val_max, pattern,
                              n_points=None, interval_months=None):
    """Generate timepoints with realistic noise."""
    if n_points is None:
        n_points = random.choice([3, 4])
    if interval_months is None:
        interval_months = random.choice([6, 9, 12])
    timepoints = []

    if pattern == "improving":
        start = random.uniform(val_max * 0.6, val_max * 0.85)
        end = random.uniform(val_min * 1.1, val_max * 0.4)
        if marker_name == "eGFR":  # Higher is better
            start, end = end, start
    elif pattern == "deteriorating":
        start = random.uniform(val_min * 1.1, val_max * 0.4)
        end = random.uniform(val_max * 0.6, val_max * 0.95)
        if marker_name == "eGFR":
            start, end = end, start
    elif pattern == "stable_suboptimal":
        center = random.uniform(val_max * 0.45, val_max * 0.65)
        start = center
        end = center * random.uniform(0.95, 1.05)
    else:  # crossing_threshold
        # Cross from normal to abnormal
        start = random.uniform(val_max * 0.3, val_max * 0.45)
        end = random.uniform(val_max * 0.55, val_max * 0.8)
        if marker_name == "eGFR":
            start, end = end, start

    for i in range(n_points):
        t = i * interval_months
        frac = i / (n_points - 1) if n_points > 1 else 0
        base_val = start + (end - start) * frac
        # Add realistic noise (±5%)
        noise = base_val * random.uniform(-0.05, 0.05)
        val = round(base_val + noise, 1)
        val = max(val_min, min(val_max, val))
        timepoints.append((t, val))

    return timepoints, interval_months


def generate_p5_example(idx):
    """Generate a trajectory interpretation example."""
    global PRISM_PROMPT
    if PRISM_PROMPT is None:
        PRISM_PROMPT = load_prism_prompt()

    # Pick 2-3 markers, one with a clear trend
    n_markers = random.choice([2, 3])
    selected = random.sample(TRAJECTORY_MARKERS, n_markers)
    patterns = [random.choice(TRAJECTORY_PATTERNS) for _ in selected]
    # Ensure at least one clear trend (not stable)
    if all(p == "stable_suboptimal" for p in patterns):
        patterns[0] = random.choice(["deteriorating", "crossing_threshold"])

    sex = random.choice(["male", "female"])
    age = random.randint(35, 68)

    # Build trajectory table — use consistent timepoints across markers
    shared_n_points = random.choice([3, 4])
    shared_interval = random.choice([6, 9, 12])
    all_data = []
    for (marker_name, unit, vmin, vmax), pattern in zip(selected, patterns):
        timepoints, interval = generate_trajectory_data(
            marker_name, unit, vmin, vmax, pattern,
            n_points=shared_n_points, interval_months=shared_interval,
        )
        all_data.append((marker_name, unit, timepoints, interval, pattern))

    # Format as table
    max_points = max(len(tp) for _, _, tp, _, _ in all_data)
    headers = ["Marker"] + [f"Month {all_data[0][2][i][0]}" for i in range(max_points)]
    header_line = " | ".join(headers)
    sep_line = " | ".join(["---"] * len(headers))

    rows = []
    for marker_name, unit, timepoints, _, _ in all_data:
        row = [f"{marker_name} ({unit})"]
        for i in range(max_points):
            if i < len(timepoints):
                row.append(str(timepoints[i][1]))
            else:
                row.append("—")
        rows.append(" | ".join(row))

    table = f"| {header_line} |\n| {sep_line} |\n" + "\n".join(f"| {r} |" for r in rows)

    # Create options based on the markers
    option_texts = []
    for marker_name, _, _, _, pattern in all_data:
        if pattern == "deteriorating":
            option_texts.append(f"Rising {marker_name} trend")
        elif pattern == "crossing_threshold":
            option_texts.append(f"{marker_name} crossing into abnormal range")
        elif pattern == "improving":
            option_texts.append(f"Declining {marker_name} trajectory")
        else:
            option_texts.append(f"Persistently suboptimal {marker_name}")

    while len(option_texts) < 4:
        option_texts.append(f"Stable {random.choice(['inflammatory', 'metabolic', 'lipid'])} markers")

    options_text = "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", option_texts[:4]))

    question = f"""A {age}-year-old {sex} has serial labs over {all_data[0][3] * (max_points - 1)} months:

{table}

What is the most concerning trend in this patient's longitudinal data?

{options_text}"""

    prompt = f"""{PRISM_PROMPT}

---

TRAJECTORY INTERPRETATION TASK:

You are generating training data for a medical reasoning model. Given the longitudinal lab data below, produce a reasoning trace that:

1. Calculates the RATE OF CHANGE for each marker (e.g., "ApoB increased from X to Y over Z months = +N/month")
2. Distinguishes between TRAJECTORY concerns (worsening trend) vs ABSOLUTE VALUE concerns (static elevation)
3. Identifies which trend is most clinically concerning and WHY (rate × clinical significance)
4. References optimal vs standard thresholds for context
5. Notes if any marker is CROSSING a clinical threshold during the observation period

FORMAT:
<reasoning>
[Your reasoning. 8-15 sentences. Must contain explicit rate-of-change calculations. Must compare trajectory significance vs absolute value significance.]
</reasoning>
<answer>[Letter]. [Most concerning trend]</answer>

Question:
{question}

Produce your trajectory analysis now. Remember: a RISING value crossing a threshold is more concerning than a STABLE elevated value."""

    return question, options_text, prompt


# ──────────────────────────────────────────────────────────────────────
# P6: ROUTING FILTER + GENERATOR
# ──────────────────────────────────────────────────────────────────────

P6_ROUTE_TARGETS = {
    "ACUTE": 50,
    "PEDIATRIC": 40,
    "PREGNANCY": 30,
    "DEFAULT": 50,
    "GERIATRIC": 20,
    "POST_OP": 10,
}


def build_p6_prompt(example, route):
    """Build prompt for P6 routing with firewall enforcement."""
    global PRISM_PROMPT
    if PRISM_PROMPT is None:
        PRISM_PROMPT = load_prism_prompt()

    question = example["question"]
    options = example.get("options", "")
    answer = example.get("answer", "")

    firewall_note = ""
    if route in ("ACUTE", "PEDIATRIC", "PREGNANCY"):
        firewall_note = f"""
FIREWALL ACTIVE — Route: {route}
You MUST NOT apply longevity-optimal thresholds, hallmark mapping, or optimization overlays.
Use ONLY standard medical ranges and acute/standard clinical reasoning.
Do NOT mention "longevity", "healthspan", "hallmarks of aging", "geroprotection", or "optimal ranges".
"""

    return f"""{PRISM_PROMPT}

---

CLINICAL ROUTING TASK:

{firewall_note}

You are generating training data for a medical reasoning model. Given the question below:

1. Classify the clinical ROUTE: {route}
2. Apply route-specific reasoning rules from the PRISM framework
3. {'Apply the FIREWALL — NO longevity lens for this route' if route in ('ACUTE', 'PEDIATRIC', 'PREGNANCY') else 'Apply longevity lens if clinically relevant (DEFAULT/GERIATRIC route)'}

FORMAT:
<reasoning>
[Route: {route}]
[Your route-appropriate reasoning. 8-15 sentences. Follow route-specific rules exactly.]
</reasoning>
<answer>[Letter]. [Answer text]</answer>

Question: {question}
Options: {options}
Ground Truth: {answer}

Produce your route-classified reasoning now."""


# ──────────────────────────────────────────────────────────────────────
# CORE GENERATION LOOP
# ──────────────────────────────────────────────────────────────────────

def generate_pack(pack_id, count, shard_spec=None, dry_run=False):
    """Generate examples for a single pack."""
    cfg = PACK_CONFIG[pack_id]
    shard_id = None
    shard_total = None

    if shard_spec:
        parts = shard_spec.split("/")
        shard_id = int(parts[0])
        shard_total = int(parts[1])
        # Calculate this shard's slice
        shard_start = (count * shard_id) // shard_total
        shard_end = (count * (shard_id + 1)) // shard_total
        shard_count = shard_end - shard_start
        print(f"  Shard {shard_id}/{shard_total}: generating indices {shard_start}-{shard_end-1} ({shard_count} examples)")
    else:
        shard_start = 0
        shard_count = count

    paths = get_pack_paths(pack_id, shard_id)
    existing_hashes = load_existing_hashes(paths["train"]) if not dry_run else set()

    # Pre-load filtered data for P2/P3/P6
    filtered_pool = None
    if pack_id in ("P2", "P3", "P6"):
        print(f"  Loading MedReason 32K...")
        all_examples = load_medreason()
        print(f"  Loaded {len(all_examples)} examples")

        if pack_id == "P2":
            filtered_pool = filter_by_keywords(all_examples, P2_INCLUDE_KW, P2_EXCLUDE_KW)
        elif pack_id == "P3":
            filtered_pool = filter_by_keywords(all_examples, P3_INCLUDE_KW)
        elif pack_id == "P6":
            # Stratified by route
            by_route = defaultdict(list)
            for ex in all_examples:
                route = classify_route(ex.get("question", ""))
                if route in P6_ROUTE_TARGETS:
                    by_route[route].append(ex)
            # Sample according to targets
            filtered_pool = []
            for route, target_n in P6_ROUTE_TARGETS.items():
                pool = by_route.get(route, [])
                n = min(target_n, len(pool))
                if n > 0:
                    sampled = random.sample(pool, n)
                    for ex in sampled:
                        ex["_route"] = route
                    filtered_pool.extend(sampled)
                print(f"    Route {route}: {len(pool)} available, sampled {n}")
            random.shuffle(filtered_pool)

        print(f"  Filtered pool: {len(filtered_pool)} candidates")
        if shard_total:
            # Take shard slice of filtered pool
            pool_start = (len(filtered_pool) * shard_id) // shard_total
            pool_end = (len(filtered_pool) * (shard_id + 1)) // shard_total
            filtered_pool = filtered_pool[pool_start:pool_end]
            shard_count = min(shard_count, len(filtered_pool))
            print(f"  Shard slice: {len(filtered_pool)} candidates")

    stats = {"generated": 0, "failed": 0, "skipped": 0, "quality_issues": defaultdict(int)}

    for i in range(shard_count):
        global_idx = shard_start + i
        print(f"\n[{i+1}/{shard_count}] {cfg['name']} #{global_idx}")

        try:
            if pack_id == "P1":
                question, options_text, prompt = generate_p1_example(global_idx)
            elif pack_id in ("P2", "P3"):
                if i >= len(filtered_pool):
                    print("  Pool exhausted")
                    break
                source_ex = filtered_pool[i]
                question = source_ex["question"]
                options_text = source_ex.get("options", "")
                if pack_id == "P2":
                    prompt = build_p2_prompt(source_ex)
                else:
                    prompt = build_p3_prompt(source_ex)
            elif pack_id == "P4":
                question, options_text, prompt, correct_letter, drug = generate_p4_example(global_idx)
            elif pack_id == "P5":
                question, options_text, prompt = generate_p5_example(global_idx)
            elif pack_id == "P6":
                if i >= len(filtered_pool):
                    print("  Pool exhausted")
                    break
                source_ex = filtered_pool[i]
                route = source_ex.get("_route", classify_route(source_ex.get("question", "")))
                question = source_ex["question"]
                options_text = source_ex.get("options", "")
                prompt = build_p6_prompt(source_ex, route)

            # Dedup check
            content_hash = make_example_hash(question[:200])
            if content_hash in existing_hashes:
                print("  SKIP (duplicate)")
                stats["skipped"] += 1
                continue

            if dry_run:
                print(f"  [DRY RUN] Would generate for: {question[:80]}...")
                parsed = {"reasoning": "[dry-run placeholder reasoning]", "answer": "A. Placeholder"}
                stats["generated"] += 1
                continue

            # Call Codex
            print(f"  Generating...")
            raw, model = call_model(prompt)
            if not raw:
                print("  FAILED: no model output")
                stats["failed"] += 1
                log_entry({
                    "pack": pack_id, "idx": global_idx, "status": "api_failure",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, paths["log"])
                continue

            # Parse output
            parsed = parse_generation(raw)
            if not parsed.get("reasoning") or not parsed.get("answer"):
                # Try fallback: sometimes output uses <think> instead
                think_match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
                if think_match:
                    parsed["reasoning"] = think_match.group(1).strip()
                # Try to extract answer from remaining text
                if not parsed.get("answer"):
                    ans_match = re.search(r'(?:^|\n)([A-D])\.\s*(.+)', raw)
                    if ans_match:
                        parsed["answer"] = f"{ans_match.group(1)}. {ans_match.group(2).strip()}"

            if not parsed.get("reasoning") or not parsed.get("answer"):
                print(f"  FAILED: parse error (got keys: {list(parsed.keys())})")
                stats["failed"] += 1
                log_entry({
                    "pack": pack_id, "idx": global_idx, "status": "parse_failure",
                    "raw_len": len(raw), "timestamp": datetime.now(timezone.utc).isoformat(),
                }, paths["log"])
                continue

            # Validate quality
            if pack_id == "P6":
                issues = validate_p6(parsed, route)
            elif pack_id in VALIDATORS:
                issues = VALIDATORS[pack_id](parsed)
            else:
                issues = validate_common(parsed)

            for iss in issues:
                stats["quality_issues"][iss] += 1

            # Build metadata
            metadata = {
                "pack": pack_id,
                "pack_name": cfg["name"],
                "idx": global_idx,
                "content_hash": content_hash,
                "generator": model,
                "quality_issues": issues,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if pack_id in ("P2", "P3", "P6"):
                metadata["source"] = source_ex.get("dataset_name", "")
                metadata["source_id"] = source_ex.get("id_in_dataset", "")
            if pack_id == "P6":
                metadata["route"] = route
            if pack_id == "P4":
                metadata["drug"] = drug
                metadata["scenario"] = P4_SCENARIOS[global_idx % len(P4_SCENARIOS)]

            # Save
            entry = make_sft_entry(question, parsed["answer"], parsed["reasoning"], metadata)
            save_example(entry, paths["train"])
            existing_hashes.add(content_hash)
            stats["generated"] += 1

            qi_str = f" [QI: {len(issues)}]" if issues else ""
            print(f"  SAVED ({len(parsed['reasoning'])} chars, model={model}){qi_str}")

            log_entry({
                "pack": pack_id, "idx": global_idx, "status": "success",
                "model": model, "reasoning_len": len(parsed["reasoning"]),
                "quality_issues": issues,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, paths["log"])

            # Brief pause for rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"  ERROR: {e}")
            stats["failed"] += 1
            log_entry({
                "pack": pack_id, "idx": global_idx, "status": "error",
                "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat(),
            }, paths["log"])

    # Print summary
    print(f"\n{'='*60}")
    print(f"{cfg['name']} Summary")
    print(f"{'='*60}")
    print(f"  Generated: {stats['generated']}")
    print(f"  Failed:    {stats['failed']}")
    print(f"  Skipped:   {stats['skipped']}")
    if stats["quality_issues"]:
        print(f"  Quality issues:")
        for iss, count in sorted(stats["quality_issues"].items(), key=lambda x: -x[1]):
            print(f"    {iss}: {count}")

    return stats


# ──────────────────────────────────────────────────────────────────────
# MERGE + STATS
# ──────────────────────────────────────────────────────────────────────

def merge_shards(pack_id):
    """Merge shard files into single train.jsonl."""
    cfg = PACK_CONFIG[pack_id]
    pack_dir = BASE_DIR / cfg["dir"]

    shard_files = sorted(pack_dir.glob("train_shard*.jsonl"))
    if not shard_files:
        print(f"No shard files found for {pack_id}")
        return

    merged = []
    seen_hashes = set()
    for sf in shard_files:
        with open(sf) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    h = entry.get("metadata", {}).get("content_hash", "")
                    if h and h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    merged.append(entry)
                except json.JSONDecodeError:
                    continue

    output = pack_dir / "train.jsonl"
    with open(output, "w") as f:
        for entry in merged:
            f.write(json.dumps(entry) + "\n")

    print(f"Merged {len(shard_files)} shards → {output} ({len(merged)} examples)")

    # Save metadata
    metadata = {
        "pack": pack_id,
        "name": cfg["name"],
        "count": len(merged),
        "shards_merged": len(shard_files),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(pack_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def combine_all():
    """Combine all packs into combined/ directory."""
    combined_dir = BASE_DIR / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    all_entries = []
    pack_stats = {}

    for pack_id, cfg in PACK_CONFIG.items():
        train_file = BASE_DIR / cfg["dir"] / "train.jsonl"
        if not train_file.exists():
            print(f"  {pack_id} ({cfg['name']}): MISSING")
            continue

        count = 0
        quality_issues = defaultdict(int)
        with open(train_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    all_entries.append(entry)
                    count += 1
                    for qi in entry.get("metadata", {}).get("quality_issues", []):
                        quality_issues[qi] += 1
                except json.JSONDecodeError:
                    continue

        pack_stats[pack_id] = {
            "name": cfg["name"],
            "count": count,
            "quality_issues": dict(quality_issues),
        }
        print(f"  {pack_id} ({cfg['name']}): {count} examples")

    # Write combined file
    output = combined_dir / "all_packs_train.jsonl"
    random.shuffle(all_entries)
    with open(output, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    # Write stats
    stats = {
        "total": len(all_entries),
        "packs": pack_stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(combined_dir / "pack_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nCombined: {len(all_entries)} total examples → {output}")


def show_stats(pack_id):
    """Display stats for a pack."""
    cfg = PACK_CONFIG[pack_id]
    train_file = BASE_DIR / cfg["dir"] / "train.jsonl"

    if not train_file.exists():
        print(f"No data for {pack_id}")
        return

    entries = []
    with open(train_file) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"\n{'='*60}")
    print(f"{cfg['name']} Stats ({len(entries)} examples)")
    print(f"{'='*60}")

    # Quality issues
    qi_counts = defaultdict(int)
    clean = 0
    for e in entries:
        issues = e.get("metadata", {}).get("quality_issues", [])
        if not issues:
            clean += 1
        for qi in issues:
            qi_counts[qi] += 1

    print(f"  Clean (no issues): {clean}/{len(entries)} ({100*clean/max(1,len(entries)):.0f}%)")
    if qi_counts:
        print(f"  Quality issues:")
        for qi, n in sorted(qi_counts.items(), key=lambda x: -x[1]):
            print(f"    {qi}: {n}")

    # Reasoning length distribution
    lengths = [len(e["messages"][1]["content"]) for e in entries if len(e.get("messages", [])) > 1]
    if lengths:
        print(f"  Reasoning length: min={min(lengths)}, max={max(lengths)}, "
              f"mean={sum(lengths)/len(lengths):.0f}")

    # Route distribution (P6)
    if pack_id == "P6":
        routes = defaultdict(int)
        for e in entries:
            routes[e.get("metadata", {}).get("route", "unknown")] += 1
        print(f"  Route distribution:")
        for route, n in sorted(routes.items(), key=lambda x: -x[1]):
            print(f"    {route}: {n}")

    # Generator distribution
    generators = defaultdict(int)
    for e in entries:
        generators[e.get("metadata", {}).get("generator", "unknown")] += 1
    print(f"  Generators: {dict(generators)}")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PRISM Skill Pack Generator")
    parser.add_argument("--pack", type=str, choices=list(PACK_CONFIG.keys()),
                        help="Pack to generate (P1-P6)")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of examples (default: pack-specific)")
    parser.add_argument("--shard", type=str, default=None,
                        help="Shard spec for parallel runs (e.g., '0/5')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview generation without calling models")
    parser.add_argument("--merge", type=str, default=None,
                        help="Merge shards for this pack ID")
    parser.add_argument("--combine-all", action="store_true",
                        help="Combine all packs into combined/")
    parser.add_argument("--stats", type=str, default=None,
                        help="Show stats for a pack ID")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.merge:
        merge_shards(args.merge)
        return

    if args.combine_all:
        combine_all()
        return

    if args.stats:
        show_stats(args.stats)
        return

    if not args.pack:
        parser.print_help()
        print("\nAvailable packs:")
        for pid, cfg in PACK_CONFIG.items():
            print(f"  {pid}: {cfg['name']} ({cfg['default_count']} examples)")
        return

    count = args.count or PACK_CONFIG[args.pack]["default_count"]
    cfg = PACK_CONFIG[args.pack]

    print("=" * 60)
    print(f"PRISM Skill Pack: {cfg['name']} ({count} examples)")
    if args.shard:
        print(f"Shard: {args.shard}")
    if args.dry_run:
        print("MODE: DRY RUN")
    print("=" * 60)

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    generate_pack(args.pack, count, args.shard, args.dry_run)


if __name__ == "__main__":
    main()
