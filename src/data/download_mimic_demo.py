"""
Download MIMIC-IV Clinical Database Demo (100 patients, no credentials needed).

This is for development and testing only. For the full dataset (300K+ admissions),
you need PhysioNet credentialed access: https://physionet.org/content/mimiciv/
"""

import os
import subprocess
import sys
from pathlib import Path


DEMO_URL = "https://physionet.org/static/published-projects/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2.zip"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
DEMO_DIR = DATA_DIR / "mimic-iv-demo"


def download_demo():
    """Download and extract MIMIC-IV demo dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DEMO_DIR.exists() and any(DEMO_DIR.rglob("*.csv*")):
        print(f"MIMIC-IV demo already exists at {DEMO_DIR}")
        return

    zip_path = DATA_DIR / "mimic-iv-demo.zip"

    print(f"Downloading MIMIC-IV demo to {zip_path}...")
    subprocess.run(
        ["curl", "-L", "-o", str(zip_path), DEMO_URL],
        check=True,
    )

    print("Extracting...")
    subprocess.run(
        ["unzip", "-o", str(zip_path), "-d", str(DATA_DIR)],
        check=True,
    )

    # The zip extracts to mimic-iv-clinical-database-demo-2.2/
    extracted = DATA_DIR / "mimic-iv-clinical-database-demo-2.2"
    if extracted.exists() and not DEMO_DIR.exists():
        extracted.rename(DEMO_DIR)
    elif extracted.exists():
        # Move contents
        import shutil
        shutil.copytree(extracted, DEMO_DIR, dirs_exist_ok=True)
        shutil.rmtree(extracted)

    # Cleanup zip
    zip_path.unlink(missing_ok=True)

    # Verify
    csv_files = list(DEMO_DIR.rglob("*.csv*"))
    print(f"Done. Found {len(csv_files)} CSV files in {DEMO_DIR}")
    for f in sorted(csv_files)[:10]:
        print(f"  {f.relative_to(DEMO_DIR)}")
    if len(csv_files) > 10:
        print(f"  ... and {len(csv_files) - 10} more")


if __name__ == "__main__":
    download_demo()
