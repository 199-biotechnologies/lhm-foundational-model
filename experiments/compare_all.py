"""
Load and compare results from all completed experiments.
"""

import json
from pathlib import Path

from src.evaluation.metrics import ExperimentResults, print_comparison_table


EXPERIMENTS_DIR = Path(__file__).parent


def load_result(exp_dir: Path) -> ExperimentResults | None:
    results_path = exp_dir / "outputs" / "results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        d = json.load(f)

    return ExperimentResults(
        experiment_name=d.get("experiment", exp_dir.name),
        readmission_auroc=float(d.get("readmission_auroc", 0)),
        readmission_auprc=float(d.get("readmission_auprc", 0)),
        diagnosis_top1=float(d.get("diagnosis_top1", 0)),
        diagnosis_top5=float(d.get("diagnosis_top5", 0)),
        lab_mae=float(d.get("lab_mae", 0)),
        mortality_auroc=float(d.get("mortality_auroc", 0)),
        mortality_auprc=float(d.get("mortality_auprc", 0)),
        training_time_seconds=float(d.get("training_time_s", 0)),
        model_params=int(d.get("params", "0").replace(",", "")),
    )


def main():
    exp_dirs = sorted(EXPERIMENTS_DIR.glob("exp*"))
    results = []
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue
        r = load_result(exp_dir)
        if r:
            results.append(r)
            print(f"  Loaded: {r.experiment_name}")

    if results:
        print_comparison_table(results)
    else:
        print("No experiment results found.")


if __name__ == "__main__":
    main()
