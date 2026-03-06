"""
Shared evaluation metrics for all experiments.

All experiments use the same metrics for fair comparison:
- 30-day readmission: AUC-ROC, AUC-PR
- Next-diagnosis: Top-k accuracy
- Lab trajectory: MAE
- Mortality: AUC-ROC, AUC-PR
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error


@dataclass
class ExperimentResults:
    """Container for experiment results."""
    experiment_name: str
    readmission_auroc: float = 0.0
    readmission_auprc: float = 0.0
    diagnosis_top1: float = 0.0
    diagnosis_top5: float = 0.0
    lab_mae: float = 0.0
    mortality_auroc: float = 0.0
    mortality_auprc: float = 0.0
    training_time_seconds: float = 0.0
    model_params: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "experiment": self.experiment_name,
            "readmission_auroc": f"{self.readmission_auroc:.4f}",
            "readmission_auprc": f"{self.readmission_auprc:.4f}",
            "diagnosis_top1": f"{self.diagnosis_top1:.4f}",
            "diagnosis_top5": f"{self.diagnosis_top5:.4f}",
            "lab_mae": f"{self.lab_mae:.4f}",
            "mortality_auroc": f"{self.mortality_auroc:.4f}",
            "mortality_auprc": f"{self.mortality_auprc:.4f}",
            "training_time_s": f"{self.training_time_seconds:.1f}",
            "params": f"{self.model_params:,}",
        }


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute AUC-ROC and AUC-PR for binary classification."""
    if len(np.unique(y_true)) < 2:
        return {"auroc": 0.0, "auprc": 0.0}
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
    }


def compute_topk_accuracy(y_true: list[set], y_pred: list[list], k: int = 5) -> float:
    """
    Compute top-k accuracy for multi-label prediction.
    y_true: list of sets of true labels
    y_pred: list of ranked prediction lists
    """
    hits = 0
    total = 0
    for true_set, pred_list in zip(y_true, y_pred):
        if not true_set:
            continue
        top_k = set(pred_list[:k])
        hits += len(true_set & top_k)
        total += len(true_set)
    return hits / total if total > 0 else 0.0


def compute_lab_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE for lab value trajectory prediction."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return 0.0
    return mean_absolute_error(y_true[mask], y_pred[mask])


def print_comparison_table(results: list[ExperimentResults]):
    """Print a formatted comparison table of all experiment results."""
    print("\n" + "=" * 100)
    print("ARCHITECTURE COMPARISON — LHM Health Trajectory Prediction")
    print("=" * 100)

    headers = [
        "Experiment", "Readm. AUROC", "Readm. AUPRC",
        "Diag Top-1", "Diag Top-5", "Lab MAE",
        "Mort. AUROC", "Mort. AUPRC", "Time(s)", "Params"
    ]
    row_format = "{:<25} {:>12} {:>12} {:>10} {:>10} {:>8} {:>11} {:>11} {:>8} {:>10}"

    print(row_format.format(*headers))
    print("-" * 100)

    for r in results:
        d = r.to_dict()
        print(row_format.format(
            d["experiment"], d["readmission_auroc"], d["readmission_auprc"],
            d["diagnosis_top1"], d["diagnosis_top5"], d["lab_mae"],
            d["mortality_auroc"], d["mortality_auprc"],
            d["training_time_s"], d["params"],
        ))

    print("=" * 100)
