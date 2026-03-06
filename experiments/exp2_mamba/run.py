"""
Experiment 2: EHRMamba — State Space Model for EHR sequences.

Tests whether linear-complexity sequence modeling (Mamba/SSM) outperforms
transformers on long patient histories.

Architecture: Embedding → Mamba blocks (conv + gated SSM) → task heads
Key property: O(n) complexity vs O(n²) for transformers.
"""

import json
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.data.mimic_loader import build_patient_records
from src.data.medical_tokenizer import (
    tokenize_all_patients, TokenizedPatient,
    VOCAB_SIZE, PAD_TOKEN, CLS_TOKEN,
)
from src.evaluation.metrics import (
    ExperimentResults, compute_binary_metrics, print_comparison_table,
)


OUTPUT_DIR = Path(__file__).parent / "outputs"


# ---------------------------------------------------------------------------
# Mamba Block (pure PyTorch — no CUDA selective scan dependency)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    Simplified Mamba block: 1D conv → gated SSM → output projection.
    Captures the core Mamba inductive bias (selective state spaces)
    without requiring the CUDA-only mamba-ssm package.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)

        # SSM parameters (S4-style discretized)
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)  # B and C projections
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)

        # Learnable A matrix (log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        residual = x
        x = self.norm(x)

        # Project and split into two branches
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # 1D causal convolution
        x_conv = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :x_branch.shape[1]]  # causal: trim
        x_conv = x_conv.transpose(1, 2)
        x_branch = F.silu(x_conv)

        # Selective SSM
        y = self._ssm(x_branch)

        # Gate and project
        y = y * F.silu(z)
        return self.out_proj(y) + residual

    def _ssm(self, x):
        """Discretized state-space model scan."""
        B_sz, L, D = x.shape
        d_state = self.A_log.shape[1]

        A = -torch.exp(self.A_log)  # (D, d_state)
        D_param = self.D

        # Input-dependent B, C
        x_dbl = self.x_proj(x)  # (B, L, 2*d_state)
        B_mat, C_mat = x_dbl.split(d_state, dim=-1)  # each (B, L, d_state)

        # Time step
        dt = F.softplus(self.dt_proj(x))  # (B, L, D)

        # Discretize: A_bar = exp(dt * A)
        # For efficiency, do the recurrence step by step
        h = torch.zeros(B_sz, D, d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            dt_t = dt[:, t, :]  # (B, D)
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, D, d_state)
            B_bar = dt_t.unsqueeze(-1) * B_mat[:, t, :].unsqueeze(1)  # (B, D, d_state)
            x_t = x[:, t, :].unsqueeze(-1)  # (B, D, 1)

            h = A_bar * h + B_bar * x_t
            y_t = (h * C_mat[:, t, :].unsqueeze(1)).sum(-1)  # (B, D)
            y_t = y_t + D_param * x[:, t, :]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, D)


# ---------------------------------------------------------------------------
# EHRMamba Model
# ---------------------------------------------------------------------------

class EHRMamba(nn.Module):
    """
    Mamba-based model for EHR token sequences.
    Embedding → N Mamba blocks → task-specific classification heads.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        n_token_types: int = 5,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.type_embedding = nn.Embedding(n_token_types, d_model)

        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)
        ])

        self.readmission_head = nn.Linear(d_model, 1)
        self.mortality_head = nn.Linear(d_model, 1)

    def forward(self, token_ids, token_types, attention_mask=None):
        """
        token_ids: (B, L) int
        token_types: (B, L) int
        Returns: dict with 'readmission_logit' and 'mortality_logit'
        """
        x = self.token_embedding(token_ids) + self.type_embedding(token_types)

        for block in self.blocks:
            x = block(x)

        # Pool: use CLS token position (first token) representation
        cls_repr = x[:, 0, :]

        return {
            "readmission_logit": self.readmission_head(cls_repr).squeeze(-1),
            "mortality_logit": self.mortality_head(cls_repr).squeeze(-1),
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EHRTokenDataset(Dataset):
    """Dataset of tokenized patient sequences for classification."""

    def __init__(self, patients: list[TokenizedPatient], max_len: int = 512):
        self.patients = patients
        self.max_len = max_len

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        tokens = p.token_ids[:self.max_len]
        types = p.token_types[:self.max_len]
        seq_len = len(tokens)

        # Pad
        pad_len = self.max_len - seq_len
        tokens = tokens + [PAD_TOKEN] * pad_len
        types = types + [0] * pad_len
        mask = [1] * seq_len + [0] * pad_len

        return {
            "token_ids": torch.tensor(tokens, dtype=torch.long),
            "token_types": torch.tensor(types, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.float),
            "readmission": torch.tensor(p.readmission_30d, dtype=torch.float),
            "mortality": torch.tensor(p.mortality, dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        token_ids = batch["token_ids"].to(device)
        token_types = batch["token_types"].to(device)
        readmission = batch["readmission"].to(device)
        mortality = batch["mortality"].to(device)

        outputs = model(token_ids, token_types)

        loss_r = F.binary_cross_entropy_with_logits(outputs["readmission_logit"], readmission)
        loss_m = F.binary_cross_entropy_with_logits(outputs["mortality_logit"], mortality)
        loss = loss_r + loss_m

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_readm_logits = []
    all_mort_logits = []
    all_readm_true = []
    all_mort_true = []
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        token_ids = batch["token_ids"].to(device)
        token_types = batch["token_types"].to(device)
        readmission = batch["readmission"].to(device)
        mortality = batch["mortality"].to(device)

        outputs = model(token_ids, token_types)

        loss_r = F.binary_cross_entropy_with_logits(outputs["readmission_logit"], readmission)
        loss_m = F.binary_cross_entropy_with_logits(outputs["mortality_logit"], mortality)
        total_loss += (loss_r + loss_m).item()
        n_batches += 1

        all_readm_logits.append(torch.sigmoid(outputs["readmission_logit"]).cpu().numpy())
        all_mort_logits.append(torch.sigmoid(outputs["mortality_logit"]).cpu().numpy())
        all_readm_true.append(readmission.cpu().numpy())
        all_mort_true.append(mortality.cpu().numpy())

    y_readm_pred = np.concatenate(all_readm_logits)
    y_readm_true = np.concatenate(all_readm_true)
    y_mort_pred = np.concatenate(all_mort_logits)
    y_mort_true = np.concatenate(all_mort_true)

    readm_metrics = compute_binary_metrics(y_readm_true, y_readm_pred)
    mort_metrics = compute_binary_metrics(y_mort_true, y_mort_pred)

    return {
        "loss": total_loss / max(n_batches, 1),
        "readmission": readm_metrics,
        "mortality": mort_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 60)
    print("EXPERIMENT 2: EHRMamba (State Space Model)")
    print("=" * 60)

    start_time = time.time()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n   Device: {device}")

    # Load and tokenize data
    print("\n1. Loading and tokenizing MIMIC-IV data...")
    records = build_patient_records()
    tokenized, code_to_idx, lab_quantiles = tokenize_all_patients(records, max_tokens=512)
    print(f"   Tokenized patients: {len(tokenized)}")
    print(f"   Diagnosis vocab: {len(code_to_idx)} codes")

    # Split by subject_id
    np.random.seed(42)
    np.random.shuffle(tokenized)
    n_test = max(1, int(len(tokenized) * 0.15))
    n_val = max(1, int(len(tokenized) * 0.15))
    test_data = tokenized[:n_test]
    val_data = tokenized[n_test:n_test + n_val]
    train_data = tokenized[n_test + n_val:]
    print(f"   Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # Datasets
    max_len = 256
    train_dataset = EHRTokenDataset(train_data, max_len=max_len)
    val_dataset = EHRTokenDataset(val_data, max_len=max_len)
    test_dataset = EHRTokenDataset(test_data, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Model
    print("\n2. Building EHRMamba model...")
    model = EHRMamba(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        n_layers=4,
        d_state=16,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {n_params:,} (all trainable)")

    # Training
    print("\n3. Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_results = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"   Epoch {epoch+1:2d} | Train loss: {train_loss:.4f} | "
              f"Val loss: {val_results['loss']:.4f} | "
              f"Readm AUROC: {val_results['readmission']['auroc']:.4f} | "
              f"Mort AUROC: {val_results['mortality']['auroc']:.4f}")

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            patience_counter = 0
            # Save best model
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - start_time

    # Load best and evaluate on test
    print("\n4. Evaluating on test set...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt", weights_only=True))
    test_results = evaluate(model, test_loader, device)

    print(f"   Readmission AUROC: {test_results['readmission']['auroc']:.4f} | "
          f"AUPRC: {test_results['readmission']['auprc']:.4f}")
    print(f"   Mortality AUROC: {test_results['mortality']['auroc']:.4f} | "
          f"AUPRC: {test_results['mortality']['auprc']:.4f}")

    # Results
    results = ExperimentResults(
        experiment_name="EHRMamba (SSM)",
        readmission_auroc=test_results["readmission"]["auroc"],
        readmission_auprc=test_results["readmission"]["auprc"],
        mortality_auroc=test_results["mortality"]["auroc"],
        mortality_auprc=test_results["mortality"]["auprc"],
        training_time_seconds=train_time,
        model_params=n_params,
        notes=f"d_model=128, n_layers=4, d_state=16",
    )

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"\n5. Results saved to {OUTPUT_DIR / 'results.json'}")
    print(f"   Training time: {train_time:.1f}s")

    print_comparison_table([results])
    return results


if __name__ == "__main__":
    run_experiment()
