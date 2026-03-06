"""
Experiment 3: Continuous-Time Model (TrajGPT / ContiFormer style).

Tests whether encoding exact timestamps and time gaps improves trajectory
prediction on irregularly-sampled clinical data.

Architecture: Time-aware embeddings → Temporal attention with exponential
decay → task heads.

Key property: Attention weights decay with temporal distance, and the model
uses continuous time encodings rather than discrete position embeddings.
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
# Continuous-Time Position Encoding
# ---------------------------------------------------------------------------

class ContinuousTimeEncoding(nn.Module):
    """
    Encode absolute timestamps using learnable sinusoidal functions.
    Unlike discrete position embeddings, this handles irregular time gaps.
    """

    def __init__(self, d_model: int, max_period: float = 87600.0):
        super().__init__()
        # max_period ~10 years in hours
        self.d_model = d_model
        half = d_model // 2
        # Learnable frequency and phase
        self.freq = nn.Parameter(torch.randn(half) * 0.01)
        self.phase = nn.Parameter(torch.zeros(half))
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, timestamps):
        """timestamps: (B, L) float tensor in hours."""
        t = timestamps.unsqueeze(-1)  # (B, L, 1)
        freq = F.softplus(self.freq)  # positive frequencies
        # Sinusoidal encoding
        sin_enc = torch.sin(t * freq + self.phase)
        cos_enc = torch.cos(t * freq + self.phase)
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)  # (B, L, d_model)
        return self.linear(encoding)


# ---------------------------------------------------------------------------
# Time-Aware Attention
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Multi-head attention with temporal decay bias.
    Attention weights are biased by exp(-alpha * |t_i - t_j|),
    so temporally distant events have weaker influence (unless learned otherwise).
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable temporal decay rate per head
        self.decay_rate = nn.Parameter(torch.ones(n_heads) * 0.01)

    def forward(self, x, timestamps, mask=None):
        """
        x: (B, L, D)
        timestamps: (B, L) in hours
        mask: (B, L) float, 1=valid, 0=pad
        """
        B, L, D = x.shape
        H = self.n_heads

        qkv = self.qkv(x).reshape(B, L, 3, H, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, d_head)

        # Standard attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        # Temporal decay bias
        t_diff = timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)  # (B, L, L)
        t_diff = t_diff.abs().unsqueeze(1)  # (B, 1, L, L)
        decay = F.softplus(self.decay_rate).view(1, H, 1, 1)  # per-head decay
        temporal_bias = -decay * t_diff / 24.0  # normalize to days
        attn = attn + temporal_bias

        # Causal mask (optional, for autoregressive tasks)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Padding mask
        if mask is not None:
            pad_mask = (1 - mask).bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attn = attn.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Continuous-Time Transformer Block
# ---------------------------------------------------------------------------

class ContinuousTimeBlock(nn.Module):
    """Transformer block with temporal attention."""

    def __init__(self, d_model: int, n_heads: int = 4, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TemporalAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, timestamps, mask=None):
        x = x + self.attn(self.norm1(x), timestamps, mask)
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Continuous-Time EHR Model
# ---------------------------------------------------------------------------

class ContinuousTimeEHR(nn.Module):
    """
    Continuous-time transformer for EHR sequences.
    Uses real timestamps instead of discrete positions.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_token_types: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.type_embedding = nn.Embedding(n_token_types, d_model)
        self.time_encoding = ContinuousTimeEncoding(d_model)

        self.blocks = nn.ModuleList([
            ContinuousTimeBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.readmission_head = nn.Linear(d_model, 1)
        self.mortality_head = nn.Linear(d_model, 1)

    def forward(self, token_ids, token_types, timestamps, attention_mask=None):
        x = self.token_embedding(token_ids) + self.type_embedding(token_types)
        x = x + self.time_encoding(timestamps)

        for block in self.blocks:
            x = block(x, timestamps, attention_mask)

        x = self.norm(x)
        cls_repr = x[:, 0, :]

        return {
            "readmission_logit": self.readmission_head(cls_repr).squeeze(-1),
            "mortality_logit": self.mortality_head(cls_repr).squeeze(-1),
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ContinuousTimeDataset(Dataset):
    """Dataset with timestamps preserved for continuous-time modeling."""

    def __init__(self, patients: list[TokenizedPatient], max_len: int = 256):
        self.patients = patients
        self.max_len = max_len

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        tokens = p.token_ids[:self.max_len]
        types = p.token_types[:self.max_len]
        times = p.timestamps[:self.max_len]
        seq_len = len(tokens)

        pad_len = self.max_len - seq_len
        tokens = tokens + [PAD_TOKEN] * pad_len
        types = types + [0] * pad_len
        times = times + [0.0] * pad_len
        mask = [1.0] * seq_len + [0.0] * pad_len

        return {
            "token_ids": torch.tensor(tokens, dtype=torch.long),
            "token_types": torch.tensor(types, dtype=torch.long),
            "timestamps": torch.tensor(times, dtype=torch.float32),
            "attention_mask": torch.tensor(mask, dtype=torch.float32),
            "readmission": torch.tensor(p.readmission_30d, dtype=torch.float32),
            "mortality": torch.tensor(p.mortality, dtype=torch.float32),
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
        timestamps = batch["timestamps"].to(device)
        mask = batch["attention_mask"].to(device)
        readmission = batch["readmission"].to(device)
        mortality = batch["mortality"].to(device)

        outputs = model(token_ids, token_types, timestamps, mask)

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
        timestamps = batch["timestamps"].to(device)
        mask = batch["attention_mask"].to(device)
        readmission = batch["readmission"].to(device)
        mortality = batch["mortality"].to(device)

        outputs = model(token_ids, token_types, timestamps, mask)

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
    print("EXPERIMENT 3: Continuous-Time Model")
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

    # Split
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
    train_dataset = ContinuousTimeDataset(train_data, max_len=max_len)
    val_dataset = ContinuousTimeDataset(val_data, max_len=max_len)
    test_dataset = ContinuousTimeDataset(test_data, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Model
    print("\n2. Building Continuous-Time model...")
    model = ContinuousTimeEHR(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        n_layers=4,
        n_heads=4,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # Training
    print("\n3. Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
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
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - start_time

    # Test evaluation
    print("\n4. Evaluating on test set...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt", weights_only=True))
    test_results = evaluate(model, test_loader, device)

    print(f"   Readmission AUROC: {test_results['readmission']['auroc']:.4f} | "
          f"AUPRC: {test_results['readmission']['auprc']:.4f}")
    print(f"   Mortality AUROC: {test_results['mortality']['auroc']:.4f} | "
          f"AUPRC: {test_results['mortality']['auprc']:.4f}")

    # Results
    results = ExperimentResults(
        experiment_name="Continuous-Time",
        readmission_auroc=test_results["readmission"]["auroc"],
        readmission_auprc=test_results["readmission"]["auprc"],
        mortality_auroc=test_results["mortality"]["auroc"],
        mortality_auprc=test_results["mortality"]["auprc"],
        training_time_seconds=train_time,
        model_params=n_params,
        notes="d_model=128, n_layers=4, temporal_attention",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    print(f"\n5. Results saved to {OUTPUT_DIR / 'results.json'}")
    print(f"   Training time: {train_time:.1f}s")

    print_comparison_table([results])
    return results


if __name__ == "__main__":
    run_experiment()
