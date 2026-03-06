"""
Experiment 5: Hybrid LHM Architecture.

Combines the best ideas from experiments 2-4:
- Mamba blocks for linear-complexity sequence modeling (Exp 2)
- Continuous-time position encoding (Exp 3)
- Medical tokenization + next-token generative objective (Exp 4)
- Sparse attention layers interleaved with Mamba (HyMaTE-inspired)

This is the first LHM backbone candidate.
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
# Components (reused from experiments 2-3)
# ---------------------------------------------------------------------------

class ContinuousTimeEncoding(nn.Module):
    """Learnable sinusoidal encoding for absolute timestamps."""

    def __init__(self, d_model: int):
        super().__init__()
        half = d_model // 2
        self.freq = nn.Parameter(torch.randn(half) * 0.01)
        self.phase = nn.Parameter(torch.zeros(half))
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, timestamps):
        t = timestamps.unsqueeze(-1)
        freq = F.softplus(self.freq)
        sin_enc = torch.sin(t * freq + self.phase)
        cos_enc = torch.cos(t * freq + self.phase)
        return self.linear(torch.cat([sin_enc, cos_enc], dim=-1))


class MambaBlock(nn.Module):
    """Simplified Mamba block: conv1d + gated SSM."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :x_branch.shape[1]]
        x_conv = x_conv.transpose(1, 2)
        x_branch = F.silu(x_conv)
        y = self._ssm(x_branch)
        y = y * F.silu(z)
        return self.out_proj(y) + residual

    def _ssm(self, x):
        B_sz, L, D = x.shape
        d_state = self.A_log.shape[1]
        A = -torch.exp(self.A_log)
        x_dbl = self.x_proj(x)
        B_mat, C_mat = x_dbl.split(d_state, dim=-1)
        dt = F.softplus(self.dt_proj(x))
        h = torch.zeros(B_sz, D, d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            dt_t = dt[:, t, :]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt_t.unsqueeze(-1) * B_mat[:, t, :].unsqueeze(1)
            x_t = x[:, t, :].unsqueeze(-1)
            h = A_bar * h + B_bar * x_t
            y_t = (h * C_mat[:, t, :].unsqueeze(1)).sum(-1) + self.D * x[:, t, :]
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)


class SparseAttentionBlock(nn.Module):
    """
    Lightweight attention block interleaved with Mamba.
    Uses temporal decay (from Exp 3) for time-aware attention.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.decay_rate = nn.Parameter(torch.ones(n_heads) * 0.01)

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model, bias=False),
        )

    def forward(self, x, timestamps=None, mask=None):
        B, L, D = x.shape
        H = self.n_heads

        residual = x
        x_norm = self.norm(x)

        qkv = self.qkv(x_norm).reshape(B, L, 3, H, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Temporal decay bias
        if timestamps is not None:
            t_diff = (timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)).abs()
            t_diff = t_diff.unsqueeze(1) / 24.0
            decay = F.softplus(self.decay_rate).view(1, H, 1, 1)
            attn = attn - decay * t_diff

        # Causal mask
        causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        if mask is not None:
            pad_mask = (1 - mask).bool().unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = residual + self.out_proj(out)

        x = x + self.ff(self.ff_norm(x))
        return x


# ---------------------------------------------------------------------------
# Hybrid LHM Model
# ---------------------------------------------------------------------------

class HybridLHM(nn.Module):
    """
    Hybrid architecture combining:
    - Medical token embeddings
    - Continuous-time position encoding
    - Alternating Mamba + sparse attention blocks (3:1 ratio)
    - Next-token prediction + classification heads
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        n_layers: int = 8,
        n_heads: int = 4,
        d_state: int = 16,
        n_token_types: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.type_embedding = nn.Embedding(n_token_types, d_model)
        self.time_encoding = ContinuousTimeEncoding(d_model)

        # Alternating blocks: 3 Mamba + 1 attention
        self.blocks = nn.ModuleList()
        self.block_types = []
        for i in range(n_layers):
            if (i + 1) % 4 == 0:
                self.blocks.append(SparseAttentionBlock(d_model, n_heads, dropout))
                self.block_types.append("attention")
            else:
                self.blocks.append(MambaBlock(d_model, d_state))
                self.block_types.append("mamba")

        self.norm = nn.LayerNorm(d_model)

        # Generative head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Classification heads
        self.readmission_head = nn.Linear(d_model, 1)
        self.mortality_head = nn.Linear(d_model, 1)

    def forward(self, token_ids, token_types, timestamps, attention_mask=None):
        x = self.token_embedding(token_ids) + self.type_embedding(token_types)
        x = x + self.time_encoding(timestamps)

        for block, btype in zip(self.blocks, self.block_types):
            if btype == "attention":
                x = block(x, timestamps, attention_mask)
            else:
                x = block(x)

        x = self.norm(x)
        cls_repr = x[:, 0, :]

        return {
            "lm_logits": self.lm_head(x),
            "readmission_logit": self.readmission_head(cls_repr).squeeze(-1),
            "mortality_logit": self.mortality_head(cls_repr).squeeze(-1),
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HybridDataset(Dataset):
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

        labels = tokens[1:] + [PAD_TOKEN]

        pad_len = self.max_len - seq_len
        tokens = tokens + [PAD_TOKEN] * pad_len
        types = types + [0] * pad_len
        times = times + [0.0] * pad_len
        labels = labels + [PAD_TOKEN] * pad_len
        mask = [1.0] * seq_len + [0.0] * pad_len

        return {
            "token_ids": torch.tensor(tokens, dtype=torch.long),
            "token_types": torch.tensor(types, dtype=torch.long),
            "timestamps": torch.tensor(times, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.float32),
            "readmission": torch.tensor(p.readmission_30d, dtype=torch.float32),
            "mortality": torch.tensor(p.mortality, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, device, lm_weight=0.5):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        token_ids = batch["token_ids"].to(device)
        token_types = batch["token_types"].to(device)
        timestamps = batch["timestamps"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)
        readmission = batch["readmission"].to(device)
        mortality = batch["mortality"].to(device)

        outputs = model(token_ids, token_types, timestamps, mask)

        lm_loss = F.cross_entropy(
            outputs["lm_logits"].view(-1, VOCAB_SIZE),
            labels.view(-1),
            ignore_index=PAD_TOKEN,
        )
        loss_r = F.binary_cross_entropy_with_logits(outputs["readmission_logit"], readmission)
        loss_m = F.binary_cross_entropy_with_logits(outputs["mortality_logit"], mortality)
        loss = lm_weight * lm_loss + (1 - lm_weight) * (loss_r + loss_m)

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
    all_readm, all_mort = [], []
    all_readm_t, all_mort_t = [], []
    total_loss = 0
    total_lm = 0
    n = 0

    for batch in dataloader:
        token_ids = batch["token_ids"].to(device)
        token_types = batch["token_types"].to(device)
        timestamps = batch["timestamps"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)
        readmission = batch["readmission"].to(device)
        mortality = batch["mortality"].to(device)

        outputs = model(token_ids, token_types, timestamps, mask)

        lm_loss = F.cross_entropy(
            outputs["lm_logits"].view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=PAD_TOKEN)
        loss_r = F.binary_cross_entropy_with_logits(outputs["readmission_logit"], readmission)
        loss_m = F.binary_cross_entropy_with_logits(outputs["mortality_logit"], mortality)

        total_loss += (loss_r + loss_m).item()
        total_lm += lm_loss.item()
        n += 1

        all_readm.append(torch.sigmoid(outputs["readmission_logit"]).cpu().numpy())
        all_mort.append(torch.sigmoid(outputs["mortality_logit"]).cpu().numpy())
        all_readm_t.append(readmission.cpu().numpy())
        all_mort_t.append(mortality.cpu().numpy())

    return {
        "loss": total_loss / max(n, 1),
        "lm_loss": total_lm / max(n, 1),
        "readmission": compute_binary_metrics(np.concatenate(all_readm_t), np.concatenate(all_readm)),
        "mortality": compute_binary_metrics(np.concatenate(all_mort_t), np.concatenate(all_mort)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 60)
    print("EXPERIMENT 5: Hybrid LHM (Mamba + Attention + Time + Tokens)")
    print("=" * 60)

    start_time = time.time()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n   Device: {device}")

    # Data
    print("\n1. Loading and tokenizing MIMIC-IV data...")
    records = build_patient_records()
    tokenized, code_to_idx, lab_quantiles = tokenize_all_patients(records, max_tokens=512)
    print(f"   Tokenized patients: {len(tokenized)}")

    np.random.seed(42)
    np.random.shuffle(tokenized)
    n_test = max(1, int(len(tokenized) * 0.15))
    n_val = max(1, int(len(tokenized) * 0.15))
    test_data = tokenized[:n_test]
    val_data = tokenized[n_test:n_test + n_val]
    train_data = tokenized[n_test + n_val:]
    print(f"   Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    max_len = 256
    train_loader = DataLoader(HybridDataset(train_data, max_len), batch_size=8, shuffle=True)
    val_loader = DataLoader(HybridDataset(val_data, max_len), batch_size=8)
    test_loader = DataLoader(HybridDataset(test_data, max_len), batch_size=8)

    # Model: 8 layers (6 Mamba + 2 attention), with time encoding
    print("\n2. Building Hybrid LHM model...")
    model = HybridLHM(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        n_layers=8,
        n_heads=4,
        d_state=16,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    mamba_count = sum(1 for t in model.block_types if t == "mamba")
    attn_count = sum(1 for t in model.block_types if t == "attention")
    print(f"   Parameters: {n_params:,}")
    print(f"   Block layout: {mamba_count} Mamba + {attn_count} Attention")

    # Train
    print("\n3. Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(25):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_results = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"   Epoch {epoch+1:2d} | Train: {train_loss:.4f} | "
              f"Val cls: {val_results['loss']:.4f} | LM: {val_results['lm_loss']:.4f} | "
              f"Readm: {val_results['readmission']['auroc']:.4f} | "
              f"Mort: {val_results['mortality']['auroc']:.4f}")

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

    # Test
    print("\n4. Evaluating on test set...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt", weights_only=True))
    test_results = evaluate(model, test_loader, device)

    print(f"   Readmission AUROC: {test_results['readmission']['auroc']:.4f} | "
          f"AUPRC: {test_results['readmission']['auprc']:.4f}")
    print(f"   Mortality AUROC: {test_results['mortality']['auroc']:.4f} | "
          f"AUPRC: {test_results['mortality']['auprc']:.4f}")
    print(f"   LM loss: {test_results['lm_loss']:.4f}")

    results = ExperimentResults(
        experiment_name="Hybrid LHM",
        readmission_auroc=test_results["readmission"]["auroc"],
        readmission_auprc=test_results["readmission"]["auprc"],
        mortality_auroc=test_results["mortality"]["auroc"],
        mortality_auprc=test_results["mortality"]["auprc"],
        training_time_seconds=train_time,
        model_params=n_params,
        notes=f"6 Mamba + 2 Attn blocks, time enc, next-token+cls, LM={test_results['lm_loss']:.4f}",
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
