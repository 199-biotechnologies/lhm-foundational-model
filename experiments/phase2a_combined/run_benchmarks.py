"""
Phase 2a: Extended benchmarks for Hybrid LHM on combined dataset.

Tests multiple clinical prediction tasks beyond readmission:
1. 30-day readmission (primary — already done)
2. 7-day readmission (urgent)
3. 90-day readmission (long-term)
4. Length of stay > 7 days (resource utilization)
5. Emergency readmission (unplanned return)
6. Multi-visit trajectory prediction (next diagnosis)
"""

import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.combined_loader import build_combined_records
from src.data.medical_tokenizer import (
    tokenize_all_patients, TokenizedPatient,
    VOCAB_SIZE, PAD_TOKEN, CLS_TOKEN,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"


# ── Architectures (same as run_neural.py) ────────────────────────────────

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state+1, dtype=torch.float32)).unsqueeze(0).expand(d_inner, -1))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_part.transpose(1, 2))[:, :, :x_part.shape[1]].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)
        A = -torch.exp(self.A_log)
        batch, seq_len, d_inner = x_conv.shape
        h = torch.zeros(batch, d_inner, A.shape[1], device=x.device)
        outputs = []
        for t in range(seq_len):
            h = h * torch.exp(A.unsqueeze(0) * dt[:, t, :].unsqueeze(-1)) + \
                x_conv[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1) * dt[:, t, :].unsqueeze(-1)
            outputs.append((h * C[:, t, :].unsqueeze(1)).sum(-1))
        y = torch.stack(outputs, dim=1)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return residual + self.out_proj(y)


class ContinuousTimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(d_model // 2) * 0.1)
        self.phases = nn.Parameter(torch.zeros(d_model // 2))
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, timestamps):
        t = timestamps.unsqueeze(-1)
        enc = torch.cat([torch.sin(t * self.freqs + self.phases),
                         torch.cos(t * self.freqs + self.phases)], dim=-1)
        return self.linear(enc)


class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(),
                                 nn.Linear(d_model*4, d_model), nn.Dropout(dropout))

    def forward(self, x, timestamps=None):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class HybridLHM(nn.Module):
    """Multi-task Hybrid LHM with configurable output heads."""

    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_mamba=6, n_attn=2, n_tasks=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.type_embedding = nn.Embedding(10, d_model)
        self.time_encoding = ContinuousTimeEncoding(d_model)

        total = n_mamba + n_attn
        attn_every = total // n_attn if n_attn > 0 else total + 1
        blocks = []
        self._block_types = []
        for i in range(total):
            if (i + 1) % attn_every == 0:
                blocks.append(TemporalAttentionBlock(d_model))
                self._block_types.append("attn")
            else:
                blocks.append(MambaBlock(d_model))
                self._block_types.append("mamba")
        self.blocks = nn.ModuleList(blocks)

        self.task_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
            for _ in range(n_tasks)
        ])

    def forward(self, input_ids, token_types, timestamps, **kwargs):
        x = self.embedding(input_ids) + self.type_embedding(token_types) + self.time_encoding(timestamps)
        for btype, block in zip(self._block_types, self.blocks):
            if btype == "attn":
                x = block(x, timestamps)
            else:
                x = block(x)
        cls_repr = x[:, 0, :]
        return [head(cls_repr).squeeze(-1) for head in self.task_heads]


# ── Dataset ──────────────────────────────────────────────────────────────

class MultiTaskEHRDataset(Dataset):
    def __init__(self, patients, labels_dict, max_len=512):
        self.patients = patients
        self.labels = labels_dict  # {task_name: [label_per_patient]}
        self.max_len = max_len
        self.task_names = list(labels_dict.keys())

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        tokens = p.token_ids[:self.max_len]
        types = p.token_types[:self.max_len]
        times = p.timestamps[:self.max_len]
        pad_len = self.max_len - len(tokens)
        tokens = tokens + [PAD_TOKEN] * pad_len
        types = types + [0] * pad_len
        times = times + [0.0] * pad_len

        item = {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "token_types": torch.tensor(types, dtype=torch.long),
            "timestamps": torch.tensor(times, dtype=torch.float32),
        }
        for task_name in self.task_names:
            item[task_name] = torch.tensor(self.labels[task_name][idx], dtype=torch.float32)
        return item


# ── Label Generation ─────────────────────────────────────────────────────

def compute_labels(records, tokenized_subject_ids):
    """Compute multiple clinical prediction labels from records."""
    import pandas as pd

    labels = {
        "readmission_30d": [],
        "readmission_7d": [],
        "readmission_90d": [],
        "long_los": [],         # length of stay > 7 days
        "high_utilization": [], # > 5 visits total
    }

    for sid in tokenized_subject_ids:
        patient = records[records["subject_id"] == sid].sort_values("admittime")

        if len(patient) == 0:
            for k in labels:
                labels[k].append(0)
            continue

        last_row = patient.iloc[-1]

        # Readmission labels (based on second-to-last visit predicting last)
        if len(patient) >= 2:
            prev = patient.iloc[-2]
            try:
                gap = (pd.to_datetime(last_row["admittime"]) - pd.to_datetime(prev["dischtime"])).total_seconds() / 86400
                labels["readmission_7d"].append(1 if gap <= 7 else 0)
                labels["readmission_30d"].append(1 if gap <= 30 else 0)
                labels["readmission_90d"].append(1 if gap <= 90 else 0)
            except Exception:
                labels["readmission_7d"].append(0)
                labels["readmission_30d"].append(0)
                labels["readmission_90d"].append(0)
        else:
            labels["readmission_7d"].append(0)
            labels["readmission_30d"].append(0)
            labels["readmission_90d"].append(0)

        # Length of stay
        try:
            los = (pd.to_datetime(last_row["dischtime"]) - pd.to_datetime(last_row["admittime"])).total_seconds() / 86400
            labels["long_los"].append(1 if los > 7 else 0)
        except Exception:
            labels["long_los"].append(0)

        # High utilization
        labels["high_utilization"].append(1 if len(patient) > 5 else 0)

    return labels


# ── Training & Evaluation ────────────────────────────────────────────────

def train_and_evaluate(task_name, model, train_loader, val_loader, test_loader,
                       device, epochs=30, lr=1e-3, task_idx=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**{k: batch[k] for k in ["input_ids", "token_types", "timestamps"]})
            loss = loss_fn(outputs[task_idx], batch[task_name])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**{k: batch[k] for k in ["input_ids", "token_types", "timestamps"]})
                val_loss += loss_fn(outputs[task_idx], batch[task_name]).item()
        avg = val_loss / max(len(val_loader), 1)
        if avg < best_val_loss:
            best_val_loss = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**{k: batch[k] for k in ["input_ids", "token_types", "timestamps"]})
            all_preds.append(torch.sigmoid(outputs[task_idx]).cpu())
            all_labels.append(batch[task_name].cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    metrics = {"n_positive": int(labels.sum()), "n_total": len(labels)}
    if len(np.unique(labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(labels, preds))
        metrics["auprc"] = float(average_precision_score(labels, preds))
        metrics["f1"] = float(f1_score(labels, (preds > 0.5).astype(int)))
    else:
        metrics["auroc"] = 0.5
        metrics["auprc"] = 0.0
        metrics["f1"] = 0.0

    return metrics


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(" Phase 2a: Extended Benchmarks for Hybrid LHM")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load data
    print("\n1. Loading combined data...")
    records = build_combined_records()

    print("\n2. Tokenizing patients...")
    tokenized, code_to_idx, lab_quantiles = tokenize_all_patients(records, max_tokens=512)
    subject_ids = [p.subject_id for p in tokenized]
    print(f"   Tokenized: {len(tokenized)} patients")

    # Compute labels
    print("\n3. Computing multi-task labels...")
    labels = compute_labels(records, subject_ids)
    for task, vals in labels.items():
        pos_rate = sum(vals) / len(vals)
        print(f"   {task}: {sum(vals)}/{len(vals)} positive ({pos_rate:.1%})")

    # Split
    np.random.seed(42)
    indices = np.arange(len(tokenized))
    np.random.shuffle(indices)
    n = len(indices)
    n_test = max(1, int(n * 0.15))
    n_val = max(1, int(n * 0.15))
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    def subset(data_list, idx):
        return [data_list[i] for i in idx]

    train_patients = subset(tokenized, train_idx)
    val_patients = subset(tokenized, val_idx)
    test_patients = subset(tokenized, test_idx)

    task_names = list(labels.keys())

    all_results = {}

    for task_i, task_name in enumerate(task_names):
        print(f"\n{'─' * 70}")
        print(f" Benchmark: {task_name}")
        print(f"{'─' * 70}")

        train_labels = {task_name: subset(labels[task_name], train_idx)}
        val_labels = {task_name: subset(labels[task_name], val_idx)}
        test_labels = {task_name: subset(labels[task_name], test_idx)}

        train_ds = MultiTaskEHRDataset(train_patients, train_labels)
        val_ds = MultiTaskEHRDataset(val_patients, val_labels)
        test_ds = MultiTaskEHRDataset(test_patients, test_labels)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        test_loader = DataLoader(test_ds, batch_size=32)

        # Fresh model per task
        model = HybridLHM(d_model=128, n_mamba=6, n_attn=2, n_tasks=1).to(device)
        t0 = time.time()
        metrics = train_and_evaluate(
            task_name, model, train_loader, val_loader, test_loader,
            device, epochs=25, task_idx=0,
        )
        elapsed = time.time() - t0
        metrics["training_time_s"] = round(elapsed, 1)

        print(f"   AUROC: {metrics['auroc']:.4f}")
        print(f"   AUPRC: {metrics['auprc']:.4f}")
        print(f"   F1:    {metrics['f1']:.4f}")
        print(f"   Positives: {metrics['n_positive']}/{metrics['n_total']}")
        print(f"   Time: {elapsed:.0f}s")

        all_results[task_name] = metrics

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "extended_benchmarks.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print(" EXTENDED BENCHMARK RESULTS — HYBRID LHM")
    print(f"{'=' * 70}")
    print(f" {'Task':<25} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Pos%':>8} {'Time':>8}")
    print(f" {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for task, m in all_results.items():
        pos_pct = m["n_positive"] / m["n_total"] * 100
        print(f" {task:<25} {m['auroc']:>8.4f} {m['auprc']:>8.4f} {m['f1']:>8.4f} {pos_pct:>7.1f}% {m['training_time_s']:>7.0f}s")
    print(f"{'=' * 70}")
    print(f"\nResults saved to {OUTPUT_DIR / 'extended_benchmarks.json'}")


if __name__ == "__main__":
    main()
