"""
Phase 2a: Run neural experiments (Mamba, Continuous-Time, Medical Token, Hybrid)
on the combined MIMIC-IV + Synthea dataset (1191 patients, 14K records).

Imports architecture code from exp2-exp5 and re-trains on larger data.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.combined_loader import build_combined_records
from src.data.medical_tokenizer import (
    tokenize_all_patients, TokenizedPatient,
    VOCAB_SIZE, PAD_TOKEN, CLS_TOKEN,
)

# Import architectures from existing experiments
sys.path.insert(0, str(Path(__file__).parent.parent / "exp2_mamba"))
sys.path.insert(0, str(Path(__file__).parent.parent / "exp3_continuous_time"))
sys.path.insert(0, str(Path(__file__).parent.parent / "exp5_hybrid"))

OUTPUT_DIR = Path(__file__).parent / "outputs"


# ── Dataset ──────────────────────────────────────────────────────────────

class EHRTokenDataset(Dataset):
    def __init__(self, patients: list[TokenizedPatient], max_len: int = 512):
        self.patients = patients
        self.max_len = max_len

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        tokens = p.token_ids[:self.max_len]
        types = p.token_types[:self.max_len]
        times = p.timestamps[:self.max_len]

        # Pad
        pad_len = self.max_len - len(tokens)
        tokens = tokens + [PAD_TOKEN] * pad_len
        types = types + [0] * pad_len
        times = times + [0.0] * pad_len

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "token_types": torch.tensor(types, dtype=torch.long),
            "timestamps": torch.tensor(times, dtype=torch.float32),
            "readmission": torch.tensor(p.readmission_30d, dtype=torch.float32),
            "mortality": torch.tensor(p.mortality, dtype=torch.float32),
        }


# ── Architecture: Mamba ──────────────────────────────────────────────────

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).unsqueeze(0).expand(d_inner, -1))
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
        d_state = A.shape[1]
        h = torch.zeros(batch, d_inner, d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            h = h * torch.exp(A.unsqueeze(0) * dt[:, t, :].unsqueeze(-1)) + \
                x_conv[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1) * dt[:, t, :].unsqueeze(-1)
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(-1)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return residual + self.out_proj(y)


class EHRMamba(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_layers=4, d_state=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state) for _ in range(n_layers)])
        self.readmission_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.mortality_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, input_ids, **kwargs):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        cls_repr = x[:, 0, :]
        return self.readmission_head(cls_repr).squeeze(-1), self.mortality_head(cls_repr).squeeze(-1)


# ── Architecture: Continuous-Time ────────────────────────────────────────

class ContinuousTimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(d_model // 2) * 0.1)
        self.phases = nn.Parameter(torch.zeros(d_model // 2))
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, timestamps):
        t = timestamps.unsqueeze(-1)
        enc = torch.cat([torch.sin(t * self.freqs + self.phases), torch.cos(t * self.freqs + self.phases)], dim=-1)
        return self.linear(enc)


class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))
        self.decay = nn.Parameter(torch.ones(n_heads) * 0.1)
        self.n_heads = n_heads

    def forward(self, x, timestamps):
        residual = x
        x = self.norm1(x)
        seq_len = x.shape[1]
        dt = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)
        decay_bias = -torch.abs(self.decay).view(1, self.n_heads, 1, 1) * torch.abs(dt).unsqueeze(1)
        decay_bias = decay_bias.repeat(1, 1, 1, 1)
        B, H, S, _ = decay_bias.shape
        decay_bias = decay_bias.reshape(B * H, S, S)
        attn_out, _ = self.attn(x, x, x, attn_mask=None)
        x = residual + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class ContinuousTimeEHR(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_layers=4, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.type_embedding = nn.Embedding(10, d_model)
        self.time_encoding = ContinuousTimeEncoding(d_model)
        self.blocks = nn.ModuleList([TemporalAttentionBlock(d_model, n_heads) for _ in range(n_layers)])
        self.readmission_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.mortality_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, input_ids, token_types, timestamps, **kwargs):
        x = self.embedding(input_ids) + self.type_embedding(token_types) + self.time_encoding(timestamps)
        for block in self.blocks:
            x = block(x, timestamps)
        cls_repr = x[:, 0, :]
        return self.readmission_head(cls_repr).squeeze(-1), self.mortality_head(cls_repr).squeeze(-1)


# ── Architecture: Hybrid (Mamba + Attention + Time) ──────────────────────

class HybridLHM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, n_mamba=6, n_attn=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.type_embedding = nn.Embedding(10, d_model)
        self.time_encoding = ContinuousTimeEncoding(d_model)
        total = n_mamba + n_attn
        attn_every = total // n_attn if n_attn > 0 else total + 1
        block_list = []
        self._block_types = []
        for i in range(total):
            if (i + 1) % attn_every == 0:
                block_list.append(TemporalAttentionBlock(d_model))
                self._block_types.append("attn")
            else:
                block_list.append(MambaBlock(d_model))
                self._block_types.append("mamba")
        self.blocks = nn.ModuleList(block_list)
        self.readmission_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.mortality_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, input_ids, token_types, timestamps, **kwargs):
        x = self.embedding(input_ids) + self.type_embedding(token_types) + self.time_encoding(timestamps)
        for btype, block in zip(self._block_types, self.blocks):
            if btype == "attn":
                x = block(x, timestamps)
            else:
                x = block(x)
        cls_repr = x[:, 0, :]
        return self.readmission_head(cls_repr).squeeze(-1), self.mortality_head(cls_repr).squeeze(-1)


# ── Training ─────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            r_logit, m_logit = model(**batch)
            loss = loss_fn(r_logit, batch["readmission"])
            if batch["mortality"].sum() > 0:
                loss = loss + loss_fn(m_logit, batch["mortality"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                r_logit, m_logit = model(**batch)
                loss = loss_fn(r_logit, batch["readmission"])
                val_loss += loss.item()

        avg_val = val_loss / max(len(val_loader), 1)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: train_loss={train_loss/len(train_loader):.4f}, val_loss={avg_val:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    all_r_logits, all_r_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            r_logit, _ = model(**batch)
            all_r_logits.append(r_logit.cpu())
            all_r_labels.append(batch["readmission"].cpu())
    preds = torch.sigmoid(torch.cat(all_r_logits)).numpy()
    labels = torch.cat(all_r_labels).numpy()
    if len(np.unique(labels)) > 1:
        return roc_auc_score(labels, preds), average_precision_score(labels, preds)
    return 0.5, 0.0


def main():
    print("=" * 70)
    print(" Phase 2a: Neural Architectures on Combined Dataset")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load and tokenize
    print("\n1. Loading combined data...")
    records = build_combined_records()

    print("\n2. Tokenizing patients...")
    tokenized, code_to_idx, lab_quantiles = tokenize_all_patients(records, max_tokens=512)
    print(f"   Tokenized patients: {len(tokenized)}")

    # Split (by patient)
    np.random.seed(42)
    np.random.shuffle(tokenized)
    n = len(tokenized)
    n_test = max(1, int(n * 0.15))
    n_val = max(1, int(n * 0.15))
    test_data = tokenized[:n_test]
    val_data = tokenized[n_test:n_test + n_val]
    train_data = tokenized[n_test + n_val:]

    print(f"   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_ds = EHRTokenDataset(train_data, max_len=512)
    val_ds = EHRTokenDataset(val_data, max_len=512)
    test_ds = EHRTokenDataset(test_data, max_len=512)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    all_results = {}

    # ── Experiment 2: EHRMamba ──
    print("\n3. Training EHRMamba (Mamba SSM)...")
    t0 = time.time()
    mamba_model = EHRMamba(d_model=128, n_layers=4).to(device)
    n_params = sum(p.numel() for p in mamba_model.parameters())
    print(f"   Parameters: {n_params:,}")
    mamba_model = train_model(mamba_model, train_loader, val_loader, device, epochs=30)
    auroc, auprc = evaluate_model(mamba_model, test_loader, device)
    elapsed = time.time() - t0
    print(f"   Readmission AUROC: {auroc:.4f}, AUPRC: {auprc:.4f} ({elapsed:.1f}s)")
    all_results["mamba"] = {"auroc": auroc, "auprc": auprc, "params": n_params, "time": elapsed}
    del mamba_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── Experiment 3: Continuous-Time ──
    print("\n4. Training Continuous-Time Model...")
    t0 = time.time()
    ct_model = ContinuousTimeEHR(d_model=128, n_layers=4).to(device)
    n_params = sum(p.numel() for p in ct_model.parameters())
    print(f"   Parameters: {n_params:,}")
    ct_model = train_model(ct_model, train_loader, val_loader, device, epochs=30)
    auroc, auprc = evaluate_model(ct_model, test_loader, device)
    elapsed = time.time() - t0
    print(f"   Readmission AUROC: {auroc:.4f}, AUPRC: {auprc:.4f} ({elapsed:.1f}s)")
    all_results["continuous_time"] = {"auroc": auroc, "auprc": auprc, "params": n_params, "time": elapsed}
    del ct_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── Experiment 5: Hybrid LHM ──
    print("\n5. Training Hybrid LHM (Mamba + Attention + Time)...")
    t0 = time.time()
    hybrid_model = HybridLHM(d_model=128, n_mamba=6, n_attn=2).to(device)
    n_params = sum(p.numel() for p in hybrid_model.parameters())
    print(f"   Parameters: {n_params:,}")
    hybrid_model = train_model(hybrid_model, train_loader, val_loader, device, epochs=30)
    auroc, auprc = evaluate_model(hybrid_model, test_loader, device)
    elapsed = time.time() - t0
    print(f"   Readmission AUROC: {auroc:.4f}, AUPRC: {auprc:.4f} ({elapsed:.1f}s)")
    all_results["hybrid"] = {"auroc": auroc, "auprc": auprc, "params": n_params, "time": elapsed}
    del hybrid_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── Save results ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "Phase 2a: Neural architectures (MIMIC + Synthea)",
        "n_patients": len(tokenized),
        "models": {
            name: {
                "readmission_auroc": f"{r['auroc']:.4f}",
                "readmission_auprc": f"{r['auprc']:.4f}",
                "params": r["params"],
                "training_time_s": f"{r['time']:.1f}",
            }
            for name, r in all_results.items()
        },
    }
    with open(OUTPUT_DIR / "neural_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print(" PHASE 2a RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f" {'Model':<25} {'AUROC':>10} {'AUPRC':>10} {'Params':>12} {'Time':>8}")
    print(f" {'-'*25} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
    for name, r in all_results.items():
        print(f" {name:<25} {r['auroc']:>10.4f} {r['auprc']:>10.4f} {r['params']:>12,} {r['time']:>7.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
