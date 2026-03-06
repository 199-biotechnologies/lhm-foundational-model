"""
Experiment 4: Medical Token Decoder (ETHOS/CoMET style).

Tests whether purpose-built medical tokenization + a small decoder-only
transformer beats text-based LLMs and generic architectures.

Architecture: Medical tokenizer → small decoder transformer with RoPE →
next-event prediction + classification heads.

Key difference from Exp 2/3: this model does autoregressive next-token
prediction on medical tokens (not just classification), capturing the
generative modeling aspect of health trajectories.
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
    VOCAB_SIZE, PAD_TOKEN, CLS_TOKEN, VISIT_START, VISIT_END,
)
from src.evaluation.metrics import (
    ExperimentResults, compute_binary_metrics, print_comparison_table,
)


OUTPUT_DIR = Path(__file__).parent / "outputs"


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """RoPE for position-aware attention without absolute position embeddings."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Decoder Transformer Block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Decoder block with causal self-attention + RoPE."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm1 = nn.RMSNorm(d_model)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult, bias=False),
            nn.SiLU(),
            nn.Linear(d_model * ff_mult, d_model, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin, mask=None):
        B, L, D = x.shape
        H, DH = self.n_heads, self.d_head

        # Self-attention with RoPE
        residual = x
        x_norm = self.norm1(x)

        q = self.wq(x_norm).view(B, L, H, DH).transpose(1, 2)
        k = self.wk(x_norm).view(B, L, H, DH).transpose(1, 2)
        v = self.wv(x_norm).view(B, L, H, DH).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        attn = (q @ k.transpose(-2, -1)) * (DH ** -0.5)

        # Causal mask
        causal = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        if mask is not None:
            pad_mask = (1 - mask).bool().unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = residual + self.wo(out)

        # FFN
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Medical Token Decoder Model
# ---------------------------------------------------------------------------

class MedicalTokenDecoder(nn.Module):
    """
    Decoder-only transformer on medical token sequences.
    Combines next-token prediction (generative) with classification heads.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_token_types: int = 5,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.type_embedding = nn.Embedding(n_token_types, d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.RMSNorm(d_model)

        # Generative head: predict next medical token
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Classification heads
        self.readmission_head = nn.Linear(d_model, 1)
        self.mortality_head = nn.Linear(d_model, 1)

    def forward(self, token_ids, token_types, attention_mask=None):
        B, L = token_ids.shape
        x = self.token_embedding(token_ids) + self.type_embedding(token_types)

        cos, sin = self.rope(L)
        for block in self.blocks:
            x = block(x, cos, sin, attention_mask)

        x = self.norm(x)

        # Next-token prediction logits
        lm_logits = self.lm_head(x)

        # Classification from CLS position
        cls_repr = x[:, 0, :]

        return {
            "lm_logits": lm_logits,
            "readmission_logit": self.readmission_head(cls_repr).squeeze(-1),
            "mortality_logit": self.mortality_head(cls_repr).squeeze(-1),
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MedicalTokenDataset(Dataset):
    """Dataset for medical token sequences with next-token prediction."""

    def __init__(self, patients: list[TokenizedPatient], max_len: int = 256):
        self.patients = patients
        self.max_len = max_len

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        tokens = p.token_ids[:self.max_len]
        types = p.token_types[:self.max_len]
        seq_len = len(tokens)

        # Next-token prediction labels (shifted by 1)
        labels = tokens[1:] + [PAD_TOKEN]

        pad_len = self.max_len - seq_len
        tokens = tokens + [PAD_TOKEN] * pad_len
        types = types + [0] * pad_len
        labels = labels + [PAD_TOKEN] * pad_len
        mask = [1.0] * seq_len + [0.0] * pad_len

        return {
            "token_ids": torch.tensor(tokens, dtype=torch.long),
            "token_types": torch.tensor(types, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.float32),
            "readmission": torch.tensor(p.readmission_30d, dtype=torch.float32),
            "mortality": torch.tensor(p.mortality, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, device, lm_weight=0.5):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        token_ids = batch["token_ids"].to(device)
        token_types = batch["token_types"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)
        readmission = batch["readmission"].to(device)
        mortality = batch["mortality"].to(device)

        outputs = model(token_ids, token_types, mask)

        # Next-token prediction loss (ignore PAD)
        lm_loss = F.cross_entropy(
            outputs["lm_logits"].view(-1, VOCAB_SIZE),
            labels.view(-1),
            ignore_index=PAD_TOKEN,
        )

        # Classification losses
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
    all_readm_logits = []
    all_mort_logits = []
    all_readm_true = []
    all_mort_true = []
    total_loss = 0
    total_lm_loss = 0
    n_batches = 0

    for batch in dataloader:
        token_ids = batch["token_ids"].to(device)
        token_types = batch["token_types"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)
        readmission = batch["readmission"].to(device)
        mortality = batch["mortality"].to(device)

        outputs = model(token_ids, token_types, mask)

        lm_loss = F.cross_entropy(
            outputs["lm_logits"].view(-1, VOCAB_SIZE),
            labels.view(-1),
            ignore_index=PAD_TOKEN,
        )
        loss_r = F.binary_cross_entropy_with_logits(outputs["readmission_logit"], readmission)
        loss_m = F.binary_cross_entropy_with_logits(outputs["mortality_logit"], mortality)

        total_loss += (loss_r + loss_m).item()
        total_lm_loss += lm_loss.item()
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
        "lm_loss": total_lm_loss / max(n_batches, 1),
        "readmission": readm_metrics,
        "mortality": mort_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 60)
    print("EXPERIMENT 4: Medical Token Decoder")
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

    # Load and tokenize
    print("\n1. Loading and tokenizing MIMIC-IV data...")
    records = build_patient_records()
    tokenized, code_to_idx, lab_quantiles = tokenize_all_patients(records, max_tokens=512)
    print(f"   Tokenized patients: {len(tokenized)}")
    print(f"   Medical vocab: {len(code_to_idx)} diagnosis codes + labs + special")

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
    train_dataset = MedicalTokenDataset(train_data, max_len=max_len)
    val_dataset = MedicalTokenDataset(val_data, max_len=max_len)
    test_dataset = MedicalTokenDataset(test_data, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Model
    print("\n2. Building Medical Token Decoder...")
    model = MedicalTokenDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        n_layers=4,
        n_heads=4,
        max_seq_len=max_len,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")

    # Training
    print("\n3. Training (next-token + classification)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(20):
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
    print(f"   LM loss (next-token): {test_results['lm_loss']:.4f}")

    # Results
    results = ExperimentResults(
        experiment_name="Medical Token Decoder",
        readmission_auroc=test_results["readmission"]["auroc"],
        readmission_auprc=test_results["readmission"]["auprc"],
        mortality_auroc=test_results["mortality"]["auroc"],
        mortality_auprc=test_results["mortality"]["auprc"],
        training_time_seconds=train_time,
        model_params=n_params,
        notes=f"d_model=128, n_layers=4, RoPE, next-token+cls, LM loss={test_results['lm_loss']:.4f}",
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
