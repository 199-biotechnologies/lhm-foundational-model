# MedReason Dataset

- **File**: `medreason_32k.jsonl` (110 MB)
- **Source**: [UCSC-VLAA/MedReason](https://huggingface.co/datasets/UCSC-VLAA/MedReason)
- **Paper**: [arxiv:2504.00993](https://arxiv.org/abs/2504.00993) — "MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs"
- **License**: Apache 2.0
- **Award**: 3rd place HuggingFace Reasoning Datasets Competition (May 2025)

## Size & Structure

- **32,682 examples** (filtered from 45K generated — 71.4% retention)
- Columns: `dataset_name`, `id_in_dataset`, `question`, `answer`, `reasoning`, `options`

## Generation Pipeline

**Model**: GPT-4o (`gpt-4o-0806-nofilter-global`, API v2024-12-01-preview)
**Knowledge Graph**: PrimeKG (precision medicine KG)
**Total API cost**: ~$3,600

### 4-Stage Pipeline
1. **Entity Extraction & Mapping** — GPT-4o extracts medical entities → maps to PrimeKG nodes (exact match → similarity τ=0.85 → LLM contextual selection)
2. **Path Searching** — Shortest paths in PrimeKG connecting question entities → answer entities
3. **Path Pruning** — GPT-4o filters to K=3 most relevant paths per question
4. **CoT Generation** — GPT-4o generates step-by-step reasoning using pruned KG paths as scaffold

### Quality Control
- **Self-verification**: GPT-4o re-answers using ONLY the generated reasoning. Discarded if wrong answer.
- **Expert validation**: Licensed physicians across 7 specialties evaluated. Gastroenterology 100% preference over HuatuoGPT-o1.

## Source Distribution

| Source | Count | % |
|---|---|---|
| pubmedqa_artificial | 8,094 | 24.8% |
| medqa | 8,016 | 24.5% |
| huatuo | 6,475 | 19.8% |
| medmcqa | 6,197 | 19.0% |
| pubmedqa_unlabeled | 1,747 | 5.3% |
| MMLU | 827 | 2.5% |
| MedXpertQA | 666 | 2.0% |
| pubmedqa | 603 | 1.8% |
| LastHumanity | 57 | 0.2% |

## Reasoning Quality

- Mean length: 2,659 chars (median 2,681)
- Min: 916 chars — **zero short/empty reasoning**
- Max: 4,923 chars
- 100% contain "Reasoning Process" + "Conclusion" structure
- 81% use markdown headers (###)
- 20.8% include explicit "Finding reasoning paths" section

## Reasoning Format

```
### Finding Reasoning Paths:
1. [Entity A] -> [KG path] -> [Entity B]
2. [Entity C] -> [KG path] -> [Entity D]

### Reasoning Process:
[Multi-paragraph clinical reasoning grounded in KG paths]

### Conclusion:
[Final answer with justification]
```

## Benchmark Results (MedReason-8B)

| Benchmark | Accuracy |
|---|---|
| MedQA | 71.8% |
| MedMCQA | 60.7% |
| PubMedQA | 79.4% |
| MedBullets (op4) | 57.5% |
| MedBullets (op5) | 55.5% |

- DeepSeek-Distill-8B: +7.7% from MedReason training
- MedReason-8B outperforms HuatuoGPT-o1-8B by 4.2% on MedBullets
