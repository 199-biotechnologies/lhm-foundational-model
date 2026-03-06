# LHM — Foundational Model for Health Outcome Prediction

Architecture shootout: comparing Transformer, Mamba, Continuous-Time, and Hybrid architectures for predicting health outcomes from electronic health records.

## Goal

Proof of principle — find the best neural architecture for health trajectory prediction by running 6 experiments on MIMIC-IV data with Qwen3.5-0.8B as the LLM baseline.

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run experiments (see docs/plans/ for full design)
python -m src.training.run_experiment --exp 0  # XGBoost baseline
python -m src.training.run_experiment --exp 1  # Text-LLM (Qwen3.5-0.8B)
python -m src.training.run_experiment --exp 2  # EHRMamba
python -m src.training.run_experiment --exp 3  # Continuous-time
python -m src.training.run_experiment --exp 4  # Medical tokens
python -m src.training.run_experiment --exp 5  # Hybrid winner
```

## Design

See [docs/plans/2026-03-06-lhm-architecture-shootout-design.md](docs/plans/2026-03-06-lhm-architecture-shootout-design.md) for the full architecture thesis and experiment design.

## Research

See [research/](research/) for background research on the health foundation model landscape.

## License

Private — 199 Biotechnologies
