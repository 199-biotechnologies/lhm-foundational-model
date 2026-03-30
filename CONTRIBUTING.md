# Contributing to LHM

Thanks for your interest. Here is how to contribute effectively.

## Getting Started

1. Fork the repo and clone your fork
2. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Create a branch: `git checkout -b your-feature`

## What We Need

- **Clinical prediction tasks** — New evaluation metrics and prediction targets beyond readmission and mortality
- **Multimodal integration** — Wearable data, genomics, imaging pipelines
- **Scaling experiments** — Benchmarks on larger patient cohorts (full MIMIC-IV, eICU, etc.)
- **Architecture variants** — New temporal encoding strategies, attention mechanisms, or SSM configurations
- **PRISM skill packs** — Additional clinical reasoning examples with fact-checked references
- **Documentation** — Clearer explanations of the training pipeline and architecture decisions

## Pull Request Process

1. Run existing experiments to verify nothing breaks: `python experiments/compare_all.py`
2. If you add a new experiment, follow the existing structure in `experiments/`
3. Include results in your PR description
4. Keep PRs focused. One feature or fix per PR.

## Code Style

- Python 3.12+
- Type hints where practical
- Docstrings for public functions
- No unused imports or dead code

## Experiment Guidelines

Each experiment lives in its own directory under `experiments/` with:
- `run.py` — Main training and evaluation script
- `config.yaml` — Hyperparameters and settings
- `outputs/` — Results directory (gitignored for large files)

Use the same evaluation tasks and test splits as existing experiments so results are comparable.

## Reporting Issues

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version, OS, GPU (if relevant)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
