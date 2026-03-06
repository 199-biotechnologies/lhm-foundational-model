# LHM Dataset Strategy

## Files

- `candidate_datasets.csv` — Master list of 28+ datasets with access level, size, modalities, and phase assignment
- `synthetic_data_strategies.md` — Approaches for synthetic EHR generation (LLM distillation, GANs, DualAlign)

## Scaling Phases

| Phase | Datasets | Total Scale | Key Unlock |
|-------|----------|-------------|------------|
| 2a (immediate) | NHANES, PhysioNet challenges, mPower, OAI, OASIS-4, Medicaid Open Data | ~200K records | No application needed |
| 2b (registered) | All of Us, FinnGen, ADNI, MESA, Bridge2AI, Human Phenotype Project | ~1M records | Simple application |
| 2c (credentialed) | Full MIMIC-IV, eICU, UK Biobank, TOPMed | ~1M+ patients | CITI training + DUA |
| 3 (multi-modal) | UK Biobank + All of Us + Bridge2AI combined | 500K+ multi-modal | Architecture ready for fusion |
