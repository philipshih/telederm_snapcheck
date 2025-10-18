# TeleDerm SnapCheck

TeleDerm SnapCheck is a lightweight, fully-scripted pipeline that measures how automated dermatology image quality gating affects downstream teledermatology triage safety. The workflow augments public skin-lesion datasets with synthetic quality defects, trains a compact quality classifier, and compares baseline versus gated triage outputs from vision-language models. This repository is designed for rapid, publication-ready experimentation without manual labeling.

## Repo Layout

- `configs/` – YAML configs for data prep, model training, and triage experiments.
- `data/` – Scripts expect raw datasets or symlinks here; subfolders created during preprocessing.
- `docs/` – Manuscript outline, dataset cards, and supplementary text.
- `models/` – Saved weights, ONNX/TFLite exports, and metrics.
- `notebooks/` – Optional exploratory notebooks (EDA, visualization).
- `reports/` – Auto-generated figures, tables, and final evaluation summaries.
- `scripts/` – Python entrypoints for augmentation, training, evaluation, and dashboard.
- `tests/` – Lightweight unit tests for core utilities.

## Getting Started

1. Create a virtual environment and install requirements.
2. Run `scripts/download_public_datasets.py` to grab ISIC/HAM10000 metadata (or manually place data).
3. Launch the augmentation pipeline: `python scripts/build_quality_dataset.py --config configs/augmentation.yaml`.
4. Train the quality classifier: `python scripts/train_quality_model.py --config configs/train_diqa.yaml`.
5. Evaluate triage safety with and without gating: `python scripts/run_triage_simulation.py --config configs/triage_eval.yaml`.
6. Generate manuscript-ready figures/tables via `scripts/summarize_results.py`.

A detailed quick-start and publication checklist live in `docs/publication_plan.md`.

