# TeleDerm SnapCheck

TeleDerm SnapCheck explores how automated dermatology image quality gating affects downstream teledermatology triage safety. Synthetic quality defects are applied to public datasets, and VLM performance on urgent triage is compared with and without the use of a ViT. 

## Getting Started

1. Create a virtual environment and install requirements.
2. Provide dataset in /data or use `scripts/download_public_datasets.py` to grab ISIC/HAM10000 metadata.
3. Launch synthetic augmentation pipeline: `python scripts/build_quality_dataset.py --config configs/augmentation.yaml`.
4. Train ViT quality classifier: `python scripts/train_quality_model.py --config configs/train_diqa.yaml`.
5. Run: `python scripts/run_triage_simulation.py --config configs/triage_eval.yaml`.
6. Generate figures/tables via `scripts/summarize_results.py`.

Current progress on SnapCheck in `docs/publication_plan.md`.
