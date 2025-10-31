<p align="center">
  <img src="snapcheck/logo.jpg" alt="TeleDerm SnapCheck logo" width="200">
</p>

# TeleDerm SnapCheck
© 2025 Philip Shih. Released under the MIT License. 

Unless stated otherwise:
- **Ownership**: All original code in this repo is owned by Philip Shih.
- **Third-party content**: This project may reference or include third-party code or assets that are subject to their own licenses.
- **Contributions**: By submitting a contribution, you agree it’s your own work (or you have the right to submit it) and you license it under the repository’s license.

TeleDerm SnapCheck explores how automated dermatologic image quality assessment (DIQA) impacts downstream teledermatology triage safety. We apply synthetic quality defects to public dermoscopy and teledermatology datasets, then assess VLM performance with/without a calibrated ViT DIQA gate. This is the first study to demonstrate successful use of a DIQA gate for VLMs and assess how image defects impact VLM diagnostic performance.

The ViT image quality gate reduced urgent miss rate on public datasets by 69% (26.3%→8.2%) when set to reject 15% of images based on their degree of quality. DIQA gating improved sensitivity for darker skin tones (Fitz VI +6.3 pts; MST 8–10 +5.4 pts) without increasing retake burden. Of all defect types, motion blur resulted in the highest retake burden. This may suggest interventions against camera movement during capture are most important in store-and-forward teledermatology triage workflows that use VLMs.

## Getting Started

1. Create a virtual environment and install requirements.
2. Provide datasets in `/data` or run `scripts/download_public_datasets.py` to fetch ISIC/HAM10000 metadata.
3. Launch the synthetic image augmentation pipeline: `python scripts/build_quality_dataset.py --config configs/augmentation.yaml`.
4. Train the ViT classifier: `python scripts/train_quality_model.py --config configs/train_diqa.yaml`.
5. Run the triage simulation: `python scripts/run_triage_simulation.py --config configs/triage_eval.yaml`.
6. Regenerate tables/figures: `python scripts/analyze_triage_breakdowns.py` followed by `scripts/format_triage_publication_tables.py`.

Milestones and writing tasks tracked in `docs/publication_plan.md`.

## In Progress

- Validation on external teledermatology cohorts
- VLM model performance comparisons
- Package reproducibility artifacts
