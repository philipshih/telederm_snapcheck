# Methods Outline for TeleDerm SnapCheck Manuscript

## Study Design
- Retrospective benchmarking with synthetic quality labels derived from public datasets.
- Evaluation of teledermatology triage pipelines under baseline vs quality-gated regimes.

## Datasets
- HAM10000 and Derm7pt (public) with manifest linking to diagnosis and metadata.
- Synthetic quality dataset generated via `scripts/build_quality_dataset.py`.

## DIQA Model
- Backbone: ViT-S/16 (timm) fine-tuned on synthetic dataset.
- Objective: Multi-label BCE with logits; label smoothing 0.05.
- Training schedule: 25 epochs, AdamW (lr 5e-5), cosine annealing.
- Metrics: Macro AUROC, average precision, calibration error.
- Fairness: AUROC/Recall across skin tone bins and capture channels.
- Exports: Torchscript, ONNX, optional TFLite.

## Triage Simulation
- Baseline: Direct VLM diagnosis -> triage mapping.
- Quality-gated: Assess with SnapCheck; fail -> retake, pass -> VLM.
- VLMs: Configurable via YAML (e.g., `mlfoundations/llava-med-7b`, `gpt-4o-mini`).
- Metrics: Triage accuracy, urgent recall, urgency miss rate, safe reassurance, mean latency, token usage, retake rate.
- Fairness: Stratify metrics by skin tone bin, capture channel.

## Analysis Workflow
1. Download & stage datasets: `scripts/download_public_datasets.py`.
2. Build quality dataset: `scripts/build_quality_dataset.py`.
3. Train DIQA model: `scripts/train_quality_model.py`.
4. Evaluate & export fairness: `scripts/evaluate_quality_model.py`.
5. Run triage simulations: `scripts/run_triage_simulation.py`.
6. Summaries & figures: `scripts/summarize_results.py`, `notebooks/eda_quality.ipynb` (placeholder).
7. Optional Streamlit demo: `scripts/app_streamlit.py`.

## Reproducibility
- Config-driven experiments via YAML.
- Seeds logged in manifests.
- Reports saved under `reports/` with metrics, fairness tables, confusion matrices.

## Limitations
- Synthetic quality labels approximate but do not replace expert annotations.
- Skin tone estimation via brightness proxy may misclassify deeper pigments.
- VLM responses depend on external APIs; caching recommended.
- Retake logic does not yet integrate multi-image submissions.

