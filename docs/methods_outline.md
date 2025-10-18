# Methods Outline for TeleDerm SnapCheck Manuscript

## Study Design
- Retrospective benchmarking with synthetic quality labels derived from public datasets.
- Evaluation of teledermatology triage pipelines under baseline vs calibrated quality-gated regimes.

## Datasets
- HAM10000 and Derm7pt (public) with manifests linking to diagnosis, capture metadata, and augmentation parameters.
- Synthetic quality dataset generated via `scripts/build_quality_dataset.py` (pass + degraded pairs, per-defect metadata, ITA-derived Fitzpatrick + Monk skin tone fields).

## DIQA Model
- Backbone: ViT-S/16 (timm) fine-tuned on synthetic dataset.
- Objective: Multi-label BCE with logits; label smoothing 0.05.
- Training schedule: 25 epochs, AdamW (lr 5e-5), cosine annealing, batch size 64.
- Metrics: Macro AUROC 0.968, macro AP 0.890, per-defect AUROC/AP.
- Fairness: AUROC/Recall across ITA-mapped Fitzpatrick bins (`fitzpatrick_type`) and Monk groupings.
- Exports: Torch checkpoint (`models/snapcheck_quality.pt`) with optional ONNX export.

## Threshold Calibration
- Offline replay of cached quality scores + VLM predictions via `scripts/offline_gate_analysis.py` on the held-out validation split.
- Sweep global and per-defect thresholds to balance urgent recall vs retake burden (target ~20% retakes, urgent recall +2-3 pp).
- Lock thresholds before scoring the untouched test split.
- Output artifacts: `reports/triage/offline_gate_summary.csv`, calibrated `reports/diqa/thresholds.json`, updated triage detail (`reports/triage/triage_detail_calibrated.csv`).

## Triage Simulation
- Baseline: Direct Qwen2-VL-2B diagnosis mapped to triage labels via `configs/triage_eval.yaml`.
- Quality-gated: Apply calibrated thresholds; failures swap in paired pass images, successes reuse baseline prediction.
- Metrics: Triage accuracy, urgent recall, urgency miss rate, urgent deferral rate, retake rate, mean latency, mean token usage.
- Fairness: Stratify metrics by skin tone bin, diagnosis, and defect presence.

## Analysis Workflow
1. Download & stage datasets: `scripts/download_public_datasets.py`.
2. Build quality dataset: `scripts/build_quality_dataset.py`.
3. Train DIQA model: `scripts/train_quality_model.py`.
4. Evaluate & export fairness: `scripts/evaluate_quality_model.py` (optional deep dive).
5. Run triage simulations: `scripts/run_triage_simulation.py`.
6. Calibrate thresholds offline: `scripts/offline_gate_analysis.py`.
7. Regenerate summaries & figures: `scripts/analyze_triage_breakdowns.py`, `scripts/format_triage_publication_tables.py`, `scripts/generate_workflow_figure_v2.py`.
8. Optional Streamlit demo: `scripts/app_streamlit.py`.

## Reproducibility
- Config-driven experiments via YAML; seeds stored in configs and manifests.
- Prompt caching for VLM calls enables deterministic replays.
- Reports saved under `reports/` with metrics, fairness tables, and calibration summaries.

## Limitations
- Synthetic quality labels approximate but do not replace expert annotations.
- ITA-derived skin tone remains a proxy; comparisons to MSKCC/clinician labels are planned.
- Retake logic assumes availability of paired pass images; real patient retakes may differ.
- Current evaluation uses a single VLM (Qwen2-VL-2B); future work should benchmark alternatives.
