# TeleDerm SnapCheck Publication Checklist

## Working Title
"TeleDerm SnapCheck: Rapid Image-Quality Gating Improves Vision-Language Triage Safety in Teledermatology"

## Target Venues
- *JMIR Dermatology* (Rapid Communication)
- *npj Digital Medicine* (Brief Communication)
- *Telemedicine and e-Health* (Letters)

## Core Contributions
1. **Synthetic DIQA Dataset** – Programmatic generation of labeled dermatology image-quality defects built from public datasets.
2. **Compact Quality Classifier** – ViT-S model trained with auto-labels, exported to ONNX/TFLite for point-of-care deployment.
3. **Triage Simulation Harness** – Head-to-head comparison of baseline vs quality-gated VLM triage decisions across fairness slices.
4. **Fairness & Efficiency Metrics** – Skin tone-stratified AUROC/recall, latency, and token usage deltas.
5. **Open Toolkit** – Reproducible scripts, configs, and Streamlit demo.

## Figures & Tables
- Fig 1: Workflow schematic.
- Fig 2: ROC curves per defect + calibration plot.
- Fig 3: Triage accuracy vs urgent miss rate (bar + line) baseline vs gated.
- Fig 4: Fairness violin plots (AUROC by Fitzpatrick bin).
- Table 1: Dataset composition and augmentation statistics.
- Table 2: Triage safety metrics across models.
- Table 3: Latency and token savings summary.

## Timeline (aggressive)
| Week | Milestone |
|------|-----------|
| 0 | Data download, augmentation sanity checks |
| 1 | Train DIQA model, run ablations |
| 2 | Triage simulations, fairness analysis |
| 3 | Draft manuscript + figures |
| 4 | Internal review, submission |

## Writing Outline
1. **Introduction** – Telederm access gap, image quality bottlenecks, lack of automated gating evidence.
2. **Methods** – Dataset curation, augmentation strategy, quality labels, model training, VLM triage setup, metrics, fairness algorithms.
3. **Results** – Quality classifier performance, fairness slices, triage baseline vs gated, latency/token analysis.
4. **Discussion** – Clinical implications, limitations (synthetic labels, dataset bias), future work (prospective validation, integration with patient apps).
5. **Conclusion** – SnapCheck viability.

## Reproducibility Checklist
- [ ] Seed logging (configurable).
- [ ] Deterministic dataloaders or documentation of nondeterminism.
- [ ] Model/export checksums.
- [ ] Publish metadata manifest with augmentation parameters.
- [ ] Provide inference notebook replicating key plots.
- [ ] Document API usage requirements (keys, costs, caching).

## IRB & Ethics Notes
- Uses de-identified public datasets only.
- Provide citation + licensing compliance in appendix.
- Include statement on synthetic augmentations and limitations.

## Next Actions
- [ ] Confirm dataset availability + licensing.
- [ ] Implement augmentation pipeline (scripts/build_quality_dataset.py).
- [ ] Implement training/eval (scripts/train_quality_model.py, scripts/evaluate_quality_model.py).
- [ ] Implement triage simulation (scripts/run_triage_simulation.py).
- [ ] Add Streamlit demo (scripts/app_streamlit.py).
- [ ] Draft manuscript using `docs/manuscript_template.md` (to be created after initial results).

