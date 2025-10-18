# TeleDerm SnapCheck Publication Checklist

## Working Title
"TeleDerm SnapCheck: Calibrated Image-Quality Gating Improves Vision-Language Triage Safety"

## Target Venues
- *JMIR Dermatology* (Rapid Communication)
- *npj Digital Medicine* (Brief Communication)
- *Telemedicine and e-Health* (Research Article)

## Core Contributions
1. **Synthetic DIQA Dataset** – Programmatic generation of dermatology-specific image-quality defects paired with pristine references.
2. **Compact Quality Classifier** – ViT-S/16 model achieving macro AUROC 0.968 / macro AP 0.890 with fairness slices across Fitzpatrick-like bins.
3. **Calibrated Quality Gate** – Offline threshold tuning toolkit (`scripts/offline_gate_analysis.py`) delivering urgent recall 73.2% -> 75.9% with a 18.9% retake rate on 4,800 triage images.
4. **Triage Simulation & Fairness Analysis** – Head-to-head comparison of baseline vs gated Qwen2-VL-2B triage including defect, diagnosis, and skin-tone breakdowns.
5. **Open Toolkit** – Reproducible scripts, cached prompts, and manuscript-ready tables/figures.

## Figures & Tables
- **Figure 1 (ready):** Workflow schematic (`reports/figures/telederm_snapcheck_workflow_v3.png`).
- **Figure 2 (planned):** Defect impact on accuracy/retake rate (bar chart).
- **Figure 3 (planned):** Urgent recall vs retake trade-off across threshold sweeps.
- **Table 1 (ready):** Triage performance summary (`reports/triage/triage_summary.csv`).
- **Table 2 (ready):** Fitzpatrick & diagnosis breakdowns (`reports/triage/breakdowns/triage_publication_tables.md`).
- **Table 3 (planned):** DIQA model metrics + fairness slices.

## Timeline (updated)
| Week | Milestone | Status |
|------|-----------|--------|
| 0 | Data download, augmentation sanity checks | ✅ |
| 1 | Train DIQA model, evaluate fairness | ✅ |
| 2 | Run triage simulations & threshold calibration | ✅ |
| 3 | Draft manuscript, integrate calibrated metrics | 🔄 (in progress) |
| 4 | External review, polish figures/tables, submission | ⏳ |

## Writing Outline
1. **Introduction** – Telederm access gap, image-quality failure modes, motivation for automated gating.
2. **Methods** – Dataset curation, augmentation, DIQA training, offline calibration workflow, VLM triage setup, metrics/fairness definitions.
3. **Results** – DIQA performance, threshold calibration, triage outcomes (overall + slices), defect impact analysis.
4. **Discussion** – Clinical implications, retake burden, limitations (synthetic defects, paired pass assumption), future directions.
5. **Conclusion** – SnapCheck viability and path toward clinical deployment.

## Reproducibility Checklist
- [x] Config-driven experiments (`configs/*.yaml`).
- [x] Seed logging and cached prompts (see `reports/triage/cache/`).
- [x] Threshold calibration script with deterministic replay.
- [ ] Model/export checksums (pending addition to `models/` README).
- [ ] Publish augmentation manifest metadata bundle.
- [ ] Validation notebook reproducing key plots.

## IRB & Ethics Notes
- Uses de-identified public datasets only; cite HAM10000 and Derm7pt licenses.
- Synthetic defects supplement but do not replace expert review; discuss in limitations.
- Highlight fairness monitoring across skin-tone bins.

## Next Actions
- [ ] Add checksums + version info for quality model and triage outputs.
- [ ] Produce planned figures (defect impact, threshold trade-off).
- [ ] Draft remaining manuscript sections using updated metrics.
- [ ] Conduct qualitative review of retake prompts with clinical advisors.
- [ ] Prepare supplementary materials (augmentation metadata, config archive).
