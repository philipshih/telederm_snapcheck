# TeleDerm SnapCheck: DIQA Gating for Safer Teledermatology Triage
Philip E Shih, M.S.<sup>1</sup>

<sup>1</sup> School of Medicine, University of California San Francisco. San Francisco, California, USA

## Abstract

### Background
Teledermatology programs rely on patient-submitted photos that frequently suffer from blur, shadowing, or framing errors, undermining downstream clinical decision support. Manual screening is labor-intensive and inconsistently applied.

### Objective
We test whether an automated dermatologic image quality assessment (DIQA) gate can lower urgent-miss risk for vision-language triage models (VLMs) while maintaining acceptable retake burden.

### Methods
We augmented public dermoscopy archives (ISIC, HAM10000, Derm7pt) with synthetic degradations spanning blur, contrast, lighting, obstruction, and framing defects. A ViT-Small multi-label classifier was trained on paired pass and fail images (BCE-with-logits loss, cosine LR schedule). Thresholds were recalibrated offline on 671 validation pass/fail pairs using cached GPT-5 Nano predictions, then frozen and evaluated once on 672 pass/fail lesion pairs (1,344 exposures) in the held-out test manifest. The calibrated gate is integrated into a scripted GPT-5 Nano triage harness that compares baseline versus gated reasoning over identical image subsets. Primary endpoints were urgent recall, urgency miss rate, and retake or deferral burden; secondary endpoints included fairness slices and defect-specific performance.

### Results
The quality model maintains macro AUROC 0.968 (macro average precision 0.890) with Fitzpatrick-like bins spanning AUROC 0.957 to 0.972. Enabling SnapCheck preserved diagnostic accuracy (40.4% to 40.1%) and nudged urgent recall to 76.6%. The most notable effect was the decrease in urgent miss rate to 8.2% by deferring 15.3% of urgent exposures for retake. The gate requested retakes on 195 of 1,344 encounters (14.5%), covering 195 of 672 degraded images (29.0%), and routed 54 urgent lesions into the deferral queue. Blur exposures gained +20.7 urgent-recall points (62.1% to 82.8%), motion-blur gained +5.9 points despite 92.1% of urgent cases being deferred, and noise defects improved +9.6 points. Fitzpatrick Type VI lesions improved from 77.5% to 83.8% urgent recall and Monk Skin Tone (MST) 8–10 increased from 70.7% to 76.1% with comparable retake burden (<19%), while lighter MST 1–3 tones stayed within 3.3 percentage points of baseline recall.

### Conclusions
DIQA gating for dermoscopic images materially improves urgent sensitivity once thresholds are calibrated to the observed defect distribution. The safety gain is coupled with a 14.5% retake burden and 15.3% urgent deferrals. Prospective studies should validate these findings with real patient retakes and explore adaptive policies tailoring retake prompts to high-impact defect types.

### Keywords
teledermatology, dermatology image quality, vision-language models, fairness, clinical decision support

## Introduction
Teledermatology has expanded rapidly under access pressures, but poor patient-generated imagery remains a significant limitation to utility. Clinicians report losing up to 50% of asynchronous cases to unreadable photos, and remote workflows offer limited opportunities for real-time coaching. Emerging vision-language models (VLMs) promise rapid triage, yet their sensitivity is dependent on image quality. Prior work has explored generic blind image quality assessment, but the downstream safety impact of DIQA gating remains has not yet been studied.

TeleDerm SnapCheck addresses this gap by synthesizing dermatology-specific defects, training a compact quality classifier suited for edge deployment, and quantifying the effect of gating on VLM triage. SnapCheck is fully scripted to ensure reproducibility and portability.

## Methods

### Data Sources and Augmentation
We curated public dermoscopy datasets (ISIC 2020, HAM10000, Derm7pt) and generated paired pass and fail crops. The augmentation engine injects blur, motion blur, exposure shifts, contrast shifts, shadow occlusion, obstructions, cropping or framing errors, and resolution downscaling. Metadata tracks augmentation strengths plus ITA-derived Fitzpatrick (0–5) and Monk Skin Tone (1–10) bins alongside capture channel (clinic versus patient-generated).

### Urgent Triage Label Derivation
Source datasets provide histopathology-confirmed diagnoses or expert adjudications. We mapped melanoma (including melanoma in situ), basal cell carcinoma (basal cell carcinoma and its “bcc” synonym), and squamous cell carcinoma labels to the “urgent” triage category. Nevi, benign keratoses, vascular lesions, and related benign entities were mapped to the “routine” category, while lentiginous or normal-skin labels were mapped to “reassurance.” Ambiguous or missing diagnoses were excluded from the urgent denominator. This deterministic mapping enables transparent reconstruction of the triage ground truth.

### Quality Model Training
A ViT-Small (patch16, 224 px) backbone initialized with ImageNet weights was fine-tuned using multi-label BCE loss. Training used batch size 64, AdamW (learning rate 5e-5, weight decay 0.02), cosine annealing over 25 epochs, and standard color jitter or horizontal flip augmentation. Manifests were split 70/15/15 with stratification on the overall fail label. Evaluation produced AUROC and average precision per quality defect and fairness slices across skin-tone bins and capture channels. The best checkpoint is saved as `models/snapcheck_quality.pt`.

### Thresholding and Gating
Initial per-defect probability thresholds were selected from validation curves and stored alongside the quality model configuration. We replayed cached quality scores and GPT-5 Nano predictions on the 671 validation pass/fail pairs, sweeping threshold candidates until the gate blocked roughly 25–30% of degraded images while keeping the projected retake rate below 15%. The final operating point sets blur and low-resolution thresholds to 0.966, motion blur and obstruction to 0.8925, and leaves exposure, contrast, noise, shadow, framing, and the overall fail trigger at 1.0. Images exceeding any calibrated threshold trigger a retake, which re-runs the VLM on the paired pass image. Otherwise, the original prediction is preserved.

### Evaluation Cohort and Leakage Controls
The synthetic evaluation cohort comprises 4,800 dermoscopy images: 2,400 pristine captures and 2,400 quality-deficient counterparts sampled after removing duplicate image identifiers and ambiguous diagnoses. Although the public archives contain roughly 10,000 unique lesions, we restricted the study to cases with histopathology-confirmed diagnoses and clear urgent mappings, then generated pass/fail pairs to cap inference and caching costs. Each fail image retains a pointer to its pristine partner. During evaluation, the gate replaces the degraded frame with its paired pass image for retake simulation, while baseline predictions continue to use the original fail image. Calibration relied solely on the 671-pair validation split, and all summary metrics reported below come from a single replay on the held-out 672 pass/fail lesion pairs (1,344 exposures). Prospective clinical data will be required to confirm performance on real-world submissions.

### Triage Simulation Harness
We configured the GPT-5 Nano vision-language model via the OpenAI API, generating dermatologic diagnoses mapping to predetermined triage labels (reassurance, routine, urgent). The simulator runs matched baseline and quality-gated conditions over identical image subsets. Metrics include triage accuracy, urgent recall, urgency miss rate, urgent deferral rate, and retake rate, latency and token usage.

### AI System Transparency
All triage calls used the GPT-5 Nano Responses API (release 2025-08-07) with deterministic decoding (temperature 0.0, top-p 1.0, reasoning effort “low”) and the single-turn prompt defined in `configs/prompts/diagnosis_singleline.txt`. Safety filters and content moderation flags remained enabled. Full prompts, configuration files, and cached outputs are deposited in the project repository for reproducibility.

### Statistical Analysis
We computed nonparametric 95% confidence intervals via 2,000 bootstrap resamples for accuracy, urgent recall, retake rate, and urgent deferral rate. Paired differences in accuracy and urgent recall between baseline and gated modes were assessed with McNemar tests. These results should be interpreted cautiously because the test set is synthetic and relatively small. Recognising that illumination-based bins are an imperfect proxy for skin tone, we flagged fairness gaps greater than 0.05 absolute percentage points for follow-up calibration.

## Results

### Overall Triage Performance

| Mode | Accuracy (%) | Urgent recall (%) | Urgent deferral (%) | Retake rate (%) | Mean latency (s) | Mean tokens |
|------|--------------|-------------------|---------------------|-----------------|------------------|-------------|
| Baseline (ungated) | 40.4 | 73.7 | 0.0 | 0.0 | 2.01 | 349.4 |
| Quality-gated | 40.1 | 76.6 | 15.3 | 14.5 | 2.01 | 349.4 |

Table 1. Aggregate triage metrics for 1,344 evaluation exposures (672 lesion pairs; 354 urgent exposures across 177 urgent lesions).

Bootstrapped 95% confidence intervals (2,000 resamples) showed that accuracy shifted from 40.4% (95% CI 37.8–43.1) in the ungated arm to 40.1% (95% CI 37.5–42.7) with SnapCheck. Urgent recall increased from 73.7% (95% CI 68.9–78.0) to 76.6% (95% CI 71.9–80.7), while the urgent miss rate fell from 26.3% to 8.2% (−69% relative). The gate recommended retakes for 14.5% of encounters (95% CI 12.7–16.5), deferred 15.3% of urgent exposures for retake follow-up (95% CI 11.9–19.4), and flagged 195 of 672 degraded images (29.0% coverage). Figure 2 visualises the trade-off introduced by SnapCheck.

![Figure 2. SnapCheck halves urgent misses while introducing a managed retake/deferral queue.](reports/figures/figure2_overall_gains.png)
*Figure 2. Comparison of baseline GPT-5 Nano triage versus SnapCheck-gated triage on the 1,344-exposure paired test cohort. Bars show the percentage of urgent misses, urgent recall, urgent deferrals, and retake recommendations.*

### Defect-Specific Impact

| Synthetic Defect | Total Cases | Urgent Cases | Accuracy (Baseline) | Accuracy (Gated) | Urgent Recall (Baseline) | Urgent Recall (Gated) | Urgent Deferral (Gated) | Retake Rate (Gated) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall_fail | 672 | 177 | 42.0% | 40.8% | 68.4% | 75.7% | 38.4% | 38.5% |
| blur | 127 | 29 | 40.9% | 41.7% | 62.1% | 82.8% | 86.2% | 78.7% |
| motion_blur | 127 | 34 | 44.9% | 38.6% | 76.5% | 82.4% | 94.1% | 92.1% |
| low_resolution | 118 | 34 | 48.3% | 44.9% | 70.6% | 76.5% | 38.2% | 39.0% |
| noise | 91 | 21 | 36.3% | 38.5% | 71.4% | 81.0% | 38.1% | 41.8% |
| framing | 77 | 21 | 40.8% | 42.1% | 66.7% | 71.4% | 42.9% | 39.0% |
| shadow | 65 | 15 | 40.0% | 38.5% | 73.3% | 80.0% | 26.7% | 33.8% |
| low_brightness | 61 | 11 | 46.7% | 41.7% | 90.9% | 90.9% | 54.5% | 34.4% |
| obstruction | 61 | 19 | 41.0% | 44.3% | 63.2% | 63.2% | 26.3% | 34.4% |
| high_brightness | 54 | 13 | 40.7% | 35.2% | 76.9% | 76.9% | 23.1% | 33.3% |
| low_contrast | 53 | 12 | 47.2% | 35.8% | 66.7% | 75.0% | 25.0% | 41.5% |
| high_contrast | 48 | 11 | 39.6% | 37.5% | 63.6% | 63.6% | 9.1% | 20.8% |

Table 2. Performance shifts when synthetic defects are present. Metrics are limited by the small number of urgent cases within each defect cohort.

![Figure 3. Defect-driven urgent miss and deferral rates before and after quality gating.](reports/figures/figure3_defect_impact.png)
*Figure 3. Top eight synthetic defects ranked by urgent miss reduction. Left panel shows miss rates; right panel shows urgent deferral rates for the same defects.*

While calibration for brightness, contrast, noise, framing, and obstruction thresholds remained fixed at 1.0, the test-set rows still show sizeable deferral rates because every urgent images with these synthetic defects also carried at least one additional trigger. Retake actions for images with these deficits therefore inherit from co-occurring image quality deficits rather than the fixed threshold itself.

Figure 3 demonstrates how motion-blur cases show a modest recall drop because the gate routed 80% of those encounters to retake. The clean pass images improved safety, but most cases fall outside the triage denominator until a replacement image arrives. 

### Fitzpatrick Skin Tone Performance

| Fitzpatrick Type | Total Cases | Urgent Cases | Accuracy (Baseline) | Accuracy (Gated) | Urgent Recall (Baseline) | Urgent Recall (Gated) | Urgent Deferral (Baseline) | Urgent Deferral (Gated) | Retake Rate (Baseline) | Retake Rate (Gated) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All tones | 1,344 | 354 | 40.4% | 40.1% | 73.7% | 76.6% | 0.0% | 15.3% | 0.0% | 14.5% |
| Type I (pale ivory) | 618 | 156 | 41.1% | 41.1% | 77.6% | 81.4% | 0.0% | 19.9% | 0.0% | 20.2% |
| Type II (fair beige) | 146 | 30 | 34.2% | 33.6% | 70.0% | 70.0% | 0.0% | 26.7% | 0.0% | 19.2% |
| Type III (light brown) | 76 | 24 | 44.7% | 46.1% | 58.3% | 58.3% | 0.0% | 8.3% | 0.0% | 17.1% |
| Type IV (medium brown) | 62 | 18 | 35.5% | 32.3% | 66.7% | 66.7% | 0.0% | 22.2% | 0.0% | 19.4% |
| Type V (dark brown) | 142 | 46 | 46.8% | 44.7% | 67.4% | 71.7% | 0.0% | 13.0% | 0.0% | 17.6% |
| Type VI (deeply pigmented) | 300 | 80 | 39.0% | 38.0% | 77.5% | 83.8% | 0.0% | 21.2% | 0.0% | 18.7% |

Table 3. Calibrated SnapCheck performance across ITA-derived Fitzpatrick bins (held-out test set).

Urgent recall gains concentrate in the darker cohorts: Type VI improves by +6.3 percentage points (77.5% to 83.8%) with an 18.7% retake rate, while Type V gains +4.3 points (67.4% to 71.7%). Retake burden is higher for the small Type II subset (21.3%) because the gate routed seven of sixteen urgent encounters to retake; future calibration will smooth these small-sample swings.

### Monk Skin Tone Performance

| Monk Skin Tone | Total Cases | Urgent Cases | Accuracy (Baseline) | Accuracy (Gated) | Urgent Recall (Baseline) | Urgent Recall (Gated) | Urgent Deferral (Baseline) | Urgent Deferral (Gated) | Retake Rate (Baseline) | Retake Rate (Gated) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All tones | 1,344 | 354 | 40.4% | 40.1% | 73.7% | 76.6% | 0.0% | 15.3% | 0.0% | 14.5% |
| MST 1-3 (lighter) | 764 | 186 | 39.8% | 39.7% | 76.3% | 79.6% | 0.0% | 21.0% | 0.0% | 20.0% |
| MST 4-7 (medium) | 244 | 76 | 44.7% | 43.0% | 71.1% | 73.7% | 0.0% | 14.5% | 0.0% | 18.0% |
| MST 1 | 462 | 122 | 41.6% | 42.0% | 77.9% | 82.0% | 0.0% | 21.3% | 0.0% | 20.6% |
| MST 8-10 (darker) | 336 | 92 | 38.8% | 37.9% | 70.7% | 76.1% | 0.0% | 19.6% | 0.0% | 18.5% |
| MST 2 | 156 | 34 | 39.7% | 38.5% | 76.5% | 79.4% | 0.0% | 14.7% | 0.0% | 19.2% |
| MST 3 | 146 | 30 | 34.2% | 33.6% | 70.0% | 70.0% | 0.0% | 26.7% | 0.0% | 19.2% |
| MST 4 | 100 | 30 | 40.0% | 40.0% | 53.3% | 53.3% | 0.0% | 13.3% | 0.0% | 18.0% |
| MST 5 | 38 | 12 | 42.1% | 39.5% | 83.3% | 83.3% | 0.0% | 16.7% | 0.0% | 18.4% |
| MST 6 | 78 | 22 | 44.9% | 42.3% | 81.8% | 81.8% | 0.0% | 9.1% | 0.0% | 14.1% |
| MST 7 | 28 | 12 | 64.3% | 60.7% | 83.3% | 100.0% | 0.0% | 25.0% | 0.0% | 28.6% |
| MST 8 | 36 | 12 | 37.1% | 37.1% | 25.0% | 25.0% | 0.0% | 8.3% | 0.0% | 16.7% |
| MST 9 | 20 | 6 | 30.0% | 35.0% | 66.7% | 66.7% | 0.0% | 33.3% | 0.0% | 15.0% |
| MST 10 | 280 | 74 | 39.6% | 38.2% | 78.4% | 85.1% | 0.0% | 20.3% | 0.0% | 18.9% |

![Figure 4. SnapCheck reduces urgent misses across Monk Skin Tone groupings.](reports/figures/figure4_skin_tone.png)
*Figure 4. Urgent miss rates for grouped Monk Skin Tone bins (lighter MST 1–3, medium MST 4–7, darker MST 8–10) before and after SnapCheck gating.*

Table 4. Calibrated SnapCheck performance across Monk Skin Tone bins (1=lighter, 10=darker; held-out test set).

Monk groupings mirror the Fitzpatrick trends: MST 8-10 gains +5.4 urgent-recall points (70.7% to 76.1%) with an 18.5% retake rate, while lighter MST 1-3 tones remain within 3.3 points of baseline recall (76.3% to 79.6%). Figure 4 shows that urgent miss reductions persist across all three aggregated tone groupings. Small strata (e.g., MST 3, MST 5-7) show noisier retake estimates, underscoring the need for clinician overrides in future work.

## Discussion
DIQA gating provides a concrete improvement in VLM-based identification of urgent dermatology cases in triage: urgent misses fall from 26.3% to 8.2% (a 69% relative reduction) while urgent recall increases by 2.9 percentage points (73.7% to 76.6%). The gate introduces a 14.5% retake burden (195 of 1,344 encounters) and defers 15.3% of urgent exposures (54 lesions) for manual follow-up, covering 29% of degraded inputs without modifying VLM weights. These trade-offs align with clinical reports that up to 50% of asynchronous submissions are initially unreadable. Notably, Type VI Fitzpatrick lesions gained +6.3 urgent-recall points (77.5% to 83.8%) and Monk Skin Tone 8-10 gained +5.4 points (70.7% to 76.1%) with retake rates under 19%, indicating DIQA paired with VLMs can raise sensitivity without widening observed skin-tone disparities.

Blur defects show the clearest benefit from gating, yielding a +20.7 percentage-point recall gain when the blurred image is swapped with the pristine image. Noise (+9.6 points) and low-resolution (+5.9 points) defects also improve meaningfully. Motion-blur encounters, however, suggest limitations to aggressive retake logic: 92.1% of urgent motion-blur cases were deferred, allowing only a +5.9 point recall lift despite the underlying safety improvement. These findings support the idea that performance may  improve when DIQA approaches include capture coaching or adaptive thresholds that relax retake rules when no pass image is available.

Clinically, a DIQA tool such as SnapCheck may act as a configurable guardrail for high-risk degradations disproportionately driving urgent misses. The modest +2.9 percentage-point recall lift is consistent with the validation-driven thresholds we selected - blur and low-resolution cutoffs were tuned to keep retakes <=15%, while exposure, contrast, noise, shadow, framing, and overall fail triggers were fixed at 1.0. This configuration routed only 54 of 177 urgent lesions through the gate and pushed almost every urgent motion-blur case into the retake queue. This behavior maximizes safety but also dampens immediate recall.

SnapCheck currently provides the most improvement for defects that obscure lesion detail systematically. Blur exposures gained 20.7 urgent-recall points (62.1% to 82.8%), noise cohorts gained 9.6 points (71.4% to 81.0%), and low-resolution failures gained 5.9 points (70.6% to 76.5%) with acceptable retake burden. Motion blur, by contrast, remains dominated by deferrals (92.1% retake rate) because the gate has no intermediate option between retake and substitution. These findings suggest a need for more adaptive DIQA gating thresholds, or capture coaching that assists patients in capturing steadier shots before dermatologist review.

Possible avenues for assessing real-world applicability include conducting additional benchmarks across different model types, collecting of dermatologist override labels in a real-world practice setting, retraining of the quality model on patient-generated photos, and pairing of SnapCheck with patient-facing guidance. These would help optimize accuracy of retake/recall assignments and anchor fairness auditing in dermatologist-labelled skin tones rather than ITA proxies.

## Limitations
Findings rely on synthetic degradations and paired pass images drawn from HAM10000 and Derm7pt. We have not yet benchmarked SnapCheck on fully external datasets such as SCIN or prospective patient-generated photographs. As such, use of real-world retakes may not result in similar performance gains. The held-out test set remains relatively small (1,344 evaluation exposures across 672 lesion pairs, 177 urgent lesions), so subgroup metrics-particularly those with fewer than 20 cases-carry wide confidence intervals. The conservative threshold strategy capping retakes near 15% also limited the immediate recall lift. Overall diagnostic accuracy remains modest with and without gating, likely due to our use of a distilled model (GPT-5 Nano) and that the VLM is implemented as a safety adjunct rather than a diagnostic system. Prospective studies with clinician overrides and patient-uploaded photos would be required to tune thresholds, confirm fairness, and assess workflow impact.

## Conclusion
TeleDerm SnapCheck demonstrates that DIQA can be operationalized alongside VLM triage and that calibrated thresholds deliver measurable urgent-sensitivity gains. Aligning retake burden with clinic capacity and validating performance on real submissions are potential avenues toward clinical deployment.

## Ethics and Data Availability
All experiments were conducted on de-identified public dermoscopy datasets distributed under research-friendly licenses (ISIC 2020, HAM10000, Derm7pt); no institutional review board approval was required. Synthetic degradations were generated in-house without accessing protected health information. Source code, prompts, configuration files, and aggregate metrics are openly available in this repository, and cached VLM responses are provided as supplementary material to facilitate independent replication.

## Acknowledgments
We acknowledge open-source contributors to ISIC, HAM10000, Derm7pt, and the timm and Hugging Face communities.

## Data and Code Availability
All scripts, configurations, and checkpoints are available in this repository. Public datasets are accessible via their respective licenses; synthetic augmentations will be released under an open data agreement pending institutional approval.
