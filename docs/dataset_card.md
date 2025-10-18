# TeleDerm SnapCheck Synthetic DIQA Dataset Card

## Overview
- **Dataset name:** TeleDerm SnapCheck DIQA v1
- **Task:** Multi-label classification of dermatologic image quality defects with auto-generated labels.
- **Sources:** HAM10000, Derm7pt (public, histopathologically validated skin lesion datasets).
- **Records:** 2x the number of base images (one pass + one degraded image per source frame).

## Label Taxonomy
| Label | Description |
|-------|-------------|
| blur | Gaussian blur simulating defocus |
| motion_blur | Linear streaks simulating camera shake |
| low_brightness / high_brightness | Under/over-exposed lighting |
| low_contrast / high_contrast | Dynamic range compression/expansion |
| noise | Gaussian sensor noise |
| shadow | Synthetic cast shadows |
| obstruction | Occluding patch (hair/objects) |
| framing | Improper zoom/cropping |
| low_resolution | Downsampled and upscaled image |
| overall_fail | Any quality issue present |

## Data Fields
- `image_path`: Relative path within `data/quality/images`.
- `diagnosis`: Original lesion diagnosis from source metadata.
- `ita_score`: Individual Typology Angle (median of four peripheral patches; degrees).
- `fitzpatrick_type`: ITA-mapped Fitzpatrick bin (0=Type I ... 5=Type VI).
- `monk_skin_tone`: ITA-mapped Monk Skin Tone index (1-10).
- `skin_tone_bin`: Legacy alias for `fitzpatrick_type` retained for compatibility.
- `capture_channel`: `clinic` or `patient_generated` (heuristic).
- `meta_*`: Hyperparameters used to synthesize each defect.
- Quality labels: Binary indicators matching taxonomy above.

## Generation Process
1. Sample balanced subset from each source dataset (`base_images_per_source`).
2. Compute ITA from four corner patches and convert to Fitzpatrick/Monk bins.
3. Save pristine copy (pass) with all quality labels = 0.
4. Apply up to `max_augmentations_per_image` random defects using `snapcheck.augmentations.QualityAugmentor`.
5. Store augmented image and JSONL/CSV manifests with label metadata + skin tone fields.
6. Split into train/val/test using stratified sampling on `overall_fail`.

## Ethical / Fairness Considerations
- ITA-derived skin tone remains a proxy; incorporate expert labels (e.g., MSKCC) when available.
- Obstruction patches do not guarantee realism of hair/jewelry occlusion.
- Synthetic defects should be calibrated with clinician feedback before deployment.

## Recommended Usage
- Train DIQA models for teledermatology quality gating.
- Benchmark fairness across skin tone bins and capture channels.
- Provide augmentation templates for prospective patient photo capture studies.

## Citation
Please cite the original HAM10000 and Derm7pt publications when releasing derived work.

