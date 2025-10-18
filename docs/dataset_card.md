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
- `skin_tone_score`: Proxy brightness estimate (0-1).
- `skin_tone_bin`: Discrete bin index derived from `skin_tone_score`.
- `capture_channel`: `clinic` or `patient_generated` (heuristic).
- `meta_*`: Hyperparameters used to synthesize each defect.
- Quality labels: Binary indicators matching taxonomy above.

## Generation Process
1. Sample balanced subset from each source dataset (`base_images_per_source`).
2. Save pristine copy (pass) with all quality labels = 0.
3. Apply up to `max_augmentations_per_image` random defects using `snapcheck.augmentations.QualityAugmentor`.
4. Store augmented image and JSONL/CSV manifests with label metadata.
5. Split into train/val/test using stratified sampling on `overall_fail`.

## Ethical / Fairness Considerations
- Skin tone bins rely on crude brightness proxy; fine-tuning with expert Fitzpatrick labels recommended.
- Obstruction patches do not guarantee realism of hair/jewelry occlusion.
- Synthetic defects should be calibrated with clinician feedback before deployment.

## Recommended Usage
- Train DIQA models for teledermatology quality gating.
- Benchmark fairness across skin tone bins and capture channels.
- Provide augmentation templates for prospective patient photo capture studies.

## Citation
Please cite the original HAM10000 and Derm7pt publications when releasing derived work.

