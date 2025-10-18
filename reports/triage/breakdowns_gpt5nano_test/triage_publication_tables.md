# TeleDerm SnapCheck Triage Performance Tables

All metrics computed on the 4,800-image evaluation set (baseline vs. quality-gated modes).

## Fitzpatrick Skin Tone Performance

| Fitzpatrick Type | Total Cases | Urgent Cases | Accuracy (Baseline) | Accuracy (Gated) | Urgent Recall (Baseline) | Urgent Recall (Gated) | Urgent Deferral (Baseline) | Urgent Deferral (Gated) | Retake Rate (Baseline) | Retake Rate (Gated) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All tones | 1,344 | 354 | 40.4% | 39.8% | 73.7% | 77.4% | 0.0% | 19.2% | 0.0% | 19.3% |
| Type I (pale ivory) | 618 | 156 | 41.1% | 41.1% | 77.6% | 81.4% | 0.0% | 19.9% | 0.0% | 20.2% |
| Type II (fair beige) | 146 | 30 | 34.2% | 33.6% | 70.0% | 70.0% | 0.0% | 26.7% | 0.0% | 19.2% |
| Type III (light brown) | 76 | 24 | 44.7% | 46.1% | 58.3% | 58.3% | 0.0% | 8.3% | 0.0% | 17.1% |
| Type IV (medium brown) | 62 | 18 | 35.5% | 32.3% | 66.7% | 66.7% | 0.0% | 22.2% | 0.0% | 19.4% |
| Type V (dark brown) | 142 | 46 | 46.8% | 44.7% | 67.4% | 71.7% | 0.0% | 13.0% | 0.0% | 17.6% |
| Type VI (deeply pigmented) | 300 | 80 | 39.0% | 38.0% | 77.5% | 83.8% | 0.0% | 21.2% | 0.0% | 18.7% |

## Monk Skin Tone Performance

| Monk Skin Tone | Total Cases | Urgent Cases | Accuracy (Baseline) | Accuracy (Gated) | Urgent Recall (Baseline) | Urgent Recall (Gated) | Urgent Deferral (Baseline) | Urgent Deferral (Gated) | Retake Rate (Baseline) | Retake Rate (Gated) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All tones | 1,344 | 354 | 40.4% | 39.8% | 73.7% | 77.4% | 0.0% | 19.2% | 0.0% | 19.3% |
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

## Top Diagnoses (>=50 Cases)

| Diagnosis | Total Cases | Urgent Cases | Accuracy (Baseline) | Accuracy (Gated) | Urgent Recall (Baseline) | Urgent Recall (Gated) |
| --- | --- | --- | --- | --- | --- | --- |
| nv | 458 | 0 | 27.1% | 24.5% | n/a | n/a |
| clark nevus | 250 | 0 | 26.9% | 26.9% | n/a | n/a |
| basal cell carcinoma | 224 | 224 | 69.6% | 72.8% | 69.6% | 72.8% |
| blue nevus | 128 | 0 | 32.8% | 28.9% | n/a | n/a |
| mel | 100 | 100 | 87.0% | 93.0% | 87.0% | 93.0% |
| bkl | 96 | 0 | 29.2% | 28.1% | n/a | n/a |
| all | 1,344 | 354 | 40.4% | 39.8% | 73.7% | 77.4% |

## Synthetic Defect Stress Test (Defect Present)

| Synthetic Defect | Total Cases | Urgent Cases | Accuracy (Baseline) | Accuracy (Gated) | Urgent Recall (Baseline) | Urgent Recall (Gated) | Retake Rate (Baseline) | Retake Rate (Gated) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall_fail | 672 | 177 | 42.0% | 40.8% | 68.4% | 75.7% | 0.0% | 38.5% |
| blur | 127 | 29 | 40.9% | 41.7% | 62.1% | 82.8% | 0.0% | 78.7% |
| motion_blur | 127 | 34 | 44.9% | 38.6% | 76.5% | 82.4% | 0.0% | 92.1% |
| low_resolution | 118 | 34 | 48.3% | 44.9% | 70.6% | 76.5% | 0.0% | 39.0% |
| noise | 91 | 21 | 36.3% | 38.5% | 71.4% | 81.0% | 0.0% | 41.8% |
| framing | 77 | 21 | 40.8% | 42.1% | 66.7% | 71.4% | 0.0% | 39.0% |
| shadow | 65 | 15 | 40.0% | 38.5% | 73.3% | 80.0% | 0.0% | 33.8% |
| low_brightness | 61 | 11 | 46.7% | 41.7% | 90.9% | 90.9% | 0.0% | 34.4% |
| obstruction | 61 | 19 | 41.0% | 44.3% | 63.2% | 63.2% | 0.0% | 34.4% |
| high_brightness | 54 | 13 | 40.7% | 35.2% | 76.9% | 76.9% | 0.0% | 33.3% |
| low_contrast | 53 | 12 | 47.2% | 35.8% | 66.7% | 75.0% | 0.0% | 41.5% |
| high_contrast | 48 | 11 | 39.6% | 37.5% | 63.6% | 63.6% | 0.0% | 20.8% |
