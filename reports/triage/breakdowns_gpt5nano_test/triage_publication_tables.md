# TeleDerm SnapCheck Triage Performance Tables

All metrics computed on the 721-image held-out test set (baseline vs. quality-gated modes).

## Fitzpatrick Skin Tone Performance

| Fitzpatrick Type | Total Cases | Urgent Cases | Accuracy (Baseline) | Accuracy (Gated) | Urgent Recall (Baseline) | Urgent Recall (Gated) | Urgent Deferral (Baseline) | Urgent Deferral (Gated) | Retake Rate (Baseline) | Retake Rate (Gated) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All tones | 721 | 203 | 37.3% | 37.2% | 73.9% | 74.4% | 0.0% | 12.8% | 0.0% | 14.6% |
| Type I (pale ivory) | 364 | 105 | 35.7% | 35.2% | 78.1% | 77.1% | 0.0% | 7.6% | 0.0% | 11.8% |
| Type II (fair beige) | 75 | 16 | 33.3% | 33.3% | 81.2% | 75.0% | 0.0% | 43.8% | 0.0% | 21.3% |
| Type III (light brown) | 35 | 12 | 40.0% | 40.0% | 75.0% | 75.0% | 0.0% | 8.3% | 0.0% | 14.3% |
| Type IV (medium brown) | 31 | 10 | 25.8% | 22.6% | 50.0% | 50.0% | 0.0% | 10.0% | 0.0% | 19.4% |
| Type V (dark brown) | 67 | 20 | 47.8% | 47.8% | 60.0% | 65.0% | 0.0% | 15.0% | 0.0% | 13.4% |
| Type VI (deeply pigmented) | 149 | 40 | 40.3% | 41.6% | 72.5% | 77.5% | 0.0% | 15.0% | 0.0% | 17.4% |

## Monk Skin Tone Performance

| Monk Skin Tone | Total Cases | Urgent Cases | Accuracy (Baseline) | Accuracy (Gated) | Urgent Recall (Baseline) | Urgent Recall (Gated) | Urgent Deferral (Baseline) | Urgent Deferral (Gated) | Retake Rate (Baseline) | Retake Rate (Gated) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All tones | 721 | 203 | 37.3% | 37.2% | 73.9% | 74.4% | 0.0% | 12.8% | 0.0% | 14.6% |
| MST 1-3 (lighter) | 439 | 121 | 35.3% | 34.9% | 78.5% | 76.9% | 0.0% | 12.4% | 0.0% | 13.4% |
| MST 4-7 (medium) | 122 | 39 | 40.2% | 39.3% | 61.5% | 64.1% | 0.0% | 12.8% | 0.0% | 15.6% |
| MST 1 | 272 | 74 | 33.8% | 33.5% | 79.7% | 78.4% | 0.0% | 8.1% | 0.0% | 12.5% |
| MST 8-10 (darker) | 160 | 43 | 40.6% | 41.9% | 72.1% | 76.7% | 0.0% | 14.0% | 0.0% | 16.9% |
| MST 2 | 92 | 31 | 41.3% | 40.2% | 74.2% | 74.2% | 0.0% | 6.5% | 0.0% | 9.8% |
| MST 3 | 75 | 16 | 33.3% | 33.3% | 81.2% | 75.0% | 0.0% | 43.8% | 0.0% | 21.3% |
| MST 4 | 40 | 14 | 37.5% | 37.5% | 71.4% | 71.4% | 0.0% | 14.3% | 0.0% | 17.5% |
| MST 5 | 26 | 8 | 26.9% | 23.1% | 50.0% | 50.0% | 0.0% | 0.0% | 0.0% | 15.4% |
| MST 6 | 32 | 13 | 50.0% | 50.0% | 53.8% | 61.5% | 0.0% | 23.1% | 0.0% | 18.8% |
| MST 7 | 24 | 4 | 45.8% | 45.8% | 75.0% | 75.0% | 0.0% | 0.0% | 0.0% | 8.3% |
| MST 8 | 11 | 3 | 45.5% | 45.5% | 66.7% | 66.7% | 0.0% | 0.0% | 0.0% | 9.1% |
| MST 9 | 14 | 2 | 42.9% | 42.9% | 50.0% | 50.0% | 0.0% | 0.0% | 0.0% | 7.1% |
| MST 10 | 135 | 38 | 40.0% | 41.5% | 73.7% | 78.9% | 0.0% | 15.8% | 0.0% | 18.5% |

## Top Diagnoses (>=50 Cases)

| Diagnosis | Total Cases | Urgent Cases | Accuracy (Baseline) | Accuracy (Gated) | Urgent Recall (Baseline) | Urgent Recall (Gated) |
| --- | --- | --- | --- | --- | --- | --- |
| nv | 265 | 0 | 17.7% | 17.4% | n/a | n/a |
| basal cell carcinoma | 126 | 126 | 67.5% | 66.7% | 67.5% | 66.7% |
| clark nevus | 116 | 0 | 26.7% | 26.7% | n/a | n/a |
| blue nevus | 81 | 0 | 30.9% | 30.9% | n/a | n/a |
| all | 721 | 203 | 37.3% | 37.2% | 73.9% | 74.4% |

## Synthetic Defect Stress Test (Defect Present)

| Synthetic Defect | Total Cases | Urgent Cases | Accuracy (Baseline) | Accuracy (Gated) | Urgent Recall (Baseline) | Urgent Recall (Gated) | Retake Rate (Baseline) | Retake Rate (Gated) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall_fail | 361 | 100 | 35.2% | 34.9% | 72.0% | 73.0% | 0.0% | 28.8% |
| motion_blur | 69 | 13 | 31.9% | 27.5% | 69.2% | 61.5% | 0.0% | 79.7% |
| low_resolution | 69 | 20 | 40.6% | 44.9% | 80.0% | 90.0% | 0.0% | 29.0% |
| blur | 68 | 22 | 35.3% | 33.8% | 59.1% | 63.6% | 0.0% | 64.7% |
| noise | 57 | 16 | 35.1% | 33.3% | 68.8% | 68.8% | 0.0% | 28.1% |
| shadow | 37 | 13 | 24.3% | 27.0% | 53.8% | 53.8% | 0.0% | 27.0% |
| obstruction | 37 | 11 | 32.4% | 32.4% | 63.6% | 63.6% | 0.0% | 16.2% |
| high_brightness | 33 | 12 | 42.4% | 36.4% | 83.3% | 83.3% | 0.0% | 18.2% |
| framing | 33 | 5 | 36.4% | 42.4% | 80.0% | 80.0% | 0.0% | 24.2% |
| low_brightness | 27 | 9 | 33.3% | 37.0% | 66.7% | 77.8% | 0.0% | 33.3% |
| low_contrast | 26 | 7 | 50.0% | 50.0% | 71.4% | 71.4% | 0.0% | 15.4% |
| high_contrast | 18 | 4 | 44.4% | 44.4% | 100.0% | 100.0% | 0.0% | 16.7% |
