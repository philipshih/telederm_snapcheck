# Sequential Triage Run Plan

## Purpose
- Respect API rate limits by processing small batches sequentially.
- Capture per-run metrics for manuscript tracking.
- Enable offline threshold tuning without re-querying vision-language models.

## Workflow
1. **Configure batch**: edit `configs/triage_eval.yaml` (GPT-5 Nano backend), adjusting `simulation.num_images` / `random_subset` and cache directories so each run writes to a dedicated cache (responses in `reports/triage/cache/` remain reusable).
2. **Run simulation**: `python scripts/run_triage_simulation.py --config triage_eval.yaml`. The script appends metrics to `reports/triage/run_history.csv`, emits progress to `reports/triage/triage_detail_progress.log`, and prints retake/latency summaries.
3. **Review detail output**: inspect `reports/triage/triage_detail.csv` (or the configured path). Per-label quality probabilities (`quality_score_<label>`) and failing labels are included for calibration; corresponding summaries land in `reports/triage/breakdowns/`.
4. **Recalibrate thresholds offline**: call `python scripts/reapply_quality_thresholds.py --detail-csv reports/triage/triage_detail.csv --thresholds reports/diqa/thresholds.json --set overall_fail=0.05` (override as needed). Compare baseline versus recalibrated metrics without new API traffic.
5. **Update config**: once acceptable retake/recall trade-offs are found, write the tuned thresholds back into the triage config before the next batch.
6. **Repeat sequentially**: queue the next batch only after the previous run and recalibration complete. Respect any rate limit cooldowns noted in the console output.

## GPT-5 Nano Notes
- Leverages OpenAI `gpt-5-nano` (400k context, 128k output, $0.05 / $0.40 per 1M tokens).
- Keep `max_tokens` modest (current config 2000) unless longer reasoning is required to control spend.
- Backoff is set to 45s with 6 retries; adjust if tier limits demand longer cooldowns.
## Tips
- If the API returns rate-limit errors, re-run the same config later; cached responses prevent duplicate calls.
- `reports/triage/run_history.csv` accumulates backend metrics across runs - use it to populate manuscript tables or trigger recalibration when retake rates drift.
- Store recalibrated summaries with `--out-summary` to maintain an audit trail of threshold changes over time.
