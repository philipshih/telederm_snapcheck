#!/usr/bin/env python
"""Reapply alternative quality thresholds to an existing triage detail CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import pandas as pd

from snapcheck.quality_model import TARGET_LABELS
from snapcheck.triage import compute_triage_metrics


DEFAULT_DETAIL_PATH = Path("reports/triage/triage_detail.csv")


def load_thresholds(threshold_path: Path | None) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    if threshold_path is None:
        return thresholds
    data = json.loads(threshold_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "thresholds" in data:
        for label, payload in data["thresholds"].items():
            value = payload.get("threshold") if isinstance(payload, dict) else payload
            if value is not None:
                thresholds[label] = float(value)
    elif isinstance(data, dict):
        for label, value in data.items():
            if isinstance(value, dict) and "threshold" in value:
                thresholds[label] = float(value["threshold"])
            elif value is not None:
                thresholds[label] = float(value)
    return thresholds


def apply_overrides(thresholds: Dict[str, float], overrides: List[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected format label=value")
        label, value = item.split("=", 1)
        if not label:
            raise ValueError(f"Missing label in override '{item}'")
        thresholds[label.strip()] = float(value)


def failing_labels(scores: Dict[str, float], thresholds: Dict[str, float]) -> Tuple[bool, List[str]]:
    fail_labels: List[str] = []
    overall_key = "overall_fail"
    if overall_key in scores:
        overall_threshold = thresholds.get(overall_key)
        if overall_threshold is not None and scores[overall_key] >= overall_threshold:
            fail_labels.append(overall_key)
    for label, value in scores.items():
        if label == overall_key:
            continue
        threshold = thresholds.get(label)
        if threshold is None:
            threshold = 0.5
        if value >= threshold:
            fail_labels.append(label)
    return bool(fail_labels), fail_labels


def build_results(detail_df: pd.DataFrame, column: str) -> List[SimpleNamespace]:
    results: List[SimpleNamespace] = []
    for row in detail_df.itertuples(index=False):
        results.append(
            SimpleNamespace(
                ground_truth_triage=getattr(row, "ground_truth_triage", "unknown"),
                model_triage=getattr(row, "model_triage", "unknown"),
                gated_triage=getattr(row, column, None),
                latency=float(getattr(row, "latency", 0.0) or 0.0),
                token_usage=(getattr(row, "token_usage", None)),
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Reapply quality thresholds offline using existing triage outputs")
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAIL_PATH, help="Path to triage detail CSV")
    parser.add_argument("--thresholds", type=Path, default=None, help="Threshold JSON (reports/diqa/thresholds.json)")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a threshold as label=value (can be provided multiple times)",
    )
    parser.add_argument("--out-summary", type=Path, default=None, help="Optional path to write recalibrated summary CSV")
    parser.add_argument("--out-detail", type=Path, default=None, help="Optional path to write detail CSV with recalculated columns")
    args = parser.parse_args()

    if not args.detail_csv.exists():
        raise FileNotFoundError(f"Detail CSV not found: {args.detail_csv}")

    detail_df = pd.read_csv(args.detail_csv)
    score_columns = [f"quality_score_{label}" for label in TARGET_LABELS]
    missing_cols = [col for col in score_columns if col not in detail_df.columns]
    if missing_cols:
        raise RuntimeError(
            "Detail CSV is missing quality score columns. Rerun the triage simulation with the updated pipeline to "
            "capture per-label scores."
        )

    thresholds = load_thresholds(args.thresholds)
    apply_overrides(thresholds, args.set)

    recal_fail: List[bool] = []
    recal_labels: List[str] = []
    recal_triage: List[str] = []

    for row in detail_df.itertuples(index=False):
        scores = {label: getattr(row, f"quality_score_{label}") for label in TARGET_LABELS}
        scores = {k: float(v) for k, v in scores.items() if pd.notna(v)}
        fail, labels_hit = failing_labels(scores, thresholds)
        recal_fail.append(fail)
        recal_labels.append(";".join(labels_hit))
        recal_triage.append("retake" if fail else getattr(row, "model_triage", "unknown"))

    detail_df["recal_quality_fail"] = recal_fail
    detail_df["recal_quality_fail_labels"] = recal_labels
    detail_df["recal_gated_triage"] = recal_triage

    baseline_results = build_results(detail_df, "model_triage")
    recal_results = build_results(detail_df, "recal_gated_triage")

    baseline_metrics = compute_triage_metrics(baseline_results, triage_attr="model_triage")
    recal_metrics = compute_triage_metrics(recal_results, triage_attr="gated_triage")

    print("Baseline metrics (ungated):")
    for key, value in baseline_metrics.items():
        display = f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}"
        print(display)

    print("\nRecalibrated metrics:")
    for key, value in recal_metrics.items():
        display = f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}"
        print(display)

    if args.out_summary is not None:
        summary_df = pd.DataFrame(
            [
                {"mode": "baseline", **baseline_metrics},
                {"mode": "recalibrated", **recal_metrics},
            ]
        )
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.out_summary, index=False)
        print(f"\nWrote summary metrics to {args.out_summary}")


    if args.out_detail is not None:
        args.out_detail.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(args.out_detail, index=False)
        print(f"Wrote recalibrated detail to {args.out_detail}")


if __name__ == "__main__":
    main()
