#!/usr/bin/env python
"""Build publication-ready triage breakdown tables.

This helper formats the CSV outputs from `analyze_triage_breakdowns.py`
into Markdown tables with consistent ordering, integer counts, and
percentage metrics rounded for readability.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

DEFAULT_BREAKDOWN_DIR = Path("reports/triage/breakdowns")
DEFAULT_OUTPUT = DEFAULT_BREAKDOWN_DIR / "triage_publication_tables.md"


def _format_percent(series: pd.Series, decimals: int = 1) -> pd.Series:
    return series.apply(
        lambda value: f"{value * 100:.{decimals}f}%"
        if pd.notna(value)
        else "n/a"
    )


def _format_count(series: pd.Series) -> pd.Series:
    return series.apply(lambda value: f"{int(value):,}" if pd.notna(value) else "n/a")


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in df.to_numpy()
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def _select_columns(
    df: pd.DataFrame,
    *,
    group_column: str,
    rename_map: dict[str, str],
    percent_columns: Iterable[str],
    count_columns: Iterable[str],
) -> pd.DataFrame:
    columns: List[str] = [group_column, *count_columns, *percent_columns]
    trimmed = df.loc[:, columns].copy()
    for column in count_columns:
        trimmed[column] = _format_count(trimmed[column])
    for column in percent_columns:
        trimmed[column] = _format_percent(trimmed[column])
    return trimmed.rename(columns=rename_map)


def _build_skin_tone_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["group_type"] == "skin_tone"].copy()

    def _skin_order(value: object) -> int:
        text = str(value).strip().lower()
        if text == "all":
            return -1
        if text.isdigit():
            return int(text)
        if text == "unknown":
            return 99
        # fallback for any non-numeric label
        try:
            return int(float(text))
        except ValueError:
            return 99

    df["order"] = df["group_value"].apply(_skin_order)
    df.sort_values("order", inplace=True)

    rename_map = {
        "skin_tone_label": "Fitzpatrick Type",
        "cases_total": "Total Cases",
        "cases_urgent": "Urgent Cases",
        "baseline_triage_accuracy": "Accuracy (Baseline)",
        "quality_gated_triage_accuracy": "Accuracy (Gated)",
        "baseline_urgent_recall": "Urgent Recall (Baseline)",
        "quality_gated_urgent_recall": "Urgent Recall (Gated)",
        "baseline_urgent_deferral_rate": "Urgent Deferral (Baseline)",
        "quality_gated_urgent_deferral_rate": "Urgent Deferral (Gated)",
        "baseline_retake_rate": "Retake Rate (Baseline)",
        "quality_gated_retake_rate": "Retake Rate (Gated)",
    }

    columns = [
        "skin_tone_label",
        "cases_total",
        "cases_urgent",
        "baseline_triage_accuracy",
        "quality_gated_triage_accuracy",
        "baseline_urgent_recall",
        "quality_gated_urgent_recall",
        "baseline_urgent_deferral_rate",
        "quality_gated_urgent_deferral_rate",
        "baseline_retake_rate",
        "quality_gated_retake_rate",
    ]

    percent_columns = [
        "baseline_triage_accuracy",
        "quality_gated_triage_accuracy",
        "baseline_urgent_recall",
        "quality_gated_urgent_recall",
        "baseline_urgent_deferral_rate",
        "quality_gated_urgent_deferral_rate",
        "baseline_retake_rate",
        "quality_gated_retake_rate",
    ]
    count_columns = ["cases_total", "cases_urgent"]

    table = _select_columns(
        df[columns],
        group_column="skin_tone_label",
        rename_map=rename_map,
        percent_columns=percent_columns,
        count_columns=count_columns,
    )
    return table


def _build_monk_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["group_type"] == "monk_skin_tone"].copy()
    if df.empty:
        return pd.DataFrame()

    def _order_value(value: object) -> int:
        text = str(value)
        if text == "all":
            return -2
        if text.startswith("MST 1-3"):
            return -1
        if text.startswith("MST 4-7"):
            return 0
        if text.startswith("MST 8-10"):
            return 1
        try:
            return int(value)
        except (TypeError, ValueError):
            return 100

    df["order"] = df["group_value"].apply(_order_value)
    df.sort_values(["order", "monk_label"], inplace=True)

    rename_map = {
        "monk_label": "Monk Skin Tone",
        "cases_total": "Total Cases",
        "cases_urgent": "Urgent Cases",
        "baseline_triage_accuracy": "Accuracy (Baseline)",
        "quality_gated_triage_accuracy": "Accuracy (Gated)",
        "baseline_urgent_recall": "Urgent Recall (Baseline)",
        "quality_gated_urgent_recall": "Urgent Recall (Gated)",
        "baseline_urgent_deferral_rate": "Urgent Deferral (Baseline)",
        "quality_gated_urgent_deferral_rate": "Urgent Deferral (Gated)",
        "baseline_retake_rate": "Retake Rate (Baseline)",
        "quality_gated_retake_rate": "Retake Rate (Gated)",
    }

    columns = [
        "monk_label",
        "cases_total",
        "cases_urgent",
        "baseline_triage_accuracy",
        "quality_gated_triage_accuracy",
        "baseline_urgent_recall",
        "quality_gated_urgent_recall",
        "baseline_urgent_deferral_rate",
        "quality_gated_urgent_deferral_rate",
        "baseline_retake_rate",
        "quality_gated_retake_rate",
    ]

    percent_columns = [
        "baseline_triage_accuracy",
        "quality_gated_triage_accuracy",
        "baseline_urgent_recall",
        "quality_gated_urgent_recall",
        "baseline_urgent_deferral_rate",
        "quality_gated_urgent_deferral_rate",
        "baseline_retake_rate",
        "quality_gated_retake_rate",
    ]
    count_columns = ["cases_total", "cases_urgent"]

    table = _select_columns(
        df[columns],
        group_column="monk_label",
        rename_map=rename_map,
        percent_columns=percent_columns,
        count_columns=count_columns,
    )
    return table


def _build_diagnosis_table(path: Path, minimum_cases: int = 50) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["group_type"] == "diagnosis"].copy()
    df["order"] = df.apply(
        lambda row: (-1 if row["group_value"] == "all" else row["cases_total"]),
        axis=1,
    )
    df = df.sort_values(["order"], ascending=False)
    filtered = df[(df["group_value"] == "all") | (df["cases_total"] >= minimum_cases)].copy()

    rename_map = {
        "group_value": "Diagnosis",
        "cases_total": "Total Cases",
        "cases_urgent": "Urgent Cases",
        "baseline_triage_accuracy": "Accuracy (Baseline)",
        "quality_gated_triage_accuracy": "Accuracy (Gated)",
        "baseline_urgent_recall": "Urgent Recall (Baseline)",
        "quality_gated_urgent_recall": "Urgent Recall (Gated)",
    }

    columns = [
        "group_value",
        "cases_total",
        "cases_urgent",
        "baseline_triage_accuracy",
        "quality_gated_triage_accuracy",
        "baseline_urgent_recall",
        "quality_gated_urgent_recall",
    ]
    percent_columns = [
        "baseline_triage_accuracy",
        "quality_gated_triage_accuracy",
        "baseline_urgent_recall",
        "quality_gated_urgent_recall",
    ]
    count_columns = ["cases_total", "cases_urgent"]

    table = _select_columns(
        filtered[columns],
        group_column="group_value",
        rename_map=rename_map,
        percent_columns=percent_columns,
        count_columns=count_columns,
    )
    return table


def _build_defect_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["group_type"] == "defect") & (df["defect_state"] == "present")].copy()
    df["order"] = df["cases_total"].rank(method="first", ascending=False)
    df.sort_values("order", inplace=True)

    rename_map = {
        "group_value": "Synthetic Defect",
        "cases_total": "Total Cases",
        "cases_urgent": "Urgent Cases",
        "baseline_triage_accuracy": "Accuracy (Baseline)",
        "quality_gated_triage_accuracy": "Accuracy (Gated)",
        "baseline_urgent_recall": "Urgent Recall (Baseline)",
        "quality_gated_urgent_recall": "Urgent Recall (Gated)",
        "baseline_retake_rate": "Retake Rate (Baseline)",
        "quality_gated_retake_rate": "Retake Rate (Gated)",
    }

    columns = [
        "group_value",
        "cases_total",
        "cases_urgent",
        "baseline_triage_accuracy",
        "quality_gated_triage_accuracy",
        "baseline_urgent_recall",
        "quality_gated_urgent_recall",
        "baseline_retake_rate",
        "quality_gated_retake_rate",
    ]
    percent_columns = [
        "baseline_triage_accuracy",
        "quality_gated_triage_accuracy",
        "baseline_urgent_recall",
        "quality_gated_urgent_recall",
        "baseline_retake_rate",
        "quality_gated_retake_rate",
    ]
    count_columns = ["cases_total", "cases_urgent"]

    table = _select_columns(
        df[columns],
        group_column="group_value",
        rename_map=rename_map,
        percent_columns=percent_columns,
        count_columns=count_columns,
    )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render triage breakdown tables as Markdown.")
    parser.add_argument(
        "--breakdown-dir",
        type=Path,
        default=DEFAULT_BREAKDOWN_DIR,
        help="Directory containing breakdown CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination Markdown file (default: %(default)s)",
    )
    parser.add_argument(
        "--min-diagnosis-cases",
        type=int,
        default=50,
        help="Minimum case count to include a diagnosis (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    breakdown_dir = args.breakdown_dir

    skin_tone_csv = breakdown_dir / "triage_breakdown_by_skin_tone.csv"
    diagnosis_csv = breakdown_dir / "triage_breakdown_by_diagnosis.csv"
    defect_csv = breakdown_dir / "triage_breakdown_by_defect.csv"
    monk_tone_csv = breakdown_dir / "triage_breakdown_by_monk_skin_tone.csv"

    if not skin_tone_csv.exists():
        raise FileNotFoundError(f"Missing skin tone breakdown: {skin_tone_csv}")
    if not diagnosis_csv.exists():
        raise FileNotFoundError(f"Missing diagnosis breakdown: {diagnosis_csv}")
    if not defect_csv.exists():
        raise FileNotFoundError(f"Missing defect breakdown: {defect_csv}")

    skin_table = _build_skin_tone_table(skin_tone_csv)
    monk_table = _build_monk_table(monk_tone_csv) if monk_tone_csv.exists() else pd.DataFrame()
    diagnosis_table = _build_diagnosis_table(diagnosis_csv, args.min_diagnosis_cases)
    defect_table = _build_defect_table(defect_csv)

    sections: List[tuple[str, pd.DataFrame]] = [
        ("Fitzpatrick Skin Tone Performance", skin_table),
    ]
    if not monk_table.empty:
        sections.append(("Monk Skin Tone Performance", monk_table))
    sections.extend([
        ("Top Diagnoses (>={:,} Cases)".format(args.min_diagnosis_cases), diagnosis_table),
        ("Synthetic Defect Stress Test (Defect Present)", defect_table),
    ])

    lines: List[str] = [
        "# TeleDerm SnapCheck Triage Performance Tables",
        "",
        "All metrics computed on the 4,800-image evaluation set (baseline vs. quality-gated modes).",
        "",
    ]

    for title, table in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(_dataframe_to_markdown(table))
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote publication tables to {args.output}")


if __name__ == "__main__":
    main()
