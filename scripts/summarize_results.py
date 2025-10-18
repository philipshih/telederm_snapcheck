"""Aggregate model and triage results into manuscript-ready tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from snapcheck.paths import REPORT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SnapCheck results")
    parser.add_argument("--triage", type=str, default="reports/triage/triage_summary.csv")
    parser.add_argument("--quality", type=str, default="reports/diqa/metrics.csv")
    parser.add_argument("--output", type=str, default="reports/summary/summary.xlsx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    triage_path = Path(args.triage)
    quality_path = Path(args.quality)
    output_path = Path(args.output)

    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
    if triage_path.exists():
        triage_df = pd.read_csv(triage_path)
        triage_df.to_excel(writer, sheet_name="triage", index=False)
    if quality_path.exists():
        quality_df = pd.read_csv(quality_path)
        quality_df.to_excel(writer, sheet_name="quality", index=False)
    writer.close()
    print(f"Wrote summary workbook to {output_path}")


if __name__ == "__main__":
    main()
