"""Run baseline vs quality-gated teledermatology triage simulation."""
from __future__ import annotations

import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from snapcheck.config import load_config
from snapcheck.triage import run_triage_simulation
from snapcheck.reporting import save_confusion_matrix
from snapcheck.paths import REPORT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run triage simulation")
    parser.add_argument("--config", type=str, default="triage_eval.yaml", help="Triage evaluation config")
    return parser.parse_args()


def _load_thresholds_from_json(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    thresholds: Dict[str, float] = {}
    if isinstance(data, dict):
        payload = data.get("thresholds", data)
        if isinstance(payload, dict):
            for label, value in payload.items():
                if isinstance(value, dict):
                    value = value.get("threshold")
                if value is None:
                    continue
                thresholds[label] = float(value)
    return thresholds


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config).data
    cfg.setdefault("quality_dataset_dir", str(Path(cfg.get("quality_dataset_dir", "data/quality"))))

    project_root = Path(__file__).resolve().parents[1]
    quality_cfg = cfg.setdefault("quality_model", {})

    thresholds_path_value = quality_cfg.get("thresholds_path") or cfg.get("quality_thresholds_path")
    thresholds_path: Path | None = None
    if thresholds_path_value:
        thresholds_path = Path(thresholds_path_value)
    else:
        default_path = REPORT_DIR / "diqa" / "thresholds.json"
        if default_path.exists():
            thresholds_path = default_path

    if thresholds_path is not None and not thresholds_path.is_absolute():
        thresholds_path = (project_root / thresholds_path).resolve()

    if thresholds_path is not None:
        if thresholds_path.exists():
            try:
                thresholds = _load_thresholds_from_json(thresholds_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"Failed to load quality thresholds from {thresholds_path}: {exc}")
            else:
                if thresholds:
                    quality_cfg["thresholds"] = thresholds
                    quality_cfg["thresholds_path"] = str(thresholds_path)
                    print(f"Loaded quality thresholds from {thresholds_path}")
        else:
            print(f"Quality thresholds file {thresholds_path} not found; using values from config.")

    for backend_cfg in cfg.get("vlm_backends", []):
        backend_type = backend_cfg.get("type", "huggingface")
        if backend_type == "huggingface":
            backend_cfg.setdefault("temperature", 0.0)

    outputs_cfg = cfg.get("outputs", {})
    experiment = cfg.get("experiment_name", "triage_run")
    run_timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(outputs_cfg.get("run_root", REPORT_DIR / "triage" / "runs"))
    run_dir = run_root / experiment / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    detail_target = Path(outputs_cfg.get("summary_csv", REPORT_DIR / "triage" / "triage_detail.csv"))
    summary_target = Path(outputs_cfg.get("summary_backend_csv", REPORT_DIR / "triage" / "triage_summary.csv"))

    detail_df, summary_df = run_triage_simulation(cfg)

    detail_target.parent.mkdir(parents=True, exist_ok=True)
    detail_run_path = run_dir / detail_target.name
    detail_df.to_csv(detail_run_path, index=False)
    detail_df.to_csv(detail_target, index=False)
    print(f"Saved triage trace to {detail_run_path}")
    print(f"Updated latest detail CSV at {detail_target}")

    summary_path = summary_target
    if not summary_df.empty:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_run_path = run_dir / summary_path.name
        summary_df.to_csv(summary_run_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved backend summary to {summary_run_path}")
        print(f"Updated latest backend summary at {summary_path}")

        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        def fmt_pct(value: float | int | None) -> str:
            if value is None:
                return "n/a"
            if isinstance(value, (int, float)) and not pd.isna(value):
                return f"{value:.1%}"
            return "n/a"

        def fmt_float(value: float | int | None) -> str:
            if value is None:
                return "n/a"
            if isinstance(value, (int, float)) and not pd.isna(value):
                return f"{value:.2f}"
            return "n/a"

        history_rows = []
        for record in summary_df.to_dict("records"):
            backend = record.get("backend", "unknown")
            mode = record.get("mode", "baseline")
            retake = record.get("retake_rate")
            urgent_recall = record.get("urgent_recall")
            miss_rate = record.get("urgency_miss_rate")
            latency = record.get("mean_latency")
            tokens = record.get("mean_token_usage")
            print(
                f"[{backend}][{mode}] retake={fmt_pct(retake)} urgent_recall={fmt_pct(urgent_recall)} "
                f"miss={fmt_pct(miss_rate)} latency={fmt_float(latency)}s tokens={fmt_float(tokens)}"
            )
            history_rows.append(
                {
                    "timestamp": timestamp,
                    "experiment": experiment,
                    "config": args.config,
                    **record,
                }
            )

        history_path = REPORT_DIR / "triage" / "run_history.csv"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_df = pd.DataFrame(history_rows)
        if history_path.exists():
            existing = pd.read_csv(history_path)
            history_df = pd.concat([existing, history_df], ignore_index=True)
        history_df.to_csv(history_path, index=False)
        print(f"Appended run summary to {history_path}")
    else:
        print("No backend summary generated; check triage configuration.")

    if cfg.get("outputs", {}).get("confusion_matrices_dir"):
        cm_dir = Path(cfg["outputs"]["confusion_matrices_dir"])
        cm_dir.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix(detail_df, "ground_truth_triage", "model_triage", "Baseline triage", cm_dir)
        if "gated_triage" in detail_df.columns:
            save_confusion_matrix(detail_df, "ground_truth_triage", "gated_triage", "Quality gated triage", cm_dir)


if __name__ == "__main__":
    main()
