#!/usr/bin/env python
"""Compute triage miss-rate breakdowns for TeleDerm SnapCheck.

This utility merges the triage detail CSV with the quality manifest to quantify
urgent miss rates (and related metrics) across diagnoses and injected synthetic
defects. Metric definitions mirror `snapcheck.triage.compute_triage_metrics` so
that downstream analyses stay aligned with the simulator.

Example:
    python scripts/analyze_triage_breakdowns.py \
        --triage-detail reports/triage/triage_detail.csv \
        --manifest data/quality/manifest.csv \
        --output-dir reports/triage/breakdowns
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

# Ensure `snapcheck` is importable when executed as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snapcheck.quality_model import TARGET_LABELS


DEFAULT_TRIAGE_DETAIL = Path("reports/triage/triage_detail.csv")
DEFAULT_MANIFEST = Path("data/quality/manifest.csv")
DEFAULT_OUTPUT_DIR = Path("reports/triage/breakdowns")

FITZPATRICK_LABELS: Dict[int, str] = {
    0: "Type I (pale ivory)",
    1: "Type II (fair beige)",
    2: "Type III (light brown)",
    3: "Type IV (medium brown)",
    4: "Type V (dark brown)",
    5: "Type VI (deeply pigmented)",
}


def _wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    confidence = max(min(confidence, 0.999), 0.001)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    phat = successes / n
    denom = 1 + (z ** 2) / n
    center = (phat + (z ** 2) / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + (z ** 2) / (4 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SnapCheck miss rates by diagnosis and defect.")
    parser.add_argument(
        "--triage-detail",
        type=Path,
        default=DEFAULT_TRIAGE_DETAIL,
        help="CSV exported by `run_triage_simulation.py` (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Quality manifest CSV with synthetic defect metadata (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for breakdown tables (default: %(default)s)",
    )
    parser.add_argument(
        "--modes",
        choices=["baseline", "gated", "both"],
        default="both",
        help="Which triage outputs to analyse (default: %(default)s)",
    )
    parser.add_argument(
        "--skin-tone-overrides",
        type=Path,
        default=Path("data/fairness/fitzpatrick_overrides.csv"),
        help="Optional CSV with manual Fitzpatrick/Monk annotations (default: %(default)s)",
    )
    return parser.parse_args()


def _normalize_path(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text.replace("\\", "/")


def _normalize_label(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def _normalize_triage_value(value: object) -> Tuple[str, bool]:
    if value is None:
        return "unknown", False
    text = str(value).strip().lower()
    if not text or text in {"unknown", "n/a"}:
        return "unknown", False
    if text in {"retake", "request_retake"}:
        return "retake", True
    return text, False


def _load_inputs(triage_path: Path, manifest_path: Path, overrides_path: Optional[Path]) -> pd.DataFrame:
    if not triage_path.exists():
        raise FileNotFoundError(f"Triage detail CSV not found: {triage_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    triage_df = pd.read_csv(triage_path)
    manifest_df = pd.read_csv(manifest_path)

    triage_df = triage_df.rename(columns={"image_path": "image_path_detail"}).copy()
    manifest_df = manifest_df.rename(columns={"image_path": "manifest_image_path"}).copy()

    triage_df["image_key"] = triage_df["image_path_detail"].map(_normalize_path)
    manifest_df["image_key"] = manifest_df["manifest_image_path"].map(_normalize_path)

    merged = triage_df.merge(manifest_df, on="image_key", how="left", suffixes=("", "_manifest"))
    missing_manifest = merged["manifest_image_path"].isna().sum()
    if missing_manifest:
        print(f"[warn] Missing manifest metadata for {missing_manifest} triage rows.")

    if overrides_path and overrides_path.exists():
        overrides = pd.read_csv(overrides_path)
        if "image_path" not in overrides.columns:
            raise ValueError("Overrides CSV must include an 'image_path' column.")
        overrides = overrides.copy()
        overrides["image_key"] = overrides["image_path"].map(_normalize_path)
        overrides = overrides.drop_duplicates("image_key", keep="last")
        merged = merged.merge(
            overrides.drop(columns=["image_path"]),
            on="image_key",
            how="left",
            suffixes=("", "_override"),
        )
        for col in ("fitzpatrick_type", "monk_skin_tone"):
            override_col = f"{col}_override"
            if override_col in merged.columns:
                merged[col] = merged[override_col].combine_first(merged.get(col))
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_override")])
    elif overrides_path:
        print(f"[info] Skin-tone override file not found at {overrides_path}; skipping overrides.")

    return merged


def _resolve_modes(requested: str, df: pd.DataFrame) -> List[Tuple[str, str]]:
    mode_map = {
        "baseline": ("baseline", "model_triage"),
        "gated": ("quality_gated", "gated_triage"),
    }
    requested_modes: Sequence[str]
    if requested == "both":
        requested_modes = ("baseline", "gated")
    else:
        requested_modes = (requested,)

    resolved: List[Tuple[str, str]] = []
    for mode in requested_modes:
        prefix, column = mode_map[mode]
        if column not in df.columns:
            if requested == mode:
                raise ValueError(f"Triage column '{column}' not found; cannot compute requested mode '{mode}'.")
            print(f"[warn] Skipping '{mode}' metrics: column '{column}' not present.")
            continue
        resolved.append((prefix, column))

    if not resolved:
        raise ValueError("No triage modes available to analyse. Check the input CSV columns.")
    return resolved


def _compute_mode_metrics(df: pd.DataFrame, triage_col: str) -> Dict[str, float]:
    if df.empty:
        return {
            "triage_accuracy": float("nan"),
            "triaged_cases": 0,
            "triaged_case_fraction": float("nan"),
            "urgent_recall": float("nan"),
            "urgency_miss_rate": float("nan"),
            "urgent_deferral_rate": float("nan"),
            "retake_rate": float("nan"),
            "mean_latency": float("nan"),
            "mean_token_usage": float("nan"),
        }

    if triage_col not in df.columns:
        raise KeyError(f"Column '{triage_col}' not found in dataframe.")

    truth = _normalize_label(df["ground_truth_triage"])
    predictions_raw = df[triage_col]

    labels_list: List[str] = []
    retake_flags_list: List[bool] = []
    quality_fail = df.get("quality_fail")
    for idx, value in enumerate(predictions_raw):
        label, normalized_retake = _normalize_triage_value(value)
        if normalized_retake:
            is_retake = True
        elif triage_col != "model_triage":
            is_retake = bool(quality_fail.iloc[idx]) if quality_fail is not None else False
        else:
            is_retake = False
        labels_list.append(label)
        retake_flags_list.append(is_retake)
    labels = pd.Series(labels_list, index=df.index)
    retake_flags = pd.Series(retake_flags_list, index=df.index)

    total_cases = len(df)
    total_known_truth = int((truth != "unknown").sum())

    valid_mask = (truth != "unknown") & (~labels.isin({"unknown", "retake"}))
    triaged_cases = int(valid_mask.sum())
    if triaged_cases:
        accuracy = float((truth[valid_mask] == labels[valid_mask]).mean())
    else:
        accuracy = float("nan")

    triaged_fraction = triaged_cases / max(1, total_known_truth)

    urgent_truth_mask = truth == "urgent"
    urgent_total = int(urgent_truth_mask.sum())
    if urgent_total:
        urgent_tp = int(((labels == "urgent") & urgent_truth_mask).sum())
        urgent_deferrals = int((urgent_truth_mask & retake_flags).sum())
        urgent_miss = max(urgent_total - urgent_tp - urgent_deferrals, 0)
        urgent_recall = urgent_tp / urgent_total
        urgent_miss = urgent_miss / urgent_total
        urgent_deferral_rate = urgent_deferrals / urgent_total
    else:
        urgent_recall = float("nan")
        urgent_miss = float("nan")
        urgent_deferral_rate = float("nan")

    retake_rate = float(retake_flags.mean()) if total_cases else float("nan")

    latency = pd.to_numeric(df.get("latency", pd.Series(dtype=float)), errors="coerce")
    mean_latency = float(latency.mean()) if not latency.empty else float("nan")

    token_usage = pd.to_numeric(df.get("token_usage", pd.Series(dtype=float)), errors="coerce")
    token_mean = float(token_usage.mean()) if not token_usage.empty else float("nan")

    return {
        "triage_accuracy": accuracy,
        "triaged_cases": triaged_cases,
        "triaged_case_fraction": triaged_fraction,
        "urgent_recall": urgent_recall,
        "urgency_miss_rate": urgent_miss,
        "urgent_deferral_rate": urgent_deferral_rate,
        "retake_rate": retake_rate,
        "mean_latency": mean_latency,
        "mean_token_usage": token_mean,
    }


def _summarise_subset(df: pd.DataFrame, modes: Iterable[Tuple[str, str]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    summary["cases_total"] = int(len(df))

    truth = _normalize_label(df["ground_truth_triage"])
    summary["cases_known_truth"] = int((truth != "unknown").sum())
    summary["cases_urgent"] = int((truth == "urgent").sum())

    for prefix, column in modes:
        metrics = _compute_mode_metrics(df, column)
        summary.update({f"{prefix}_{key}": value for key, value in metrics.items()})
    return summary


def _build_diagnosis_rows(df: pd.DataFrame, modes: Iterable[Tuple[str, str]]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    records.append(
        {
            "group_type": "diagnosis",
            "group_value": "all",
            **_summarise_subset(df, modes),
        }
    )

    df = df.copy()
    df["ground_truth_label"] = df["ground_truth"].fillna("unknown").astype(str)

    for diagnosis, group in df.groupby("ground_truth_label"):
        record = {
            "group_type": "diagnosis",
            "group_value": diagnosis,
            **_summarise_subset(group, modes),
        }
        records.append(record)
    return records


def _build_defect_rows(df: pd.DataFrame, modes: Iterable[Tuple[str, str]]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    records.append(
        {
            "group_type": "defect",
            "group_value": "all",
            "defect_state": "all",
            **_summarise_subset(df, modes),
        }
    )

    defect_labels = list(TARGET_LABELS)
    for label in defect_labels:
        if label not in df.columns:
            print(f"[warn] Manifest column '{label}' missing; skipping defect analysis for that label.")
            continue

        for state_value, state_name in ((1, "present"), (0, "absent")):
            subset = df[df[label] == state_value]
            if subset.empty:
                continue
            record = {
                "group_type": "defect",
                "group_value": label,
                "defect_state": state_name,
                **_summarise_subset(subset, modes),
            }
            records.append(record)

        unknown_subset = df[df[label].isna()]
        if not unknown_subset.empty:
            record = {
                "group_type": "defect",
                "group_value": label,
                "defect_state": "unknown",
                **_summarise_subset(unknown_subset, modes),
            }
            records.append(record)
    return records


def _format_skin_tone_label(bin_value: Union[int, float, None]) -> str:
    if bin_value is None or pd.isna(bin_value):
        return "Unknown"
    try:
        key = int(bin_value)
    except (TypeError, ValueError):
        return str(bin_value)
    return FITZPATRICK_LABELS.get(key, f"Type {key}")


def _build_skin_tone_rows(df: pd.DataFrame, modes: Iterable[Tuple[str, str]]) -> List[Dict[str, object]]:
    col_name = "fitzpatrick_type" if "fitzpatrick_type" in df.columns else "skin_tone_bin"
    if col_name not in df.columns:
        print("[warn] Column 'fitzpatrick_type' missing; skipping Fitzpatrick breakdown.")
        return []

    records: List[Dict[str, object]] = []
    records.append(
        {
            "group_type": "skin_tone",
            "group_value": "all",
            "skin_tone_label": "All tones",
            **_summarise_subset(df, modes),
        }
    )

    grouped = df.groupby(col_name, dropna=False)
    for bin_value, subset in grouped:
        label = _format_skin_tone_label(bin_value)
        record = {
            "group_type": "skin_tone",
            "group_value": "unknown" if pd.isna(bin_value) else int(bin_value),
            "skin_tone_label": label,
            **_summarise_subset(subset, modes),
        }
        records.append(record)
    return records


def _build_monk_rows(df: pd.DataFrame, modes: Iterable[Tuple[str, str]]) -> List[Dict[str, object]]:
    if "monk_skin_tone" not in df.columns:
        return []

    records: List[Dict[str, object]] = []
    records.append(
        {
            "group_type": "monk_skin_tone",
            "group_value": "all",
            "monk_label": "All tones",
            **_summarise_subset(df, modes),
        }
    )

    def monk_label(value: Union[int, float, None]) -> str:
        if pd.isna(value) or value in {-1, None}:
            return "Unknown"
        return f"MST {int(value)}"

    grouped = df.groupby("monk_skin_tone", dropna=False)
    for tone_value, subset in grouped:
        record = {
            "group_type": "monk_skin_tone",
            "group_value": "unknown" if pd.isna(tone_value) else int(tone_value),
            "monk_label": monk_label(tone_value),
            **_summarise_subset(subset, modes),
        }
        records.append(record)

    buckets = [
        ("MST 1-3 (lighter)", [1, 2, 3]),
        ("MST 4-7 (medium)", [4, 5, 6, 7]),
        ("MST 8-10 (darker)", [8, 9, 10]),
    ]
    for label, values in buckets:
        subset = df[df["monk_skin_tone"].isin(values)]
        if subset.empty:
            continue
        records.append(
            {
                "group_type": "monk_skin_tone",
                "group_value": label,
                "monk_label": label,
                **_summarise_subset(subset, modes),
            }
        )

    return records


def _build_combo_rows(
    df: pd.DataFrame,
    modes: Iterable[Tuple[str, str]],
    min_count: int = 5,
    max_rows: int = 50,
) -> List[Dict[str, object]]:
    if "quality_fail" not in df.columns:
        return []

    fail_df = df[df["quality_fail"]].copy()
    fail_df["combo_key"] = (
        fail_df.get("quality_fail_labels", "")
        .fillna("")
        .astype(str)
        .apply(lambda text: "+".join(sorted(filter(None, text.split(";")))))
    )
    fail_df.loc[fail_df["combo_key"] == "", "combo_key"] = "unspecified"

    records: List[Dict[str, object]] = []
    records.append(
        {
            "group_type": "fail_combo",
            "group_value": "all_failures",
            **_summarise_subset(fail_df, modes),
        }
    )

    counts = fail_df["combo_key"].value_counts()
    top_combos = counts[counts >= min_count].index.tolist()
    if len(top_combos) > max_rows:
        top_combos = top_combos[:max_rows]

    for combo in top_combos:
        subset = fail_df[fail_df["combo_key"] == combo]
        record = {
            "group_type": "fail_combo",
            "group_value": combo,
            "cases_combo": int(len(subset)),
            **_summarise_subset(subset, modes),
        }
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    merged = _load_inputs(args.triage_detail, args.manifest, args.skin_tone_overrides)
    modes = _resolve_modes(args.modes, merged)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnosis_rows = _build_diagnosis_rows(merged, modes)
    defect_rows = _build_defect_rows(merged, modes)
    combo_rows = _build_combo_rows(merged, modes)
    skin_tone_rows = _build_skin_tone_rows(merged, modes)
    monk_rows = _build_monk_rows(merged, modes)

    diagnosis_path = output_dir / "triage_breakdown_by_diagnosis.csv"
    defect_path = output_dir / "triage_breakdown_by_defect.csv"
    combo_path = output_dir / "triage_breakdown_by_fail_combo.csv"
    skin_tone_path = output_dir / "triage_breakdown_by_skin_tone.csv"
    monk_path = output_dir / "triage_breakdown_by_monk_skin_tone.csv"

    pd.DataFrame(diagnosis_rows).to_csv(diagnosis_path, index=False)
    pd.DataFrame(defect_rows).to_csv(defect_path, index=False)
    pd.DataFrame(combo_rows).to_csv(combo_path, index=False)
    if skin_tone_rows:
        pd.DataFrame(skin_tone_rows).to_csv(skin_tone_path, index=False)
    if monk_rows:
        pd.DataFrame(monk_rows).to_csv(monk_path, index=False)

    print(f"Wrote diagnosis breakdown to {diagnosis_path}")
    print(f"Wrote defect breakdown to {defect_path}")
    print(f"Wrote fail-combo breakdown to {combo_path}")
    if skin_tone_rows:
        print(f"Wrote skin-tone breakdown to {skin_tone_path}")
    if monk_rows:
        print(f"Wrote Monk skin-tone breakdown to {monk_path}")


if __name__ == "__main__":
    main()
