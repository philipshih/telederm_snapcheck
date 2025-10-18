#!/usr/bin/env python
"""Generate publication-ready figures for the TeleDerm SnapCheck manuscript.

Outputs:
    reports/figures/figure2_overall_gains.png
    reports/figures/figure3_defect_impact.png
    reports/figures/figure4_skin_tone.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_summary() -> pd.DataFrame:
    summary = pd.read_csv("reports/triage/offline_gate_summary_test_pairs.csv")
    summary = summary[summary["mode"].isin(["baseline", "gated"])].set_index("mode")
    return summary


def figure_overall(summary: pd.DataFrame) -> Path:
    metrics = {
        "Urgent miss rate": "urgency_miss_rate",
        "Urgent recall": "urgent_recall",
        "Urgent deferral rate": "urgent_deferral_rate",
        "Retake rate": "retake_rate",
    }
    modes = ["baseline", "gated"]
    data = summary.loc[modes, metrics.values()] * 100  # convert to percentages

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width / 2, data.loc["baseline"], width, label="Baseline", color="#94a3b8")
    ax.bar(x + width / 2, data.loc["gated"], width, label="SnapCheck gated", color="#2563eb")

    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), rotation=15)
    ax.set_ylim(0, max(data.max()) * 1.25)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title("Overall triage outcomes on 1,344 paired exposures")

    for idx, mode in enumerate(modes):
        for jdx, value in enumerate(data.loc[mode]):
            ax.text(
                jdx + (idx - 0.5) * width,
                value + 0.8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    output = FIG_DIR / "figure2_overall_gains.png"
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def figure_defect_impacts() -> Path:
    df = pd.read_csv("reports/triage/breakdowns_gpt5nano_test/triage_breakdown_by_defect.csv")
    df = df[df["group_type"] == "defect"].copy()

    def pick_present(group: pd.DataFrame) -> pd.Series:
        present = group[group["defect_state"] == "present"]
        row = present.iloc[0] if not present.empty else group.iloc[0]
        return row

    df = df.groupby("group_value", as_index=False).apply(pick_present).reset_index(drop=True)

    df = df.sort_values("baseline_urgency_miss_rate", ascending=False).head(9)
    df.rename(
        columns={
            "group_value": "defect",
            "baseline_urgency_miss_rate": "baseline_miss",
            "quality_gated_urgency_miss_rate": "gated_miss",
            "quality_gated_urgent_deferral_rate": "gated_deferral",
        },
        inplace=True,
    )

    width = 0.25
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width,
        df["baseline_miss"] * 100,
        width,
        label="Baseline miss",
        color="#94a3b8",
    )
    ax.bar(
        x,
        df["gated_miss"] * 100,
        width,
        label="SnapCheck miss",
        color="#2563eb",
    )
    ax.bar(
        x + width,
        df["gated_deferral"] * 100,
        width,
        label="SnapCheck deferral",
        color="#ef4444",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["defect"], rotation=25, ha="right")
    ax.set_ylabel("Percentage of urgent cases (%)")
    ax.set_title("Miss and deferral rates for major defects")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for idx, row in enumerate(df.itertuples()):
        ax.text(idx - width, row.baseline_miss * 100 + 1, f"{row.baseline_miss*100:.1f}", ha="center", fontsize=8)
        ax.text(idx, row.gated_miss * 100 + 1, f"{row.gated_miss*100:.1f}", ha="center", fontsize=8)
        ax.text(idx + width, row.gated_deferral * 100 + 1, f"{row.gated_deferral*100:.1f}", ha="center", fontsize=8)

    fig.tight_layout()
    output = FIG_DIR / "figure3_defect_impact.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def figure_skin_tone() -> Path:
    df = pd.read_csv("reports/triage/breakdowns_gpt5nano_test/triage_breakdown_by_monk_skin_tone.csv")
    groups = [
        ("MST 1-3 (lighter)", "MST 1-3 (lighter)"),
        ("MST 4-7 (medium)", "MST 4-7 (medium)"),
        ("MST 8-10 (darker)", "MST 8-10 (darker)"),
    ]
    subset = df[df["group_value"].isin([g for g, _ in groups])].copy()
    subset = subset.set_index("group_value")

    labels = [label for _, label in groups]
    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(
        x - width / 2,
        subset.loc[[g for g, _ in groups], "baseline_urgency_miss_rate"] * 100,
        width,
        label="Baseline miss",
        color="#94a3b8",
    )
    ax.bar(
        x + width / 2,
        subset.loc[[g for g, _ in groups], "quality_gated_urgency_miss_rate"] * 100,
        width,
        label="SnapCheck miss",
        color="#2563eb",
    )
    ax.set_ylabel("Urgent miss rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Urgent miss rate by Monk Skin Tone grouping")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    output = FIG_DIR / "figure4_skin_tone.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main() -> None:
    summary = load_summary()
    outputs = [
        figure_overall(summary),
        figure_defect_impacts(),
        figure_skin_tone(),
    ]
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
