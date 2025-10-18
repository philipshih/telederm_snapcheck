from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, x, y, width, height, text, facecolor, edgecolor="#1f2933", loc="center", text_kwargs=None):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    if text_kwargs is None:
        text_kwargs = {}
    text_kwargs = {
        "ha": "center",
        "va": "center",
        "fontsize": 12,
        "color": "#102a43",
        **text_kwargs,
    }
    if loc == "center":
        tx = x + width / 2.0
        ty = y + height / 2.0
    elif loc == "topleft":
        tx = x + 0.02
        ty = y + height - 0.02
        text_kwargs["ha"] = "left"
        text_kwargs["va"] = "top"
    else:
        tx = x + width / 2.0
        ty = y + height / 2.0
    ax.text(tx, ty, text, **text_kwargs)
    return box


def add_arrow(ax, start, end, connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=1.4,
        color="#1f2933",
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)
    return arrow


def main():
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "TeleDerm SnapCheck Workflow & Status",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.5,
        0.89,
        "Pipeline, evaluation focus, and current progress snapshot",
        ha="center",
        va="center",
        fontsize=12,
        color="#334155",
    )

    pipeline_color = "#dbeafe"
    highlight_color = "#fde68a"
    stage_color = "#e5e7eb"
    bottleneck_color = "#fee2e2"

    pipeline_boxes = [
        (
            0.05,
            0.72,
            0.18,
            0.12,
            "Public skin-lesion datasets\nISIC, HAM10000, Derm7pt",
        ),
        (
            0.28,
            0.72,
            0.18,
            0.12,
            "Synthetic quality augmentation\nProgrammatic blur/contrast/lighting\n+ auto labels",
        ),
        (
            0.51,
            0.72,
            0.18,
            0.12,
            "Train quality classifier\nViT-S, multi-label\nval AUROC 0.968",
        ),
        (
            0.74,
            0.72,
            0.18,
            0.12,
            "Per-defect quality gate\nThresholded pass vs retake scores",
        ),
    ]

    for x, y, w, h, text in pipeline_boxes:
        add_box(ax, x, y, w, h, text, facecolor=pipeline_color)

    for idx in range(len(pipeline_boxes) - 1):
        x, y, w, h, _ = pipeline_boxes[idx]
        nx, ny, nw, nh, _ = pipeline_boxes[idx + 1]
        start = (x + w, y + h / 2.0)
        end = (nx, ny + nh / 2.0)
        add_arrow(ax, start, end)

    add_box(
        ax,
        0.3,
        0.46,
        0.22,
        0.14,
        "Triage simulation harness\nQwen2-VL-2B VLM\nBaseline vs quality-gated flows\n721-image evaluation",
        facecolor=pipeline_color,
    )
    add_arrow(
        ax,
        (pipeline_boxes[-1][0] + pipeline_boxes[-1][2] / 2.0, pipeline_boxes[-1][1]),
        (0.3 + 0.22 / 2.0, 0.46 + 0.14),
        connectionstyle="arc3,rad=-0.25",
    )

    add_box(
        ax,
        0.58,
        0.46,
        0.22,
        0.14,
        "Safety, fairness, efficiency metrics\nUrgent recall, miss rate\nRetake rate, latency, tokens",
        facecolor=pipeline_color,
    )
    add_arrow(
        ax,
        (0.3 + 0.22, 0.46 + 0.14 / 2.0),
        (0.58, 0.46 + 0.14 / 2.0),
    )

    ax.text(
        0.5,
        0.60,
        "Test: Does automated image-quality gating\nlower urgent misses without overwhelming retakes\nwhile holding efficiency steady?",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#7c2d12",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=highlight_color,
            edgecolor="#f59e0b",
            linewidth=1.5,
        ),
    )

    stage_text = (
        "Current stage\n"
        "- Quality model trained; fairness slices logged\n"
        "- Threshold manifest saved (reports/diqa/thresholds.json)\n"
        "- First triage pilot cached (reports/triage/*)"
    )
    add_box(
        ax,
        0.05,
        0.16,
        0.42,
        0.18,
        stage_text,
        facecolor=stage_color,
        edgecolor="#1f2933",
        loc="topleft",
        text_kwargs={"fontsize": 11, "color": "#111827"},
    )

    bottleneck_text = (
        "Current bottleneck\n"
        "- Quality gate triggers 50% retakes (triage_summary_gpt5nano.csv)\n"
        "- Gated urgent miss rate = 0.69 vs baseline 0.35\n"
        "- Retake logic now simulates successful resubmission of a 'pass' image.\n"
        "- Need threshold tuning + larger evaluation batches"
    )
    add_box(
        ax,
        0.53,
        0.16,
        0.42,
        0.18,
        bottleneck_text,
        facecolor=bottleneck_color,
        edgecolor="#b91c1c",
        loc="topleft",
        text_kwargs={"fontsize": 11, "color": "#111827"},
    )

    ax.text(
        0.5,
        0.08,
        "Artifacts referenced: models/snapcheck_quality.pt | reports/diqa/*.csv | reports/triage/*",
        ha="center",
        va="center",
        fontsize=10,
        color="#475569",
    )

    output_path = output_dir / "telederm_snapcheck_workflow.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
