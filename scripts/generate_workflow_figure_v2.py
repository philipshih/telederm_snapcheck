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
        0.95,
        "TeleDerm SnapCheck: Core Workflow",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#0f172a",
    )

    # Colors
    stage_color = "#dbeafe"
    highlight_color = "#fde68a"
    bottleneck_color = "#fee2e2"

    # --- Stage 1: Model Training ---
    add_box(ax, 0.05, 0.62, 0.9, 0.2, "", facecolor=stage_color)  # Container
    ax.text(0.5, 0.79, "1. Train a Dermatology Image Quality Classifier",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#102a43")
    add_box(
        ax, 0.1, 0.65, 0.37, 0.11,
        "Input: ISIC, HAM10000, Derm7pt\n+ synthetic degradations (blur, lighting, framing)",
        facecolor="white", loc="center", text_kwargs={"fontsize": 10}
    )
    add_box(
        ax, 0.53, 0.65, 0.37, 0.11,
        "Output: ViT-Small DIQA model (AUROC 0.968)\nScores 11 quality heads + overall failure",
        facecolor="white", loc="center", text_kwargs={"fontsize": 10}
    )

    # --- Stage 2: Triage Simulation ---
    add_box(ax, 0.05, 0.33, 0.9, 0.2, "", facecolor=stage_color)  # Container
    ax.text(0.5, 0.5, "2. Simulate GPT-5 Nano Triage With and Without SnapCheck",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#102a43")
    add_box(
        ax, 0.1, 0.36, 0.37, 0.11,
        "Baseline arm: GPT-5 Nano triage only\n(4,800 cases; urgent recall 73.2%)",
        facecolor="white", loc="center", text_kwargs={"fontsize": 10}
    )
    add_box(
        ax, 0.53, 0.36, 0.37, 0.11,
        "Quality-gated arm: SnapCheck + GPT-5 Nano\n(Urgent recall 75.9%, retake rate 18.9%)",
        facecolor="white", loc="center", text_kwargs={"fontsize": 10}
    )

    # --- Stage 3: Results & Next Steps ---
    add_box(ax, 0.05, 0.04, 0.9, 0.2, "", facecolor=stage_color)  # Container
    ax.text(0.5, 0.23, "3. Findings and Forward Plan",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#102a43")
    add_box(
        ax, 0.1, 0.07, 0.37, 0.11,
        "Finding: Calibrated thresholds improve\nurgent recall (+2.7 pp) at 17.8% deferrals\nand 18.9% retakes; blur/motion blur drive lift.",
        facecolor=bottleneck_color, loc="center", text_kwargs={"fontsize": 10, "linespacing": 1.3}
    )
    add_box(
        ax, 0.53, 0.07, 0.37, 0.11,
        "Next steps: Validate with real uploads,\nrefine defect-specific policies,\nand deliver clinician-facing retake guidance.",
        facecolor=highlight_color, loc="center", text_kwargs={"fontsize": 10, "linespacing": 1.3}
    )

    # Arrows connecting stages
    add_arrow(ax, (0.5, 0.62), (0.5, 0.53))
    add_arrow(ax, (0.5, 0.33), (0.5, 0.25))

    output_path = output_dir / "telederm_snapcheck_workflow_v3.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Workflow figure saved to {output_path}")


if __name__ == "__main__":
    main()
