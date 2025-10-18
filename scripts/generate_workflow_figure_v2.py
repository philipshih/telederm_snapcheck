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
    add_box(ax, 0.05, 0.65, 0.9, 0.2, "", facecolor=stage_color)  # Container
    ax.text(0.5, 0.8, "1. Model Training: Build a Dermatology Image Quality Classifier",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#102a43")
    add_box(
        ax, 0.1, 0.68, 0.35, 0.1,
        "Input: Public datasets (ISIC, HAM10000)\n+ Synthetic quality defects (blur, lighting)",
        facecolor="white", loc="center", text_kwargs={"fontsize": 10}
    )
    add_box(
        ax, 0.55, 0.68, 0.35, 0.1,
        "Output: ViT-Small model (AUROC 0.968)\nDetects 5 types of image quality issues",
        facecolor="white", loc="center", text_kwargs={"fontsize": 10}
    )

    # --- Stage 2: Triage Simulation ---
    add_box(ax, 0.05, 0.35, 0.9, 0.2, "", facecolor=stage_color)  # Container
    ax.text(0.5, 0.5, "2. Triage Simulation: Test the Quality Gate's Impact",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#102a43")
    add_box(
        ax, 0.1, 0.38, 0.35, 0.1,
        "Control Arm: VLM Triage only\n(Qwen2-VL-2B)",
        facecolor="white", loc="center", text_kwargs={"fontsize": 10}
    )
    add_box(
        ax, 0.55, 0.38, 0.35, 0.1,
        "Test Arm: SnapCheck Quality Gate\n+ VLM Triage",
        facecolor="white", loc="center", text_kwargs={"fontsize": 10}
    )

    # --- Stage 3: Results & Next Steps ---
    add_box(ax, 0.05, 0.05, 0.9, 0.2, "", facecolor=stage_color)  # Container
    ax.text(0.5, 0.2, "3. Key Finding & Next Step",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#102a43")
    add_box(
        ax, 0.1, 0.08, 0.35, 0.1,
        "Finding: Gate is too strict (50% retake rate),\nincreasing urgent miss rate from 0.35 to 0.69.",
        facecolor=bottleneck_color, loc="center", text_kwargs={"fontsize": 10}
    )
    add_box(
        ax, 0.55, 0.08, 0.35, 0.1,
        "Next Step: Tune quality thresholds\nto balance safety and clinical throughput.",
        facecolor=highlight_color, loc="center", text_kwargs={"fontsize": 10}
    )

    # Arrows connecting stages
    add_arrow(ax, (0.5, 0.65), (0.5, 0.55))
    add_arrow(ax, (0.5, 0.35), (0.5, 0.25))

    output_path = output_dir / "telederm_snapcheck_workflow_v3.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Workflow figure saved to {output_path}")


if __name__ == "__main__":
    main()
