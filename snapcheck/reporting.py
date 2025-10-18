from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .paths import REPORT_DIR


def save_confusion_matrix(df: pd.DataFrame, column_true: str, column_pred: str, title: str, output_dir: Path) -> Path:
    matrix = pd.crosstab(df[column_true], df[column_pred])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"confusion_{column_true}_{column_pred}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_metrics_table(metrics: Dict[str, float], output_path: Path) -> Path:
    df = pd.DataFrame(metrics.items(), columns=["metric", "value"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


__all__ = ["save_confusion_matrix", "save_metrics_table"]
