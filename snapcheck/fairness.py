from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score


@dataclass
class FairnessMetric:
    metric: str
    group: str
    values: Dict[str, float]
    overall: float


def compute_group_metrics(
    manifest: pd.DataFrame,
    probabilities: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    group_column: str,
    metric: str,
) -> FairnessMetric:
    group_values = manifest[group_column].fillna("unknown")
    unique_groups = sorted(group_values.unique())
    group_scores: Dict[str, float] = {}
    for group in unique_groups:
        idx = group_values == group
        if idx.sum() < 5:
            group_scores[group] = float("nan")
            continue
        if metric == "auroc":
            try:
                score = np.nanmean([
                    roc_auc_score(labels[idx, j], probabilities[idx, j])
                    for j in range(len(label_names))
                ])
            except ValueError:
                score = float("nan")
        elif metric == "recall":
            score = np.nanmean([
                recall_score(labels[idx, j] > 0.5, probabilities[idx, j] > 0.5)
                for j in range(len(label_names))
            ])
        elif metric == "precision":
            score = np.nanmean([
                precision_score(labels[idx, j] > 0.5, probabilities[idx, j] > 0.5)
                for j in range(len(label_names))
            ])
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        group_scores[group] = float(score)
    overall = np.nanmean(list(group_scores.values()))
    return FairnessMetric(metric=metric, group=group_column, values=group_scores, overall=float(overall))


__all__ = ["FairnessMetric", "compute_group_metrics"]
