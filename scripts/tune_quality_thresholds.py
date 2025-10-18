"""Tune per-label quality thresholds on the validation set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from snapcheck.paths import QUALITY_DATA_DIR, REPORT_DIR
from snapcheck.quality_model import QualityDataset, load_quality_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune quality gating thresholds")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/snapcheck_quality.pt"))
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "diqa" / "thresholds.json")
    parser.add_argument("--strategy", choices=["youden"], default="youden")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def _resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (Path.cwd() / candidate)


def _compute_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    strategy: str,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(label_names):
        y_true = labels[:, idx]
        y_score = probs[:, idx]
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        if strategy == "youden":
            j = tpr - fpr
            best = int(np.nanargmax(j))
        else:  # pragma: no cover - future strategies
            best = int(np.nanargmax(tpr - fpr))
        threshold = float(thresholds[best])
        if np.isinf(threshold):  # handle edge case when positives absent
            threshold = 0.5
        preds = (y_score >= threshold).astype(np.int32)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        results[name] = {
            "threshold": float(threshold),
            "recall": float(recall),
            "specificity": float(specificity),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    return results


def main() -> None:
    args = parse_args()
    model, config, label_columns = load_quality_checkpoint(args.checkpoint, device=args.device)
    dataset_root = _resolve_path(config.get("quality_dataset_dir", QUALITY_DATA_DIR))
    val_manifest = _resolve_path(config.get("val_csv", dataset_root / "val_manifest.csv"))

    dataset = QualityDataset(val_manifest, dataset_root, label_columns, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["pixel_values"].to(next(model.parameters()).device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["labels"].numpy())
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    results = _compute_thresholds(probs, labels, label_columns, strategy=args.strategy)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"strategy": args.strategy, "thresholds": results}, indent=2))

    print("Recommended thresholds (strategy: {}):".format(args.strategy))
    for name, stats in results.items():
        print(f"  {name:15s} threshold={stats['threshold']:.3f} recall={stats['recall']:.3f} specificity={stats['specificity']:.3f}")


if __name__ == "__main__":
    main()
