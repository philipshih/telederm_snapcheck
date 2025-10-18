"""Compute quality gate retake rates using validation manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from snapcheck.quality_model import QualityDataset, load_quality_checkpoint
from snapcheck.paths import QUALITY_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SnapCheck retake rates")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/snapcheck_quality.pt"))
    parser.add_argument("--thresholds", type=Path, default=Path("reports/diqa/thresholds.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/quality/val_manifest.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_thresholds(path: Path, label_columns: List[str]) -> Dict[str, float]:
    data = json.loads(path.read_text())
    thresholds = data.get("thresholds", {})
    return {
        label: float(thresholds.get(label, {}).get("threshold", 0.5))
        for label in label_columns
    }


def main() -> None:
    args = parse_args()
    model, config, label_columns = load_quality_checkpoint(args.checkpoint, device=args.device)
    thresholds = load_thresholds(args.thresholds, label_columns)

    dataset_root = Path(config.get("quality_dataset_dir", QUALITY_DATA_DIR))
    manifest_path = args.manifest if args.manifest.is_absolute() else Path.cwd() / args.manifest
    dataset = QualityDataset(manifest_path, dataset_root, label_columns, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = next(model.parameters()).device
    model.eval()
    flags: List[bool] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["pixel_values"].to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            for score in probs:
                gated = False
                for idx, label in enumerate(label_columns):
                    if score[idx] >= thresholds[label]:
                        gated = True
                        break
                flags.append(gated)

    flags_arr = np.array(flags)
    retake_rate = float(flags_arr.mean()) if flags_arr.size else 0.0
    print(f"Images evaluated: {flags_arr.size}")
    print(f"Retake rate: {retake_rate:.3f}")
    print(f"Pass count: {(flags_arr == 0).sum()} | Fail count: {(flags_arr == 1).sum()}")


if __name__ == "__main__":
    main()
