"""Evaluate a trained SnapCheck DIQA model on the manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from snapcheck.config import load_config
from snapcheck.quality_model import QualityDataset, QualityModel, TARGET_LABELS
from snapcheck.paths import QUALITY_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DIQA model")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best.pt")
    parser.add_argument("--manifest", type=str, default="data/quality/test_manifest.csv")
    parser.add_argument("--device", type=str, default="cuda_if_available")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    config = state.get("config", {})
    label_columns = config.get("label_columns", TARGET_LABELS)
    model = QualityModel(
        architecture=config.get("architecture", "vit_small_patch16_224"),
        num_classes=len(label_columns),
        pretrained=False,
    )
    model.load_state_dict(state["state_dict"])
    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    dataset = QualityDataset(Path(args.manifest), QUALITY_DATA_DIR, label_columns, train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    probs_list = []
    labels_list = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["pixel_values"].to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            labels_list.append(batch["labels"].numpy())
    probs = np.concatenate(probs_list)
    labels = np.concatenate(labels_list)
    results = pd.DataFrame(probs, columns=[f"prob_{name}" for name in label_columns])
    for idx, name in enumerate(label_columns):
        results[f"label_{name}"] = labels[:, idx]
    output_path = QUALITY_DATA_DIR / "evaluation_probs.csv"
    results.to_csv(output_path, index=False)
    print(f"Saved probabilities to {output_path}")


if __name__ == "__main__":
    main()
