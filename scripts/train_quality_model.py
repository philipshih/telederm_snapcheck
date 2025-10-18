"""Train the SnapCheck dermatology image quality model."""
from __future__ import annotations

import argparse
from pathlib import Path
import numbers

from snapcheck.config import load_config
from snapcheck.quality_model import QualityTrainer, TARGET_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SnapCheck DIQA model")
    parser.add_argument("--config", type=str, default="train_diqa.yaml", help="Training config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config).data
    trainer = QualityTrainer(cfg, label_columns=cfg.get("label_columns", TARGET_LABELS))
    metrics = trainer.train()
    print("Final test metrics:")
    for key, value in metrics.items():
        if key in {"logits", "labels", "probabilities"}:
            continue
        if isinstance(value, numbers.Number):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
