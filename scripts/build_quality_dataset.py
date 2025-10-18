"""Build the synthetic dermatology image-quality dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from snapcheck.config import load_config
from snapcheck.data import build_quality_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SnapCheck quality dataset")
    parser.add_argument("--config", type=str, default="augmentation.yaml", help="Path to augmentation config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config).data
    paths = build_quality_dataset(config)
    print("Saved manifests:")
    for split, path in paths.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()
