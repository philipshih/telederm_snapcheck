"""Download public dermatology datasets via HuggingFace Hub."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from snapcheck.paths import RAW_DATA_DIR

DATASETS = {
    "ham10000": "xhlulu/ham10000-segmentation",
    "derm7pt": "midudev/derm7pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download dermatology datasets")
    parser.add_argument("--output", type=str, default=str(RAW_DATA_DIR))
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        repo = DATASETS.get(name)
        if not repo:
            print(f"Unknown dataset {name}; skipping")
            continue
        target_dir = output_root / name
        print(f"Downloading {name} -> {target_dir}")
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=target_dir, local_dir_use_symlinks=False)
    print("Done.")


if __name__ == "__main__":
    main()
