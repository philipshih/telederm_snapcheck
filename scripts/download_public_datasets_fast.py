"""Fast parallel download of dermatology datasets via HuggingFace Hub."""
from __future__ import annotations

import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import snapshot_download

from snapcheck.paths import RAW_DATA_DIR

DATASETS = {
    "ham10000": "xhlulu/ham10000-segmentation",
    "derm7pt": "midudev/derm7pt",
}


def download_dataset(name: str, repo: str, output_root: Path) -> None:
    """Download single dataset with progress."""
    target_dir = output_root / name
    print(f"Downloading {name} -> {target_dir}")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        max_workers=8,  # Parallel file downloads
        resume_download=True,  # Resume if interrupted
    )
    print(f"✓ {name} complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast parallel download")
    parser.add_argument("--output", type=str, default=str(RAW_DATA_DIR))
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    # Download datasets in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for name in args.datasets:
            repo = DATASETS.get(name)
            if not repo:
                print(f"Unknown dataset {name}; skipping")
                continue
            future = executor.submit(download_dataset, name, repo, output_root)
            futures.append(future)

        for future in futures:
            future.result()

    print("✓ All downloads complete")


if __name__ == "__main__":
    main()
