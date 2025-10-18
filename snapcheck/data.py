from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from .augmentations import QualityAugmentor
from .paths import QUALITY_DATA_DIR
from .skin_tone import compute_ita, ita_to_fitzpatrick, ita_to_monk
from .utils import configure_logging, save_jsonl, set_seed

LOGGER = configure_logging("snapcheck.data")


@dataclass
class ManifestRecord:
    image_path: str
    source_image: str
    source_dataset: str
    diagnosis: str
    split: str
    capture_channel: str
    ita_score: Optional[float]
    fitzpatrick_type: Optional[int]
    monk_skin_tone: Optional[int]
    quality_labels: Dict[str, int]
    augmentation_metadata: Dict[str, float]

    def to_row(self) -> Dict[str, any]:
        base = {
            "image_path": self.image_path,
            "source_image": self.source_image,
            "source_dataset": self.source_dataset,
            "diagnosis": self.diagnosis,
            "split": self.split,
            "capture_channel": self.capture_channel,
            "ita_score": self.ita_score,
            "fitzpatrick_type": self.fitzpatrick_type,
            "monk_skin_tone": self.monk_skin_tone,
            "skin_tone_bin": self.fitzpatrick_type,
        }
        base.update(self.quality_labels)
        for key, value in self.augmentation_metadata.items():
            base[f"meta_{key}"] = value
        return base
def _assign_capture_channel(source: str) -> str:
    if source.lower() in {"ham10000", "derm7pt"}:
        return "clinic"
    return "patient_generated"


DEFAULT_LABELS = {
    "blur": 0,
    "motion_blur": 0,
    "low_brightness": 0,
    "high_brightness": 0,
    "low_contrast": 0,
    "high_contrast": 0,
    "noise": 0,
    "shadow": 0,
    "obstruction": 0,
    "framing": 0,
    "low_resolution": 0,
    "overall_fail": 0,
}


def _lookup_diagnosis(df: pd.DataFrame, source_cfg: Dict[str, str], image_path: Path) -> str:
    diag_key = source_cfg.get("diagnosis_key", "dx")
    lookup_candidates: List[str] = []
    explicit_keys = source_cfg.get("lookup_keys")
    if isinstance(explicit_keys, list):
        lookup_candidates.extend(explicit_keys)
    for key_name in ("lesion_id_key", "image_id_key"):
        candidate = source_cfg.get(key_name)
        if candidate:
            lookup_candidates.append(candidate)
    lookup_candidates.extend(["image_id", "image", "filename", "case_id", "case_num", "lesion_id"])

    stem = image_path.stem
    for key in lookup_candidates:
        if not key or key not in df.columns:
            continue
        if key == "case_num":
            digits = "".join(ch for ch in stem if ch.isdigit())
            if not digits:
                continue
            match_value = str(int(digits))
        else:
            match_value = stem
        matches = df[df[key].astype(str) == match_value]
        if not matches.empty and diag_key in matches.columns:
            return str(matches.iloc[0][diag_key])
    if diag_key in df.columns and not df.empty:
        return str(df.iloc[0][diag_key])
    return "unknown"


def build_quality_dataset(config: Dict[str, any]) -> Dict[str, Path]:
    set_seed(config.get("seed", 1337))
    quality_dir = QUALITY_DATA_DIR
    quality_dir.mkdir(parents=True, exist_ok=True)
    images_dir = quality_dir / "images"
    images_dir.mkdir(exist_ok=True)

    sources = config.get("sources", [])
    base_images_per_source = config.get("base_images_per_source", 500)

    all_records: List[ManifestRecord] = []

    augmentor = QualityAugmentor(config=config, seed=config.get("seed", 1337))

    for source in sources:
        dataset_name = source.get("name", "unknown")
        metadata_path = Path(source.get("metadata", ""))
        image_pattern = source.get("images_glob")
        if not image_pattern:
            LOGGER.warning("No image glob specified for %s", dataset_name)
            continue
        matched_paths = list(Path().glob(image_pattern))
        if not matched_paths:
            LOGGER.warning("No images found for pattern %s", image_pattern)
            continue
        matched_paths.sort()
        rng = random.Random(config.get("seed", 1337) + hash(dataset_name))
        rng.shuffle(matched_paths)
        sampled_paths = matched_paths[:base_images_per_source]

        if metadata_path.exists():
            df_meta = pd.read_csv(metadata_path)
        else:
            LOGGER.warning("Metadata missing for %s", dataset_name)
            df_meta = pd.DataFrame()

        LOGGER.info("Processing %d images from %s", len(sampled_paths), dataset_name)

        for path in sampled_paths:
            try:
                image = Image.open(path).convert("RGB")
            except Exception as exc:
                LOGGER.warning("Skipping %s due to error: %s", path, exc)
                continue

            ita_result = compute_ita(image, strategy=config.get("skin_patch_strategy", "patches"))
            ita_score = ita_result.median
            fitzpatrick_type = ita_to_fitzpatrick(ita_score) if ita_score == ita_score else -1
            monk_tone = ita_to_monk(ita_score) if ita_score == ita_score else -1
            capture_channel = _assign_capture_channel(dataset_name)
            diagnosis = _lookup_diagnosis(df_meta, source, path) if not df_meta.empty else "unknown"

            pass_filename = f"{dataset_name}_{path.stem}_pass.jpg"
            pass_path = images_dir / pass_filename
            image.save(pass_path)
            pass_record = ManifestRecord(
                image_path=str(pass_path.relative_to(QUALITY_DATA_DIR)),
                source_image=str(path),
                source_dataset=dataset_name,
                diagnosis=diagnosis,
                split="unassigned",
                capture_channel=capture_channel,
                ita_score=float(ita_score) if ita_score == ita_score else None,
                fitzpatrick_type=int(fitzpatrick_type) if fitzpatrick_type >= 0 else None,
                monk_skin_tone=int(monk_tone) if monk_tone >= 0 else None,
                quality_labels=DEFAULT_LABELS.copy(),
                augmentation_metadata={},
            )
            all_records.append(pass_record)

            aug_result = augmentor.apply(image)
            fail_filename = f"{dataset_name}_{path.stem}_fail.jpg"
            fail_path = images_dir / fail_filename
            aug_result.image.save(fail_path)
            fail_record = ManifestRecord(
                image_path=str(fail_path.relative_to(QUALITY_DATA_DIR)),
                source_image=str(path),
                source_dataset=dataset_name,
                diagnosis=diagnosis,
                split="unassigned",
                capture_channel=capture_channel,
                ita_score=float(ita_score) if ita_score == ita_score else None,
                fitzpatrick_type=int(fitzpatrick_type) if fitzpatrick_type >= 0 else None,
                monk_skin_tone=int(monk_tone) if monk_tone >= 0 else None,
                quality_labels=aug_result.labels,
                augmentation_metadata=aug_result.metadata,
            )
            all_records.append(fail_record)

    if not all_records:
        raise RuntimeError("No records created. Verify dataset configuration.")

    manifest_df = pd.DataFrame([rec.to_row() for rec in all_records])

    split_ratios = config.get("train_val_test_split", [0.7, 0.15, 0.15])
    train_prop, val_prop, test_prop = split_ratios
    train_df, temp_df = train_test_split(
        manifest_df,
        test_size=1 - train_prop,
        random_state=config.get("seed", 1337),
        stratify=manifest_df["overall_fail"],
    )
    relative_test = test_prop / (val_prop + test_prop)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=config.get("seed", 1337),
        stratify=temp_df["overall_fail"],
    )

    train_df.loc[:, "split"] = "train"
    val_df.loc[:, "split"] = "val"
    test_df.loc[:, "split"] = "test"

    manifest_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    output_paths = {
        "full": QUALITY_DATA_DIR / "manifest.csv",
        "train": QUALITY_DATA_DIR / "train_manifest.csv",
        "val": QUALITY_DATA_DIR / "val_manifest.csv",
        "test": QUALITY_DATA_DIR / "test_manifest.csv",
    }
    manifest_df.to_csv(output_paths["full"], index=False)
    train_df.to_csv(output_paths["train"], index=False)
    val_df.to_csv(output_paths["val"], index=False)
    test_df.to_csv(output_paths["test"], index=False)

    save_jsonl((rec.to_row() for rec in all_records), QUALITY_DATA_DIR / "manifest.jsonl")

    LOGGER.info(
        "Dataset saved to %s (%d records)",
        QUALITY_DATA_DIR,
        len(manifest_df),
    )

    return output_paths


__all__ = ["build_quality_dataset", "ManifestRecord"]
