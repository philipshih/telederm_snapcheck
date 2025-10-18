from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import timm
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .fairness import compute_group_metrics
from .paths import MODEL_DIR, QUALITY_DATA_DIR, REPORT_DIR
from .utils import configure_logging, set_seed

LOGGER = configure_logging("snapcheck.quality")


TARGET_LABELS = [
    "blur",
    "motion_blur",
    "low_brightness",
    "high_brightness",
    "low_contrast",
    "high_contrast",
    "noise",
    "shadow",
    "obstruction",
    "framing",
    "low_resolution",
    "overall_fail",
]


class QualityDataset(Dataset):
    def __init__(self, manifest_path: Path, image_root: Path, label_columns: List[str], train: bool = False) -> None:
        self.manifest = pd.read_csv(manifest_path)
        self.image_root = image_root
        self.label_columns = label_columns
        self.transform = self._build_transform(train)

    def _build_transform(self, train: bool) -> transforms.Compose:
        ops: List[transforms.Compose] = [transforms.Resize((224, 224))]
        if train:
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.5),
                ]
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
            ]
        )
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.manifest.iloc[idx]
        image_path = self.image_root / row["image_path"]
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        labels = torch.tensor(row[self.label_columns].values.astype(np.float32))
        sample = {
            "pixel_values": tensor,
            "labels": labels,
            "skin_tone_bin": torch.tensor(int(row.get("skin_tone_bin", -1)) if not pd.isna(row.get("skin_tone_bin", -1)) else -1, dtype=torch.long),
            "capture_channel": row.get("capture_channel", "unknown"),
        }
        return sample


class QualityModel(nn.Module):
    def __init__(self, architecture: str, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.model = timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _resolve_device(device: str) -> torch.device:
    if device == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class QualityTrainer:
    def __init__(self, config: Dict[str, any], label_columns: Optional[List[str]] = None) -> None:
        set_seed(config.get("seed", 1337))
        self.config = config
        self.label_columns = label_columns or TARGET_LABELS
        self.device = _resolve_device(config.get("device", "cuda_if_available"))
        self.model = QualityModel(
            architecture=config.get("architecture", "vit_small_patch16_224"),
            num_classes=len(self.label_columns),
            pretrained=config.get("pretrained", True),
        ).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 5e-5),
            weight_decay=config.get("weight_decay", 0.02),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("max_epochs", 25),
        )

    def load_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_manifest = Path(self.config.get("train_csv", QUALITY_DATA_DIR / "train_manifest.csv"))
        val_manifest = Path(self.config.get("val_csv", QUALITY_DATA_DIR / "val_manifest.csv"))
        test_manifest = Path(self.config.get("test_csv", QUALITY_DATA_DIR / "test_manifest.csv"))
        image_root = QUALITY_DATA_DIR
        train_ds = QualityDataset(train_manifest, image_root, self.label_columns, train=True)
        val_ds = QualityDataset(val_manifest, image_root, self.label_columns, train=False)
        test_ds = QualityDataset(test_manifest, image_root, self.label_columns, train=False)
        batch_size = self.config.get("batch_size", 32)
        num_workers = self.config.get("num_workers", 4)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dl, val_dl, test_dl

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = batch["pixel_values"].to(self.device)
        labels = batch["labels"].to(self.device)
        logits = self.model(inputs)
        loss = self.criterion(logits, labels)
        return loss, logits.detach().cpu()

    def train(self) -> Dict[str, float]:
        train_dl, val_dl, test_dl = self.load_dataloaders()
        best_metric = -float("inf")
        best_state = None

        for epoch in range(self.config.get("max_epochs", 25)):
            self.model.train()
            for batch in train_dl:
                loss, _ = self._forward(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            self.model.eval()
            val_metrics = self.evaluate(val_dl)
            LOGGER.info("Epoch %d | val_auroc=%.4f", epoch + 1, val_metrics["auroc_macro"])
            if val_metrics["auroc_macro"] > best_metric:
                best_metric = val_metrics["auroc_macro"]
                best_state = self.model.state_dict()

        if best_state is not None:
            self.model.load_state_dict(best_state)
        test_metrics = self.evaluate(test_dl)
        LOGGER.info("Test metrics: %s", {k: v for k, v in test_metrics.items() if k not in {"probabilities", "labels"}})
        self._save_outputs(test_metrics)
        return test_metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, any]:
        self.model.eval()
        logits_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["pixel_values"].to(self.device)
                logits = self.model(inputs).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                logits_list.append(logits)
                labels_list.append(labels)
        logits = np.concatenate(logits_list)
        labels = np.concatenate(labels_list)
        probs = 1 / (1 + np.exp(-logits))
        metrics: Dict[str, float] = {}
        try:
            roc_values = [roc_auc_score(labels[:, idx], probs[:, idx]) for idx in range(probs.shape[1])]
            metrics["auroc_macro"] = float(np.nanmean(roc_values))
        except ValueError:
            metrics["auroc_macro"] = float("nan")
        try:
            ap_values = [average_precision_score(labels[:, idx], probs[:, idx]) for idx in range(probs.shape[1])]
            metrics["ap_macro"] = float(np.nanmean(ap_values))
        except ValueError:
            metrics["ap_macro"] = float("nan")
        metrics["probabilities"] = probs
        metrics["labels"] = labels
        return metrics

    def _save_outputs(self, test_metrics: Dict[str, any]) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = MODEL_DIR / "snapcheck_quality.pt"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config,
                "metrics": {k: v for k, v in test_metrics.items() if k not in {"probabilities", "labels"}},
            },
            checkpoint_path,
        )
        LOGGER.info("Saved checkpoint to %s", checkpoint_path)

        report_dir = Path(self.config.get("report_dir", REPORT_DIR / "diqa"))
        report_dir.mkdir(parents=True, exist_ok=True)
        metrics_table = pd.DataFrame(
            [[k, v] for k, v in test_metrics.items() if k not in {"probabilities", "labels"}],
            columns=["metric", "value"],
        )
        metrics_csv = report_dir / "metrics.csv"
        metrics_table.to_csv(metrics_csv, index=False)

        manifest_path = Path(self.config.get("test_csv", QUALITY_DATA_DIR / "test_manifest.csv"))
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            fairness_list = []
            for item in self.config.get("fairness_metrics", []):
                fm = compute_group_metrics(
                    manifest,
                    test_metrics["probabilities"],
                    test_metrics["labels"],
                    self.label_columns,
                    group_column=item["group"],
                    metric=item["metric"],
                )
                for group_name, value in fm.values.items():
                    fairness_list.append(
                        {
                            "metric": fm.metric,
                            "group": fm.group,
                            "group_value": group_name,
                            "value": value,
                        }
                    )
            if fairness_list:
                fairness_df = pd.DataFrame(fairness_list)
                fairness_df.to_csv(report_dir / "fairness.csv", index=False)



def load_quality_checkpoint(checkpoint: str | Path, device: str | torch.device = "cpu") -> tuple[QualityModel, Dict[str, Any], List[str]]:
    """Load a trained quality model checkpoint for inference/export."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    target_device = torch.device(device) if not isinstance(device, torch.device) else device
    state = torch.load(checkpoint_path, map_location=target_device)
    config: Dict[str, Any] = dict(state.get("config", {}))
    label_columns = config.get("label_columns", TARGET_LABELS)
    architecture = config.get("architecture", "vit_small_patch16_224")
    model = QualityModel(architecture=architecture, num_classes=len(label_columns), pretrained=False)
    model.load_state_dict(state["state_dict"])
    model.to(target_device)
    model.eval()
    return model, config, label_columns

__all__ = ["QualityTrainer", "QualityDataset", "QualityModel", "TARGET_LABELS", "load_quality_checkpoint"]
