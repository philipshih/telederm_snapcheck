from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .paths import CONFIG_DIR


@dataclass
class Config:
    data: Dict[str, Any]
    path: Path

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, item: str) -> Any:
        return self.data[item]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data=data, path=path)


__all__ = ["Config", "load_config"]
