from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch


def configure_logging(name: str = "snapcheck", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Timer:
    name: str

    def __post_init__(self) -> None:
        self.start = datetime.now()

    def stop(self) -> float:
        return (datetime.now() - self.start).total_seconds()


def save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def chunked(it: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(it), chunk_size):
        yield it[i : i + chunk_size]


__all__ = ["configure_logging", "set_seed", "Timer", "save_jsonl", "chunked"]
