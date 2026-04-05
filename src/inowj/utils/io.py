"""I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Write a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_csv(path: str | Path, keep_default_na: bool = False) -> pd.DataFrame:
    """Read a CSV file with consistent options."""
    return pd.read_csv(path, keep_default_na=keep_default_na)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

