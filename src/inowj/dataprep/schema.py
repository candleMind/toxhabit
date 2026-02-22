"""CSV schema utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

REQUIRED_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "sentence_id",
    "token",
    "is_subword",
    "start",
    "end",
)


@dataclass(frozen=True)
class SplitPaths:
    """Paths to trigger/argument token-level CSVs."""

    trigger_csv: str
    arg_csv: str


def validate_columns(columns: Sequence[str], required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Raise ValueError if required columns are missing."""
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
