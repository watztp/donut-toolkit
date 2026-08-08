"""Dependency-free JSONL reading shared by data preparation and training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and report the exact line for malformed records."""

    source = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {source}:{line_number}") from error
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object at {source}:{line_number}")
            rows.append(row)
    return rows
