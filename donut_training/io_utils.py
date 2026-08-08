"""Small, deterministic data-file utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .jsonl import load_jsonl

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write UTF-8 JSONL, creating only the required parent directories."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_jsonl(
    source: str | Path,
    output_dir: str | Path,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, Path]:
    """Shuffle and split JSONL records reproducibly into train/val/test files."""

    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios must be non-negative and sum to 1")
    rows = load_jsonl(source)
    random.Random(seed).shuffle(rows)
    train_end = int(len(rows) * ratios[0])
    validation_end = train_end + int(len(rows) * ratios[1])
    parts = {
        "train": rows[:train_end],
        "validation": rows[train_end:validation_end],
        "test": rows[validation_end:],
    }
    root = Path(output_dir).expanduser().resolve()
    paths = {name: root / f"{name}.jsonl" for name in parts}
    for name, subset in parts.items():
        write_jsonl(subset, paths[name])
    return paths


def analyze_text_lengths(
    jsonl_path: str | Path,
    tokenizer: "PreTrainedTokenizerBase",
    text_field: str,
    max_tokens: int,
) -> dict[str, float | int]:
    """Summarize target token lengths and coverage at a configured limit."""

    lengths = [
        len(tokenizer.encode(str(row.get(text_field, "")), add_special_tokens=False))
        for row in load_jsonl(jsonl_path)
    ]
    if not lengths:
        return {"count": 0, "mean": 0.0, "min": 0, "max": 0, "coverage": 0.0}
    return {
        "count": len(lengths),
        "mean": sum(lengths) / len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "coverage": sum(length <= max_tokens for length in lengths) / len(lengths),
    }
