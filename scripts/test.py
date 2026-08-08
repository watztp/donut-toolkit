#!/usr/bin/env python3
"""Evaluate a trained Donut checkpoint on the configured test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from donut_training.config import load_config
from donut_training.data import VisionSeq2SeqCollator
from donut_training.evaluation_metrics import (
    FieldMetricAccumulator,
    MetricAccumulator,
    normalize_text,
)
from donut_training.inference import generate_text, select_device
from donut_training.workflow import build_local_dataset, build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None, help="For example: cuda, cuda:0, or cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Metric JSON path (default: <train.output_dir>/test_metrics.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not config.model.checkpoint:
        raise ValueError("test.py requires model.checkpoint in the config")
    runtime = build_runtime(config)
    device = select_device(args.device)
    runtime.model.to(device)
    dataset = build_local_dataset(config, runtime.tokenizer, "test", limit=args.limit)
    loader = DataLoader(
        dataset,
        batch_size=config.generation.batch_size,
        shuffle=False,
        collate_fn=VisionSeq2SeqCollator(
            pad_token_id=runtime.tokenizer.pad_token_id,
            decoder_start_token_id=runtime.model.config.decoder_start_token_id,
        ),
        num_workers=config.train.dataloader_workers,
    )
    metrics = MetricAccumulator()
    field_metrics = FieldMetricAccumulator(config.generation.field_fuzzy_threshold)
    tags = (config.model.prompt_open, config.model.prompt_close)

    for batch in tqdm(loader, desc="Testing"):
        predictions, _ = generate_text(
            runtime.model,
            runtime.tokenizer,
            batch["pixel_values"],
            device=device,
        )
        label_rows = batch["labels"].tolist()
        for prediction, label_ids in zip(predictions, label_rows):
            reference_ids = [token for token in label_ids if token != -100]
            reference = runtime.tokenizer.decode(reference_ids, skip_special_tokens=True)
            clean_prediction = normalize_text(prediction, tags)
            clean_reference = normalize_text(reference, tags)
            prediction_tokens = runtime.tokenizer.encode(clean_prediction, add_special_tokens=False)
            reference_tokens = runtime.tokenizer.encode(clean_reference, add_special_tokens=False)
            metrics.update(clean_prediction, clean_reference, prediction_tokens, reference_tokens)
            field_metrics.update(clean_prediction, clean_reference)

    result = metrics.compute()
    character_errors = result.pop("most_common_character_errors")
    result["field_metrics"] = field_metrics.compute()
    # Keep the long diagnostic table last so headline metrics stay readable.
    result["most_common_character_errors"] = character_errors
    print(json.dumps(result, ensure_ascii=False, indent=2))
    destination = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(config.train.output_dir).expanduser().resolve() / "test_metrics.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Metrics saved to {destination}")


if __name__ == "__main__":
    main()
