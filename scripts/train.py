#!/usr/bin/env python3
"""Train a Donut model from a JSON project configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from donut_training.config import load_config
from donut_training.modeling import count_parameters
from donut_training.training import create_trainer
from donut_training.workflow import build_local_dataset, build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = build_runtime(config)
    train_dataset = build_local_dataset(config, runtime.tokenizer, "train", augment=True)
    validation_dataset = build_local_dataset(config, runtime.tokenizer, "validation")
    total, trainable = count_parameters(runtime.model)
    print(f"Parameters: {total / 1e6:.2f}M total, {trainable / 1e6:.2f}M trainable")
    print(f"Samples: {len(train_dataset)} train, {len(validation_dataset)} validation")
    print("Vocabulary:", json.dumps(runtime.vocabulary_report.to_dict(), ensure_ascii=False))
    run_output_dir = Path(config.train.output_dir).expanduser().resolve()
    run_output_dir.mkdir(parents=True, exist_ok=True)
    with (run_output_dir / "vocabulary_adaptation.json").open("w", encoding="utf-8") as file:
        json.dump(runtime.vocabulary_report.to_dict(), file, ensure_ascii=False, indent=2)
        file.write("\n")

    trainer = create_trainer(
        runtime.model,
        runtime.tokenizer,
        train_dataset,
        validation_dataset,
        config.train,
    )
    trainer.train(resume_from_checkpoint=config.train.resume_from_checkpoint)

    output_dir = run_output_dir / "best"
    trainer.save_model(output_dir)
    runtime.tokenizer.save_pretrained(output_dir)
    with (output_dir / "project_config.json").open("w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, ensure_ascii=False, indent=2)
    with (output_dir / "vocabulary_adaptation.json").open("w", encoding="utf-8") as file:
        json.dump(runtime.vocabulary_report.to_dict(), file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Best model saved to {output_dir}")


if __name__ == "__main__":
    main()
