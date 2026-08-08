#!/usr/bin/env python3
"""Run Donut inference for one or more image files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from donut_training.config import load_config
from donut_training.image_processing import process_images
from donut_training.inference import generate_text, select_device
from donut_training.workflow import build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True, nargs="+")
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not config.model.checkpoint:
        raise ValueError("predict.py requires model.checkpoint in the config")
    runtime = build_runtime(config)
    device = select_device(args.device)
    runtime.model.to(device)
    tensors: list[torch.Tensor] = []
    paths = [Path(item).expanduser().resolve() for item in args.image]
    for path in paths:
        with Image.open(path) as image:
            tensor = process_images(
                image.convert("RGB"),
                target_size=(config.data.image_height, config.data.image_width),
                grayscale=config.data.grayscale,
            )
            assert isinstance(tensor, torch.Tensor)
            tensors.append(tensor)
    predictions, _ = generate_text(
        runtime.model,
        runtime.tokenizer,
        torch.stack(tensors),
        device=device,
        prompt=args.prompt,
        strip_after=config.model.prompt_close,
    )
    for path, prediction in zip(paths, predictions):
        print(json.dumps({"image": str(path), "text": prediction}, ensure_ascii=False))


if __name__ == "__main__":
    main()
