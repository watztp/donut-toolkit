"""Factories that assemble configured project components."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import albumentations as A
from transformers import PreTrainedTokenizerBase, VisionEncoderDecoderModel

from .config import AppConfig
from .data import ReceiptJsonlDataset
from .image_processing import build_receipt_augmentation, process_images
from .modeling import (
    VocabularyAdaptationReport,
    build_model_and_tokenizer,
    configure_generation,
)


@dataclass(frozen=True)
class Runtime:
    """Objects shared by train/test workflows."""

    tokenizer: PreTrainedTokenizerBase
    model: VisionEncoderDecoderModel
    vocabulary_report: VocabularyAdaptationReport


def build_runtime(config: AppConfig) -> Runtime:
    """Create tokenizer and model from one project configuration."""

    model, tokenizer, vocabulary_report = build_model_and_tokenizer(
        config.model,
        config.data.max_target_tokens,
        (config.data.image_height, config.data.image_width),
    )
    model = configure_generation(model, tokenizer, config.generation)
    if vocabulary_report.adapted and config.train.resume_from_checkpoint:
        raise ValueError(
            "Cannot resume optimizer state after decoder vocabulary adaptation. "
            "Set train.resume_from_checkpoint=null and start a new optimizer run."
        )
    return Runtime(
        tokenizer=tokenizer,
        model=model,
        vocabulary_report=vocabulary_report,
    )


def build_local_dataset(
    config: AppConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    *,
    augment: bool = False,
    limit: int | None = None,
) -> ReceiptJsonlDataset:
    """Build a train/validation/test dataset with split-appropriate transforms."""

    split_paths = {
        "train": config.data.train_jsonl,
        "validation": config.data.validation_jsonl,
        "test": config.data.test_jsonl,
    }
    if split not in split_paths:
        raise ValueError(f"Unknown split: {split}")
    augmentation: A.Compose | None = None
    if augment:
        augmentation = build_receipt_augmentation(
            config.data.augmentation,
            config.data.keep_clean_probability,
        )
    processor = partial(
        process_images,
        target_size=(config.data.image_height, config.data.image_width),
        grayscale=config.data.grayscale,
        augmentation=augmentation,
        return_mask=False,
    )
    return ReceiptJsonlDataset(
        split_paths[split],
        config.data.image_dir,
        tokenizer,
        processor,
        config.data.max_target_tokens,
        image_field=config.data.image_field,
        text_field=config.data.text_field,
        prompt_open=config.model.prompt_open,
        prompt_close=config.model.prompt_close,
        limit=limit,
    )
