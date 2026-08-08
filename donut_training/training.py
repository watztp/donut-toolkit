"""Trainer construction and encoder freeze scheduling."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .config import TrainConfig
from .data import VisionSeq2SeqCollator


class FreezeEncoderCallback(TrainerCallback):
    """Freeze the vision encoder initially, then unfreeze it once."""

    def __init__(self, freeze_epochs: int) -> None:
        self.freeze_epochs = max(0, freeze_epochs)
        self._unfrozen = self.freeze_epochs == 0

    @staticmethod
    def _set_trainable(model: PreTrainedModel, trainable: bool) -> None:
        encoder = getattr(model, "encoder", None)
        if encoder is not None:
            for parameter in encoder.parameters():
                parameter.requires_grad = trainable

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if not self._unfrozen:
            self._set_trainable(kwargs["model"], False)
            print(f"Encoder frozen for {self.freeze_epochs} epoch(s).")

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        epoch = int(state.epoch or 0)
        if not self._unfrozen and epoch >= self.freeze_epochs:
            self._set_trainable(kwargs["model"], True)
            self._unfrozen = True
            print("Encoder unfrozen.")


def _precision_flags(precision: str) -> tuple[bool, bool]:
    normalized = precision.lower()
    if normalized not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be one of: fp32, fp16, bf16")
    has_cuda = torch.cuda.is_available()
    return normalized == "fp16" and has_cuda, normalized == "bf16" and has_cuda


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset[Any],
    validation_dataset: Dataset[Any],
    config: TrainConfig,
) -> Trainer:
    """Create a configured Hugging Face Trainer without starting training."""

    use_fp16, use_bf16 = _precision_flags(config.precision)
    arguments = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        label_smoothing_factor=config.label_smoothing,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=config.dataloader_workers,
        remove_unused_columns=False,
        seed=config.seed,
        report_to=[],
    )
    callbacks: list[TrainerCallback] = [FreezeEncoderCallback(config.freeze_encoder_epochs)]
    if config.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(config.early_stopping_patience))
    return Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=VisionSeq2SeqCollator(
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=model.config.decoder_start_token_id,
        ),
        callbacks=callbacks,
        processing_class=tokenizer,
    )
