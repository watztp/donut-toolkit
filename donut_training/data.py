"""Dataset and collation code for local receipt JSONL files."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypedDict

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .image_processing import ImageInput
from .jsonl import load_jsonl


class ImageProcessor(Protocol):
    def __call__(self, image: ImageInput) -> torch.Tensor: ...


class DonutSample(TypedDict):
    pixel_values: torch.Tensor
    labels: torch.Tensor


def extract_synthdog_text(ground_truth: object) -> str:
    """Extract ``text_sequence`` from the common SynthDog ground-truth shape."""

    value = ground_truth
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value.strip()
    if isinstance(value, Mapping):
        parsed = value.get("gt_parse")
        if isinstance(parsed, Mapping) and isinstance(parsed.get("text_sequence"), str):
            return str(parsed["text_sequence"]).strip()
        for key in ("text", "label", "target", "sequence"):
            if isinstance(value.get(key), str):
                return str(value[key]).strip()
    return ""


class ReceiptJsonlDataset(Dataset[DonutSample]):
    """Map local receipt images and target text to Donut training samples.

    Labels do not contain BOS. The collator shifts padded labels and inserts
    ``decoder_start_token_id`` so training also works with label smoothing.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        image_dir: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        image_processor: ImageProcessor,
        max_target_tokens: int,
        *,
        image_field: str = "img_path",
        text_field: str = "gt",
        prompt_open: str = "",
        prompt_close: str = "",
        limit: int | None = None,
    ) -> None:
        self.rows = load_jsonl(jsonl_path)
        if limit is not None:
            self.rows = self.rows[:limit]
        self.image_dir = Path(image_dir).expanduser().resolve()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_target_tokens = max_target_tokens
        self.image_field = image_field
        self.text_field = text_field
        self.prompt_open = prompt_open
        self.prompt_close = prompt_close

        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        if max_target_tokens < 2:
            raise ValueError("max_target_tokens must be at least 2")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> DonutSample:
        row = self.rows[index]
        relative_path = row.get(self.image_field)
        if not isinstance(relative_path, str):
            raise KeyError(f"Row {index} has no string field '{self.image_field}'")
        image_path = (self.image_dir / relative_path).resolve()
        # Prevent a JSONL entry from escaping the configured image root.
        if not image_path.is_relative_to(self.image_dir):
            raise ValueError(f"Image path escapes image_dir: {relative_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        raw_text = row.get(self.text_field, "")
        text = raw_text.strip() if isinstance(raw_text, str) else str(raw_text)
        target = f"{self.prompt_open}{text}{self.prompt_close}"
        # Reserve one position for EOS; BOS is injected by the collator.
        token_ids = self.tokenizer.encode(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_tokens - 1,
        )
        token_ids.append(self.tokenizer.eos_token_id)

        with Image.open(image_path) as image:
            pixel_values = self.image_processor(image.convert("RGB"))
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(token_ids, dtype=torch.long),
        }


class VisionSeq2SeqCollator:
    """Pad labels and build shifted decoder inputs for label smoothing.

    ``Trainer`` removes labels before the model forward pass when label
    smoothing is enabled. Supplying decoder inputs explicitly keeps the
    VisionEncoderDecoder model runnable in both smoothed and regular training.
    """

    def __init__(
        self,
        pad_token_id: int,
        decoder_start_token_id: int,
        label_pad_id: int = -100,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        if pad_token_id < 0 or decoder_start_token_id < 0:
            raise ValueError("pad_token_id and decoder_start_token_id must be non-negative")
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.label_pad_id = label_pad_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[DonutSample]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        pixel_values = torch.stack([feature["pixel_values"] for feature in features])
        labels = pad_sequence(
            [feature["labels"] for feature in features],
            batch_first=True,
            padding_value=self.label_pad_id,
        )
        if self.pad_to_multiple_of:
            remainder = labels.shape[1] % self.pad_to_multiple_of
            if remainder:
                extra = self.pad_to_multiple_of - remainder
                labels = torch.nn.functional.pad(labels, (0, extra), value=self.label_pad_id)

        # Equivalent to VisionEncoderDecoderModel.shift_tokens_right:
        # decoder inputs are [decoder_start] + labels[:-1], while -100 is a
        # loss-only sentinel and must never reach the embedding lookup.
        decoder_input_ids = torch.full_like(labels, self.pad_token_id)
        decoder_input_ids[:, 0] = self.decoder_start_token_id
        if labels.shape[1] > 1:
            decoder_input_ids[:, 1:] = labels[:, :-1]
        decoder_input_ids.masked_fill_(
            decoder_input_ids == self.label_pad_id,
            self.pad_token_id,
        )
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id)

        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }


def make_image_processor(
    process_fn: Callable[..., torch.Tensor],
    **kwargs: object,
) -> ImageProcessor:
    """Bind preprocessing options into a worker-safe top-level callable."""

    def processor(image: ImageInput) -> torch.Tensor:
        return process_fn(image, **kwargs)

    return processor
