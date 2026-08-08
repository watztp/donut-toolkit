"""Typed project configuration loaded from JSON files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class ModelConfig:
    """Model and tokenizer sources used for training or evaluation."""

    # Custom tokenizer is used only when this path is set and vocabulary
    # adaptation is explicitly enabled. Otherwise the model's tokenizer wins.
    tokenizer_path: str | None = None
    # Legacy checkpoints may not contain tokenizer files. Point this to the
    # exact tokenizer originally used to train that checkpoint.
    source_tokenizer_path: str | None = None
    checkpoint: str | None = None
    pretrained_model_name_or_path: str = "naver-clova-ix/donut-base-finetuned-cord-v2"
    adapt_decoder_vocabulary: bool = False
    allow_checkpoint_vocabulary_adaptation: bool = False
    vocabulary_init_strategy: Literal["exact", "exact_or_subtokens"] = "exact_or_subtokens"
    prompt_open: str = ""
    prompt_close: str = ""

    def __post_init__(self) -> None:
        if self.source_tokenizer_path and not self.checkpoint:
            raise ValueError("model.source_tokenizer_path is only valid with checkpoint")
        if self.adapt_decoder_vocabulary and not self.tokenizer_path:
            raise ValueError(
                "model.adapt_decoder_vocabulary=true requires model.tokenizer_path"
            )
        if self.allow_checkpoint_vocabulary_adaptation and not self.checkpoint:
            raise ValueError(
                "model.allow_checkpoint_vocabulary_adaptation is only valid with checkpoint"
            )
        if self.allow_checkpoint_vocabulary_adaptation and (
            not self.adapt_decoder_vocabulary or not self.tokenizer_path
        ):
            raise ValueError(
                "checkpoint vocabulary adaptation requires tokenizer_path and "
                "adapt_decoder_vocabulary=true"
            )


@dataclass(frozen=True)
class DataConfig:
    """Local JSONL dataset paths and image preprocessing settings."""

    train_jsonl: str
    validation_jsonl: str
    test_jsonl: str
    image_dir: str
    image_height: int = 1280
    image_width: int = 736
    max_target_tokens: int = 768
    grayscale: bool = True
    augmentation: Literal["light", "mid", "hard"] = "light"
    keep_clean_probability: float = 0.4
    image_field: str = "img_path"
    text_field: str = "gt"


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters passed to Hugging Face Trainer."""

    output_dir: str = "outputs/donut"
    epochs: float = 20.0
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    label_smoothing: float = 0.01
    freeze_encoder_epochs: int = 1
    early_stopping_patience: int = 5
    dataloader_workers: int = 4
    precision: str = "fp16"
    seed: int = 42
    resume_from_checkpoint: str | bool | None = None


@dataclass(frozen=True)
class GenerationConfig:
    """Generation parameters shared by prediction and test scripts."""

    batch_size: int = 1
    max_new_tokens: int = 768
    do_sample: bool = False
    num_beams: int = 1
    num_beam_groups: int = 1
    early_stopping: bool = False
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    encoder_no_repeat_ngram_size: int = 0
    length_penalty: float = 1.0
    diversity_penalty: float = 0.0
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    typical_p: float = 1.0
    field_fuzzy_threshold: float = 0.9

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("generation.batch_size must be at least 1")
        if self.max_new_tokens < 1:
            raise ValueError("generation.max_new_tokens must be at least 1")
        if self.num_beams < 1:
            raise ValueError("generation.num_beams must be at least 1")
        if self.num_beam_groups < 1:
            raise ValueError("generation.num_beam_groups must be at least 1")
        if self.num_beam_groups > self.num_beams:
            raise ValueError("generation.num_beam_groups cannot exceed num_beams")
        if self.num_beams % self.num_beam_groups != 0:
            raise ValueError("generation.num_beams must be divisible by num_beam_groups")
        if self.repetition_penalty <= 0:
            raise ValueError("generation.repetition_penalty must be greater than 0")
        if self.no_repeat_ngram_size < 0:
            raise ValueError("generation.no_repeat_ngram_size cannot be negative")
        if self.encoder_no_repeat_ngram_size < 0:
            raise ValueError("generation.encoder_no_repeat_ngram_size cannot be negative")
        if self.diversity_penalty < 0:
            raise ValueError("generation.diversity_penalty cannot be negative")
        if self.temperature <= 0:
            raise ValueError("generation.temperature must be greater than 0")
        if self.top_k < 0:
            raise ValueError("generation.top_k cannot be negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("generation.top_p must be greater than 0 and at most 1")
        if not 0.0 < self.typical_p <= 1.0:
            raise ValueError("generation.typical_p must be greater than 0 and at most 1")
        if not 0.0 <= self.field_fuzzy_threshold <= 1.0:
            raise ValueError("generation.field_fuzzy_threshold must be between 0 and 1")


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object for the whole project."""

    model: ModelConfig
    data: DataConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Config section '{name}' must be a JSON object")
    return value


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a JSON config into typed dataclasses."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, Mapping):
        raise TypeError("The config root must be a JSON object")

    return AppConfig(
        model=ModelConfig(**_section(raw, "model")),
        data=DataConfig(**_section(raw, "data")),
        train=TrainConfig(**_section(raw, "train")),
        generation=GenerationConfig(**_section(raw, "generation")),
    )
