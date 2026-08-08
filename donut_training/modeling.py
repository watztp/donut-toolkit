"""Construction, tokenizer validation, and vocabulary-safe Donut loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    GenerationConfig as TransformersGenerationConfig,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    VisionEncoderDecoderModel,
)

from .config import GenerationConfig, ModelConfig


@dataclass(frozen=True)
class VocabularyAdaptationReport:
    """Audit record describing whether and how decoder vocabulary was changed."""

    source_tokenizer: str
    target_tokenizer: str
    custom_tokenizer_configured: bool
    custom_tokenizer_used: bool
    source_vocab_size: int
    target_vocab_size: int
    mapping_identical: bool
    adapted: bool
    embedding_resized: bool
    embedding_reordered: bool
    matched_by_exact_token: int
    matched_by_subtokens: int
    newly_initialized: int
    coverage: float
    embeddings_tied: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_tokenizer(
    path: str,
    max_length: int,
    *,
    add_missing_special_tokens: bool = True,
) -> PreTrainedTokenizerBase:
    """Load any compatible HF tokenizer and validate required special tokens."""

    tokenizer_kwargs = {
        "clean_up_tokenization_spaces": False,
        "model_max_length": max_length,
        "padding_side": "right",
        "truncation_side": "right",
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, **tokenizer_kwargs)
    except (OSError, ValueError) as auto_error:
        # A tokenizer trained locally may contain only tokenizer.json and no
        # tokenizer_config.json/model config for AutoTokenizer class inference.
        try:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(path, **tokenizer_kwargs)
        except (OSError, ValueError):
            raise auto_error

    # Bare tokenizer.json files often contain the token strings but not their
    # role metadata. Assign existing IDs before considering vocabulary growth.
    existing_vocab = tokenizer.get_vocab()
    conventional_tokens = {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    }
    for attribute, token in conventional_tokens.items():
        if getattr(tokenizer, attribute) is None and token in existing_vocab:
            setattr(tokenizer, attribute, token)
    missing: dict[str, str] = {}
    if tokenizer.pad_token is None:
        missing["pad_token"] = "<pad>"
    if tokenizer.unk_token is None:
        missing["unk_token"] = "<unk>"
    if tokenizer.bos_token is None:
        missing["bos_token"] = "<s>"
    if tokenizer.eos_token is None:
        missing["eos_token"] = "</s>"
    if missing and not add_missing_special_tokens:
        raise ValueError(f"Source tokenizer {path!r} is missing required tokens: {sorted(missing)}")
    if missing:
        tokenizer.add_special_tokens(missing)
    return tokenizer


def _tokenizer_mapping(tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    mapping = {str(token): int(token_id) for token, token_id in tokenizer.get_vocab().items()}
    if mapping and (min(mapping.values()) < 0 or max(mapping.values()) >= len(tokenizer)):
        raise ValueError("Tokenizer contains token IDs outside its declared vocabulary size")
    return mapping


def _token_sources(
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
    strategy: Literal["exact", "exact_or_subtokens"],
    source_rows: int,
) -> tuple[dict[int, list[int]], int, int]:
    """Map each reusable target row to one or more source embedding rows."""

    source_vocab = _tokenizer_mapping(source_tokenizer)
    target_vocab = _tokenizer_mapping(target_tokenizer)
    mapping: dict[int, list[int]] = {}
    exact_matches = 0
    subtoken_matches = 0
    target_special_tokens = set(target_tokenizer.all_special_tokens)

    for token, target_id in target_vocab.items():
        source_id = source_vocab.get(token)
        if source_id is not None and source_id < source_rows:
            mapping[target_id] = [source_id]
            exact_matches += 1
            continue
        if strategy == "exact" or token in target_special_tokens:
            continue

        # Convert tokenizer-internal pieces (e.g. SentencePiece's ▁) back to
        # surface text before asking the source tokenizer to segment them.
        surface = target_tokenizer.convert_tokens_to_string([token])
        if not surface:
            continue
        source_ids = source_tokenizer.encode(surface, add_special_tokens=False)
        if not source_ids or any(token_id < 0 or token_id >= source_rows for token_id in source_ids):
            continue
        if source_tokenizer.unk_token_id is not None and source_tokenizer.unk_token_id in source_ids:
            continue
        mapping[target_id] = [int(token_id) for token_id in source_ids]
        subtoken_matches += 1
    return mapping, exact_matches, subtoken_matches


def _new_embedding_like(
    old_embedding: nn.Embedding,
    size: int,
    padding_idx: int | None,
    init_std: float,
) -> nn.Embedding:
    embedding = nn.Embedding(
        size,
        old_embedding.embedding_dim,
        padding_idx=padding_idx,
        device=old_embedding.weight.device,
        dtype=old_embedding.weight.dtype,
    )
    nn.init.normal_(embedding.weight, mean=0.0, std=init_std)
    if padding_idx is not None:
        with torch.no_grad():
            embedding.weight[padding_idx].zero_()
    return embedding


def _adapt_decoder_vocabulary(
    model: VisionEncoderDecoderModel,
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
    strategy: Literal["exact", "exact_or_subtokens"],
) -> tuple[int, int, int, bool]:
    """Replace only decoder embeddings/output projection, preserving decoder blocks."""

    decoder = model.decoder
    old_input = decoder.get_input_embeddings()
    old_output = decoder.get_output_embeddings()
    if not isinstance(old_input, nn.Embedding):
        raise TypeError("Decoder input embeddings must be torch.nn.Embedding")
    if old_output is None or not hasattr(old_output, "weight"):
        raise TypeError("Decoder must expose an output projection with weights")

    source_rows, hidden_size = old_input.weight.shape
    if old_output.weight.shape[0] != source_rows:
        raise ValueError(
            "Decoder input embedding and output projection have different vocabulary sizes"
        )
    if len(source_tokenizer) != source_rows:
        raise ValueError(
            "Source tokenizer/model mismatch: "
            f"tokenizer has {len(source_tokenizer)} tokens but decoder has {source_rows} rows"
        )
    target_size = len(target_tokenizer)
    sources, exact_matches, subtoken_matches = _token_sources(
        source_tokenizer,
        target_tokenizer,
        strategy,
        source_rows,
    )
    init_std = float(getattr(decoder.config, "init_std", 0.02))
    new_input = _new_embedding_like(
        old_input,
        target_size,
        target_tokenizer.pad_token_id,
        init_std,
    )
    new_output = nn.Linear(
        hidden_size,
        target_size,
        bias=False,
        device=old_output.weight.device,
        dtype=old_output.weight.dtype,
    )
    nn.init.normal_(new_output.weight, mean=0.0, std=init_std)

    with torch.no_grad():
        for target_id, source_ids in sources.items():
            source_index = torch.tensor(source_ids, device=old_input.weight.device)
            new_input.weight[target_id].copy_(old_input.weight.index_select(0, source_index).mean(0))
            output_index = source_index.to(old_output.weight.device)
            new_output.weight[target_id].copy_(
                old_output.weight.index_select(0, output_index).mean(0)
            )

    was_tied = old_input.weight.data_ptr() == old_output.weight.data_ptr() or bool(
        getattr(decoder.config, "tie_word_embeddings", False)
    )
    decoder.set_input_embeddings(new_input)
    decoder.set_output_embeddings(new_output)

    # Some seq2seq heads keep a separate vocabulary bias; preserve mapped rows.
    if hasattr(decoder, "final_logits_bias"):
        old_bias = decoder.final_logits_bias.detach()
        new_bias = old_bias.new_zeros((1, target_size))
        with torch.no_grad():
            for target_id, source_ids in sources.items():
                valid_ids = [index for index in source_ids if index < old_bias.shape[-1]]
                if valid_ids:
                    new_bias[0, target_id] = old_bias[0, valid_ids].mean()
        decoder.register_buffer("final_logits_bias", new_bias)

    decoder.config.vocab_size = target_size
    if hasattr(decoder, "model"):
        decoder.model.config.vocab_size = target_size
    model.config.decoder.vocab_size = target_size
    if was_tied:
        decoder.config.tie_word_embeddings = True
        decoder.tie_weights()

    current_input = decoder.get_input_embeddings().weight
    current_output = decoder.get_output_embeddings().weight
    tied_after = current_input.data_ptr() == current_output.data_ptr()
    initialized = target_size - exact_matches - subtoken_matches
    return exact_matches, subtoken_matches, initialized, tied_after


def _configure_model(
    model: VisionEncoderDecoderModel,
    tokenizer: PreTrainedTokenizerBase,
    image_size: tuple[int, int],
) -> VisionEncoderDecoderModel:
    """Synchronize IDs and image shape without implicitly resizing vocabulary."""

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = (
        tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    )
    model.decoder.config.pad_token_id = tokenizer.pad_token_id
    model.decoder.config.eos_token_id = tokenizer.eos_token_id
    model.decoder.config.bos_token_id = tokenizer.bos_token_id
    model.config.use_cache = False
    model.decoder.config.use_cache = False
    model.encoder.config.image_size = list(image_size)
    return model


def configure_generation(
    model: VisionEncoderDecoderModel,
    tokenizer: PreTrainedTokenizerBase,
    config: GenerationConfig,
) -> VisionEncoderDecoderModel:
    """Replace inherited generation defaults with project-owned settings.

    Donut checkpoints can carry the legacy ``max_length=20`` default. The
    project controls generated length with ``max_new_tokens`` instead, so a
    fresh config intentionally leaves ``max_length`` unset. This config is
    attached before training, which makes Trainer persist it in every saved
    checkpoint as well as the final model.
    """

    model.generation_config = TransformersGenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        num_beams=config.num_beams,
        num_beam_groups=config.num_beam_groups,
        early_stopping=config.early_stopping,
        repetition_penalty=config.repetition_penalty,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=config.encoder_no_repeat_ngram_size,
        length_penalty=config.length_penalty,
        diversity_penalty=config.diversity_penalty,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        typical_p=config.typical_p,
    )
    return model


def build_model_and_tokenizer(
    config: ModelConfig,
    max_length: int,
    image_size: tuple[int, int],
) -> tuple[VisionEncoderDecoderModel, PreTrainedTokenizerBase, VocabularyAdaptationReport]:
    """Build runtime objects and adapt vocabulary only under explicit policy."""

    model_source = config.checkpoint or config.pretrained_model_name_or_path
    source_path = config.source_tokenizer_path or model_source
    try:
        source_tokenizer = load_tokenizer(
            source_path,
            max_length,
            add_missing_special_tokens=False,
        )
    except (OSError, ValueError) as error:
        if config.checkpoint and not config.source_tokenizer_path:
            raise ValueError(
                "Checkpoint does not contain a loadable tokenizer. Set "
                "model.source_tokenizer_path to the exact tokenizer originally used "
                "for this checkpoint."
            ) from error
        raise
    custom_configured = config.tokenizer_path is not None
    custom_used = custom_configured and config.adapt_decoder_vocabulary
    target_path = config.tokenizer_path if custom_used else source_path
    target_tokenizer = (
        load_tokenizer(target_path, max_length)
        if custom_used
        else source_tokenizer
    )

    # Both remote Donut pretraining and local fine-tuned checkpoints are loaded
    # as complete encoder-decoder models; no manual encoder/decoder assembly.
    model = VisionEncoderDecoderModel.from_pretrained(model_source)

    source_vocab = _tokenizer_mapping(source_tokenizer)
    target_vocab = _tokenizer_mapping(target_tokenizer)
    mapping_identical = source_vocab == target_vocab
    source_rows = model.decoder.get_input_embeddings().num_embeddings
    output_rows = model.decoder.get_output_embeddings().weight.shape[0]
    model_size_matches = source_rows == len(target_tokenizer) == output_rows
    needs_adaptation = not mapping_identical or not model_size_matches

    if needs_adaptation and not config.adapt_decoder_vocabulary:
        raise ValueError(
            "Tokenizer mapping does not match decoder vocabulary. Set "
            "model.adapt_decoder_vocabulary=true to adapt it explicitly."
        )
    if (
        needs_adaptation
        and config.checkpoint
        and not config.allow_checkpoint_vocabulary_adaptation
    ):
        raise ValueError(
            "Custom tokenizer differs from checkpoint tokenizer. Set "
            "model.allow_checkpoint_vocabulary_adaptation=true only for an intentional "
            "checkpoint vocabulary migration."
        )

    if needs_adaptation:
        exact, subtokens, initialized, tied = _adapt_decoder_vocabulary(
            model,
            source_tokenizer,
            target_tokenizer,
            config.vocabulary_init_strategy,
        )
    else:
        exact, subtokens, initialized = len(target_tokenizer), 0, 0
        input_weight = model.decoder.get_input_embeddings().weight
        output_weight = model.decoder.get_output_embeddings().weight
        tied = input_weight.data_ptr() == output_weight.data_ptr()

    model = _configure_model(model, target_tokenizer, image_size)
    report = VocabularyAdaptationReport(
        source_tokenizer=source_path,
        target_tokenizer=target_path,
        custom_tokenizer_configured=custom_configured,
        custom_tokenizer_used=custom_used,
        source_vocab_size=len(source_tokenizer),
        target_vocab_size=len(target_tokenizer),
        mapping_identical=mapping_identical,
        adapted=needs_adaptation,
        embedding_resized=needs_adaptation and source_rows != len(target_tokenizer),
        embedding_reordered=needs_adaptation and not mapping_identical,
        matched_by_exact_token=exact,
        matched_by_subtokens=subtokens,
        newly_initialized=initialized,
        coverage=(exact + subtokens) / max(1, len(target_tokenizer)),
        embeddings_tied=tied,
    )
    return model, target_tokenizer, report


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
