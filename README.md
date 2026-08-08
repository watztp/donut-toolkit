# Donut Receipt Training Toolkit

This repository contains a small, configuration-driven toolkit for fine-tuning,
evaluating, and running inference with
[Donut](https://huggingface.co/docs/transformers/model_doc/donut) receipt models.
It is training and evaluation code only: no trained weights, private datasets, or
benchmark claims are included.

The default model source is
[`naver-clova-ix/donut-base-finetuned-cord-v2`](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2).
The toolkit can use its original tokenizer or explicitly adapt the decoder to a
custom tokenizer.

## What is included

- JSON configuration shared by training, testing, and prediction
- Image resizing, letterboxing, grayscale conversion, and train-only augmentation
- JSONL datasets and a collator compatible with label smoothing
- Full Donut checkpoint loading without manually rebuilding the encoder and decoder
- Explicit, auditable decoder vocabulary adaptation for custom tokenizers
- Encoder freeze/unfreeze scheduling, early stopping, and best-checkpoint loading
- Text, character, token, and structured field-level evaluation metrics
- Reproducible JSONL splitting and target-length analysis

## Project layout

```text
donut_training/
  config.py              Typed project configuration and validation
  data.py                JSONL dataset and seq2seq batch collation
  evaluation_metrics.py  Text, character, token, and field metrics
  image_processing.py    Image transforms and augmentation
  inference.py           Generation and device selection
  io_utils.py            Dataset splitting and length analysis
  jsonl.py                JSONL loading helpers
  modeling.py             Model loading and vocabulary adaptation
  training.py             Hugging Face Trainer construction
  workflow.py             Shared train/test/predict factories
scripts/
  prepare_data.py         Split data or analyze target lengths
  train.py                Fine-tune a model
  test.py                 Evaluate a checkpoint
  predict.py              Run inference on image files
configs/
  train.example.json      Version-controlled configuration template
labels.example.jsonl      Example dataset records
tests/                    Unit tests
```

## Requirements and installation

- Python 3.10 or newer
- PyTorch 2.2 or newer
- A CUDA-capable GPU is recommended for training but is not required for basic checks

Create an isolated environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The pretrained model is downloaded from Hugging Face on first use. For offline
operation, download it in advance and set `model.pretrained_model_name_or_path`
to its local directory.

## Dataset format

The input is JSON Lines: one JSON object per line. By default, each record uses
`img_path` for the image path and `gt` for the target sequence:

```json
{"img_path": "receipt_0001.png", "gt": "<s_receipt><store><store_nm>Example Shop</store_nm></store><summary><total_price>12.50</total_price></summary></s_receipt>"}
```

See [`labels.example.jsonl`](labels.example.jsonl) for complete sample records.
`img_path` must be relative to `data.image_dir`; absolute image paths and paths
that escape the configured image directory are rejected. Field names can be
changed with `data.image_field` and `data.text_field`.

A typical local layout is:

```text
data/
  images/
    receipt_0001.png
    receipt_0002.png
  train.jsonl
  validation.jsonl
  test.jsonl
```

Keep train, validation, and test data separate for real evaluation. Reusing the
training records for validation and testing is useful only as an end-to-end smoke
test and does not measure generalization.

## Configuration

Create a private working config from the tracked template:

```bash
mkdir -p configs_own
cp configs/train.example.json configs_own/train.json
```

Edit the paths and hyperparameters in `configs_own/train.json`. The
`configs_own/` directory is ignored by Git so machine-specific paths and local
experiments are not committed.

### Pretrained model and tokenizer

To fine-tune CORD v2 with its original tokenizer, use:

```json
{
  "checkpoint": null,
  "pretrained_model_name_or_path": "naver-clova-ix/donut-base-finetuned-cord-v2",
  "tokenizer_path": null,
  "adapt_decoder_vocabulary": false
}
```

When `model.checkpoint` is set, the complete encoder-decoder model and its
tokenizer are loaded from that checkpoint. `train.resume_from_checkpoint` is
separate: it resumes Trainer state such as the optimizer and scheduler.

### Custom tokenizer

Custom vocabulary adaptation is opt-in:

```json
{
  "tokenizer_path": "data/tokenizer_hf",
  "adapt_decoder_vocabulary": true,
  "vocabulary_init_strategy": "exact_or_subtokens",
  "allow_checkpoint_vocabulary_adaptation": false
}
```

The toolkit compares the complete token-to-ID mapping, not only vocabulary size.
If the mappings and dimensions already match, weights are left unchanged. When
adaptation is required, exact token rows are copied, compatible source subtokens
can be averaged, and only unmatched rows are initialized. Decoder transformer
blocks are preserved.

Changing the vocabulary of an existing checkpoint requires the additional
`allow_checkpoint_vocabulary_adaptation=true` switch and a fresh optimizer run.
The adaptation report is written to `vocabulary_adaptation.json` in the run and
final model directories.

### Generation settings

The `generation` section in the project config is the source of truth for test
and prediction behavior. It includes maximum generated tokens, beam and sampling
settings, repetition controls, and field-matching threshold. Before training or
inference, these values replace generation defaults inherited from an older
checkpoint. Saved checkpoints receive a portable `generation_config.json`
snapshot automatically.

`data.max_target_tokens` limits training targets. `generation.max_new_tokens`
limits newly generated tokens. The toolkit intentionally does not set the legacy
`max_length` alongside `max_new_tokens`.

## Prepare data

Split one JSONL file reproducibly:

```bash
python -m scripts.prepare_data split \
  --source data/all.jsonl \
  --output-dir data \
  --ratios 0.8 0.1 0.1 \
  --seed 42
```

Inspect target token lengths before choosing `max_target_tokens`:

```bash
python -m scripts.prepare_data analyze \
  --source data/train.jsonl \
  --tokenizer naver-clova-ix/donut-base-finetuned-cord-v2 \
  --text-field gt \
  --max-tokens 768
```

## Train

```bash
python -m scripts.train --config configs_own/train.json
```

The final best model, tokenizer, project config, generation config, and vocabulary
adaptation report are saved under `<train.output_dir>/best`. Intermediate
checkpoints are managed by Hugging Face Trainer.

Training behavior worth noting:

- Augmentation is applied only to the training split.
- Image size is always configured as height followed by width.
- The dataset emits labels; the collator shifts them into `decoder_input_ids` so
  label smoothing remains valid.
- Label padding uses `-100` and is excluded from the loss.
- The encoder is frozen for the configured number of epochs and then unfrozen.
- Validation loss selects the best checkpoint, and early stopping may finish
  before the configured epoch count.

## Test

Create a test config from the same template and set `model.checkpoint` to the
saved `best` directory:

```bash
cp configs/train.example.json configs_own/test.json
python -m scripts.test \
  --config configs_own/test.json \
  --output outputs/test_metrics.json
```

The output JSON contains:

- Exact Match and whitespace-insensitive Exact Match
- Character Error Rate and character accuracy
- Token Error Rate and token F1
- Normalized edit similarity and chrF
- Field-name, exact-value, and fuzzy-value precision/recall/F1
- Per-field results for structured targets
- The 20 most common character substitutions, deletions, and insertions

Field metrics support JSON-like structured targets and Donut-style tagged target
sequences. Plain-text targets still receive the non-field OCR metrics.

## Predict

```bash
python -m scripts.predict \
  --config configs_own/test.json \
  --image samples/receipt_a.png samples/receipt_b.png
```

Use `--device cuda`, `--device cuda:0`, or `--device cpu` to override automatic
device selection for testing and prediction.

## Checks

Run syntax checks and the unit test suite before publishing changes:

```bash
python -m compileall donut_training scripts donut_main.py
python -m unittest discover -s tests
```

This repository validates the training and evaluation workflow; model quality
depends on the dataset, labels, tokenizer, training setup, and held-out evaluation.
