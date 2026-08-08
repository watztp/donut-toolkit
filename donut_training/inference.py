"""Batch generation helpers shared by prediction and evaluation."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.inference_mode()
def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pixel_values: torch.Tensor,
    *,
    device: torch.device | None = None,
    prompt: str | None = None,
    strip_after: str | None = None,
) -> tuple[list[str], list[list[int]]]:
    """Generate decoded text and token IDs for a preprocessed image batch."""

    target_device = device or next(model.parameters()).device
    if pixel_values.ndim == 3:
        pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.to(target_device, non_blocking=True)
    model.eval()

    decoder_input_ids: torch.Tensor | None = None
    prompt_length = 0
    if prompt:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_length = len(prompt_ids)
        decoder_input_ids = torch.tensor(prompt_ids, device=target_device).unsqueeze(0)
        decoder_input_ids = decoder_input_ids.repeat(pixel_values.shape[0], 1)

    output = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    token_rows = output[:, prompt_length:].detach().cpu().tolist() if prompt_length else output.detach().cpu().tolist()
    texts = tokenizer.batch_decode(token_rows, skip_special_tokens=True)
    cleaned: list[str] = []
    for text in texts:
        if strip_after and strip_after in text:
            text = text.split(strip_after, 1)[0]
        cleaned.append(text.strip())
    return cleaned, token_rows


def select_device(requested: str | None = None) -> torch.device:
    """Resolve an explicit device or choose CUDA, MPS, then CPU."""

    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
