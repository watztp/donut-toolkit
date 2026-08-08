"""Receipt-oriented augmentation and SwinV2 image preprocessing."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal, TypeAlias

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ImageInput: TypeAlias = Image.Image | np.ndarray | torch.Tensor
Augmentation: TypeAlias = Callable[..., object]


def build_receipt_augmentation(
    strength: Literal["light", "mid", "hard"] = "light",
    keep_clean_probability: float = 0.4,
) -> A.Compose:
    """Create conservative geometric/noise augmentation for receipt images."""

    if not 0.0 <= keep_clean_probability <= 1.0:
        raise ValueError("keep_clean_probability must be between 0 and 1")

    settings = {
        "light": (5.0, 0.03, 0.15, (0.01, 0.03)),
        "mid": (2.0, 0.05, 0.25, (0.02, 0.06)),
        "hard": (3.0, 0.07, 0.35, (0.04, 0.10)),
    }
    if strength not in settings:
        raise ValueError(f"Unknown augmentation strength: {strength}")
    rotation, perspective, blur_probability, noise_range = settings[strength]

    return A.Compose(
        [
            A.OneOf(
                [
                    A.NoOp(p=keep_clean_probability),
                    A.Compose(
                        [
                            A.SafeRotate(limit=rotation, border_mode=cv2.BORDER_REPLICATE, p=1.0),
                            A.Perspective(scale=(0.03, perspective), keep_size=True, fit_output=True, p=0.8),
                            A.RandomScale(scale_limit=0.05, p=0.5),
                        ],
                        p=1.0,
                    ),
                ],
                p=1.0,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=0.2),
                    A.RandomBrightnessContrast(0.05, 0.05, p=0.5),
                    A.RandomGamma(gamma_limit=(90, 110), p=0.3),
                ],
                p=0.8,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=noise_range, mean_range=(0.0, 0.0), p=0.6),
                    A.ISONoise(intensity=(0.1, 0.3), p=0.2),
                    A.NoOp(p=0.2),
                ],
                p=0.6,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=0.2),
                    A.GaussianBlur(blur_limit=3, p=blur_probability),
                    A.NoOp(p=1.0 - blur_probability),
                ],
                p=0.5,
            ),
        ]
    )


def _to_numpy_rgb(image: ImageInput) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        array = image
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = array * 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError("Image must have shape HxW, HxWx1, HxWx3, or HxWx4")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return np.clip(array[..., :3], 0, 255).astype(np.uint8)


def _letterbox(
    tensor: torch.Tensor,
    target_size: tuple[int, int],
    keep_ratio: bool,
    fill_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize CHW tensor and return both padded pixels and a content mask."""

    target_height, target_width = target_size
    _, height, width = tensor.shape
    if keep_ratio:
        scale = min(target_height / height, target_width / width)
        new_height = max(1, round(height * scale))
        new_width = max(1, round(width * scale))
    else:
        new_height, new_width = target_height, target_width

    resized = F.interpolate(
        tensor.unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=False
    ).squeeze(0)
    pad_height = target_height - new_height
    pad_width = target_width - new_width
    left, right = pad_width // 2, pad_width - pad_width // 2
    top, bottom = pad_height // 2, pad_height - pad_height // 2
    padded = F.pad(resized, (left, right, top, bottom), value=fill_value)
    mask = torch.zeros((target_height, target_width), dtype=torch.float32)
    mask[top : top + new_height, left : left + new_width] = 1.0
    return padded, mask


def process_images(
    images: ImageInput | Iterable[ImageInput],
    *,
    target_size: tuple[int, int] = (1280, 736),
    keep_ratio: bool = True,
    fill_value: float = 0.0,
    grayscale: bool = True,
    augmentation: A.Compose | None = None,
    return_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Convert one image or a collection into ImageNet-normalized CHW tensors.

    ``target_size`` is always ``(height, width)``. Augmentation is intentionally
    applied before letterboxing so padded borders do not become training signal.
    """

    is_single = isinstance(images, (Image.Image, np.ndarray, torch.Tensor))
    image_list = [images] if is_single else list(images)
    if not image_list:
        raise ValueError("At least one image is required")

    processed: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for image in image_list:
        array = _to_numpy_rgb(image)
        if augmentation is not None:
            array = augmentation(image=array)["image"]
        # PIL/Albumentations may expose a read-only NumPy view. ``from_numpy``
        # shares that memory, so create an explicit writable C-order copy first.
        writable_array = np.array(array, copy=True, order="C")
        tensor = torch.from_numpy(writable_array).permute(2, 0, 1).float() / 255.0
        if grayscale:
            gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
            tensor = gray.unsqueeze(0).repeat(3, 1, 1)
        tensor, mask = _letterbox(tensor, target_size, keep_ratio, fill_value)
        processed.append(tensor)
        masks.append(mask)

    batch = torch.stack(processed)
    mean = batch.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = batch.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    batch = (batch - mean) / std
    mask_batch = torch.stack(masks)

    pixels = batch[0] if is_single else batch
    mask_output = mask_batch[0] if is_single else mask_batch
    return (pixels, mask_output) if return_mask else pixels
