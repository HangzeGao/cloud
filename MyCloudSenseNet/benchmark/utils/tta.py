"""
Test-Time Augmentation (TTA) module for cloud detection.

This module provides TTA functionality for improving prediction accuracy
by applying augmentations during inference and averaging results.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from benchmark.core.config import DEFAULT_TTA_MODES, VALID_TTA_MODES
from benchmark.core.cloud_model import CloudModel


def normalize_tta_modes(
    tta_modes: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    """
    Normalize and validate TTA mode specifications.

    Args:
        tta_modes: Sequence of TTA mode names, or None for defaults

    Returns:
        Tuple of validated TTA mode names (always includes 'none')

    Raises:
        ValueError: If unknown TTA modes are specified
    """
    if tta_modes is None:
        return DEFAULT_TTA_MODES
    if not tta_modes:
        return ("none",)

    modes = tuple(tta_modes)
    unknown = sorted(set(modes) - VALID_TTA_MODES)
    if unknown:
        raise ValueError(
            f"Unknown TTA modes: {unknown}. "
            f"Choose from {sorted(VALID_TTA_MODES)}"
        )

    # Ensure 'none' is always included
    if "none" not in modes:
        modes = ("none",) + modes

    return modes


def apply_tta(image: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Apply TTA transformation to an image tensor.

    Args:
        image: Input image tensor with shape (..., H, W)
        mode: TTA mode name. Defaults use D4 modes:
            'none', 'rot90', 'rot180', 'rot270',
            'hflip', 'vflip', 'diag', 'anti_diag'.
            'hvflip' is also accepted for backward compatibility.

    Returns:
        Transformed image tensor
    """
    if mode == "hflip":
        return torch.flip(image, dims=(-1,))
    if mode == "vflip":
        return torch.flip(image, dims=(-2,))
    if mode in {"hvflip", "rot180"}:
        return torch.flip(image, dims=(-2, -1))
    if mode == "rot90":
        return torch.rot90(image, k=1, dims=(-2, -1))
    if mode == "rot270":
        return torch.rot90(image, k=3, dims=(-2, -1))
    if mode == "diag":
        return image.transpose(-2, -1)
    if mode == "anti_diag":
        return torch.flip(image.transpose(-2, -1), dims=(-2, -1))
    return image


def undo_tta(pred: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Undo TTA transformation from prediction tensor.

    Note: Most D4 transforms are self-inverse. 90 and 270 degree rotations
    undo each other.

    Args:
        pred: Prediction tensor with shape (..., H, W)
        mode: TTA mode name that was applied

    Returns:
        Untransformed prediction tensor
    """
    if mode == "rot90":
        return torch.rot90(pred, k=3, dims=(-2, -1))
    if mode == "rot270":
        return torch.rot90(pred, k=1, dims=(-2, -1))
    return apply_tta(pred, mode)


def predict_with_tta(
    model: CloudModel,
    x: torch.Tensor,
    tta_modes: Optional[Sequence[str]] = DEFAULT_TTA_MODES,
) -> torch.Tensor:
    """
    Predict with Test-Time Augmentation.

    Applies multiple augmentations during inference and averages the results.

    Args:
        model: Cloud detection model
        x: Input tensor with shape (B, C, H, W)
        tta_modes: Sequence of TTA modes to use

    Returns:
        Averaged probability tensor with shape (B, num_classes, H, W)
    """
    modes = normalize_tta_modes(tta_modes)
    prob_sum: Optional[torch.Tensor] = None

    for mode in modes:
        # Apply augmentation, predict, then undo augmentation
        x_aug = apply_tta(x, mode)
        logits = model(x_aug)
        logits = undo_tta(logits, mode)
        probs = F.softmax(logits, dim=1)

        if prob_sum is None:
            prob_sum = probs
        else:
            prob_sum = prob_sum + probs

    return prob_sum / len(modes)


def get_tta_mode_count(tta_modes: Optional[Sequence[str]] = None) -> int:
    """
    Get the number of TTA modes that will be used.

    Args:
        tta_modes: Sequence of TTA mode names, or None for defaults

    Returns:
        Number of TTA modes
    """
    modes = normalize_tta_modes(tta_modes)
    return len(modes)
