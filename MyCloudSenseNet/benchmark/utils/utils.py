"""
Utility functions for cloud detection benchmark.

This module contains general-purpose utility functions used across the benchmark package.
"""

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt
from loguru import logger
import rasterio

from benchmark.core.config import (
    STRETCH_PERCENTILE,
    THUMBNAIL_SIZE,
    LabelRanges,
)


def mask2label(mask: np.ndarray) -> np.ndarray:
    """
    Convert raw mask values to class labels.

    Converts continuous mask values to discrete class labels:
    - Background: mask in [0, 50] -> label 0
    - Shadow: mask in [100, 200] -> label 1
    - Cloud: mask in [250, 255] -> label 2

    Args:
        mask: Input mask array with raw values

    Returns:
        Array with class labels (0, 1, 2)
    """
    label = np.zeros_like(mask, dtype=np.uint8)
    label[
        (mask >= LabelRanges.BACKGROUND[0]) & (mask <= LabelRanges.BACKGROUND[1])
    ] = 0
    label[
        (mask >= LabelRanges.SHADOW[0]) & (mask <= LabelRanges.SHADOW[1])
    ] = 1
    label[(mask >= LabelRanges.CLOUD[0]) & (mask <= LabelRanges.CLOUD[1])] = 2
    return label


def stretch(
    arr: np.ndarray,
    p: int = STRETCH_PERCENTILE,
    a_min: float = 0,
    a_max: float = 1,
) -> np.ndarray:
    """
    Apply percentile-based contrast stretching to an array.

    Args:
        arr: Input array
        p: Percentile for stretching (default: 2, meaning 2nd and 98th percentile)
        a_min: Minimum value of output range
        a_max: Maximum value of output range

    Returns:
        Stretched array clipped to [a_min, a_max]
    """
    lo = np.percentile(arr, p)
    hi = np.percentile(arr, 100 - p)
    range_val = hi - lo
    if range_val == 0:
        return np.clip(arr, a_min, a_max)
    return np.clip((arr - lo) / range_val, a_min, a_max)


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure

    Returns:
        The ensured directory path
    """
    path.mkdir(exist_ok=True, parents=True)
    return path


def round_up(value: int, divisor: int) -> int:
    """
    Round up a value to the nearest multiple of divisor.

    Args:
        value: Value to round up
        divisor: Divisor to round to

    Returns:
        Rounded up value
    """
    if divisor <= 1:
        return value
    return ((value + divisor - 1) // divisor) * divisor


def calculate_bit_depth(data: np.ndarray) -> int:
    """
    Calculate the bit depth of image data.

    Args:
        data: Input image data

    Returns:
        Estimated bit depth (e.g., 8, 10, 16)
    """
    max_val = data.max()
    if max_val <= 0:
        return 8
    return int(math.ceil(math.log2(max_val + 1)))


def is_label_valid(
    label: np.ndarray,
    valid_threshold: Tuple[float, float] = (0.001, 0.999),
) -> bool:
    """
    Check if a label chip is valid for training.

    A chip is valid if the percentage of valid pixels (non-background, non-invalid)
    falls within the specified threshold.

    Args:
        label: Label array
        valid_threshold: (min_ratio, max_ratio) for valid pixel percentage

    Returns:
        True if the label is valid, False otherwise
    """
    valid_pixel = np.sum((label > 0.01) & (label <= 2.0))
    total_pixel = label.size
    if total_pixel == 0:
        return False
    valid_percent = valid_pixel / total_pixel
    return valid_threshold[0] <= valid_percent <= valid_threshold[1]


def create_rgb_composite(
    data: np.ndarray,
    r_band: int = 2,
    g_band: int = 1,
    b_band: int = 0,
) -> np.ndarray:
    """
    Create RGB composite from multi-band image data.

    Args:
        data: Input image data with shape (bands, height, width) or (height, width, bands)
        r_band: Index of red band
        g_band: Index of green band
        b_band: Index of blue band

    Returns:
        RGB composite with shape (height, width, 3)
    """
    if data.ndim == 3:
        if data.shape[0] <= max(r_band, g_band, b_band):
            # Assume (height, width, bands) format
            rgb = np.dstack((data[:, :, r_band], data[:, :, g_band], data[:, :, b_band]))
        else:
            # Assume (bands, height, width) format
            rgb = np.dstack((data[r_band], data[g_band], data[b_band]))
    else:
        raise ValueError(f"Expected 3D array, got {data.ndim}D")

    return stretch(rgb)


def display_thumbnail(
    src: rasterio.DatasetReader,
    max_size: int = THUMBNAIL_SIZE,
) -> np.ndarray:
    """
    Read a thumbnail from a rasterio dataset.

    Args:
        src: Opened rasterio dataset
        max_size: Maximum thumbnail dimension

    Returns:
        Thumbnail data with shape (bands, height, width)
    """
    h, w = src.height, src.width
    if h <= max_size and w <= max_size:
        return src.read()

    scale = min(max_size / w, max_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return src.read(
        out_shape=(src.count, new_h, new_w),
        resampling=rasterio.enums.Resampling.bilinear,
    )


def create_comparison_figure(
    rgb: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray = None,
    iou: float = None,
    true_coverage: list = None,
    pred_coverage: list = None,
    title: str = "",
) -> plt.Figure:
    """
    Create a comparison figure with RGB, true mask, and predicted mask.

    Args:
        rgb: RGB composite image
        true_mask: Ground truth mask
        pred_mask: Predicted mask (optional)
        iou: IoU score (optional)
        true_coverage: List of true class coverages [shadow, cloud] (optional)
        pred_coverage: List of predicted class coverages [shadow, cloud] (optional)
        title: Figure title

    Returns:
        Matplotlib figure
    """
    num_plots = 3 if pred_mask is not None else 2
    fig, ax = plt.subplots(1, num_plots, figsize=(8 * num_plots, 8))

    if num_plots == 2:
        ax_rgb, ax_true = ax
    else:
        ax_rgb, ax_true, ax_pred = ax

    # RGB image
    ax_rgb.imshow(rgb)
    ax_rgb.set_title("RGB Image")
    ax_rgb.axis("off")

    # True mask
    true_title = "True Mask"
    if true_coverage:
        true_title += (
            f"\nshadow={true_coverage[0]:.4f} | cloud={true_coverage[1]:.4f}"
        )
    ax_true.imshow(true_mask)
    ax_true.set_title(true_title)
    ax_true.axis("off")

    # Predicted mask
    if pred_mask is not None:
        pred_title = "Predicted Mask"
        if iou is not None:
            pred_title += f" (IoU={iou:.4f})"
        if pred_coverage:
            pred_title += (
                f"\nshadow={pred_coverage[0]:.4f} | cloud={pred_coverage[1]:.4f}"
            )
        ax_pred.imshow(pred_mask)
        ax_pred.set_title(pred_title)
        ax_pred.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def validate_paths_exist(paths: Dict[str, Path]) -> None:
    """
    Validate that all paths in a dictionary exist.

    Args:
        paths: Dictionary of path names to Path objects

    Raises:
        FileNotFoundError: If any path does not exist
    """
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")


def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage statistics.

    Returns:
        Dictionary with memory usage information
    """
    import psutil

    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        "rss_mb": mem_info.rss / (1024 * 1024),
        "vms_mb": mem_info.vms / (1024 * 1024),
        "percent": process.memory_percent(),
    }
