"""Raster and label helpers used by core GeoTIFF components."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from ..configs.defaults import LabelRanges, NUM_CLASSES, STRETCH_PERCENTILE, THUMBNAIL_SIZE


def mask2label(mask: np.ndarray) -> np.ndarray:
    """Convert raw mask values to class labels."""
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
    """Apply percentile-based contrast stretching."""
    lo = np.percentile(arr, p)
    hi = np.percentile(arr, 100 - p)
    range_val = hi - lo
    if range_val == 0:
        return np.clip(arr, a_min, a_max)
    return np.clip((arr - lo) / range_val, a_min, a_max)


def calculate_bit_depth(data: np.ndarray) -> int:
    """Estimate bit depth from image dataset."""
    max_val = data.max()
    if max_val <= 0:
        return 8
    return int(math.ceil(math.log2(max_val + 1)))


def label_class_stats(
    label: np.ndarray,
    num_classes: int = NUM_CLASSES,
    prefix: str = "label_class",
) -> dict[str, float | int]:
    """Return per-class pixel ratios for a label chip.

    Class 0 is background in this benchmark.
    """
    if label.size and int(np.nanmax(label)) >= num_classes:
        label = mask2label(label)

    total = int(label.size)
    stats: dict[str, float | int] = {}

    if total == 0:
        for class_id in range(num_classes):
            stats[f"{prefix}_{class_id}_ratio"] = 0.0
        stats["label_non_background_ratio"] = 0.0
        stats["dominant_label_class"] = -1
        return stats

    counts = np.bincount(label.astype(np.int64).ravel(), minlength=num_classes)
    counts = counts[:num_classes]

    for class_id, count in enumerate(counts):
        stats[f"{prefix}_{class_id}_ratio"] = float(count / total)

    stats["label_non_background_ratio"] = 1.0 - float(stats[f"{prefix}_0_ratio"])
    stats["dominant_label_class"] = int(np.argmax(counts))
    return stats


def create_rgb_composite(
    data: np.ndarray,
    r_band: int = 2,
    g_band: int = 1,
    b_band: int = 0,
) -> np.ndarray:
    """Create an RGB composite from multi-band image dataset."""
    if data.ndim == 3:
        if data.shape[0] <= max(r_band, g_band, b_band):
            rgb = np.dstack((data[:, :, r_band], data[:, :, g_band], data[:, :, b_band]))
        else:
            rgb = np.dstack((data[r_band], data[g_band], data[b_band]))
    else:
        raise ValueError(f"Expected 3D array, got {data.ndim}D")

    return stretch(rgb)


def display_thumbnail(
    src,
    max_size: int = THUMBNAIL_SIZE,
) -> np.ndarray:
    """Read a thumbnail from a rasterio dataset."""
    h, w = src.height, src.width
    if h <= max_size and w <= max_size:
        return src.read()

    import rasterio

    scale = min(max_size / w, max_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return src.read(
        out_shape=(src.count, new_h, new_w),
        resampling=rasterio.enums.Resampling.bilinear,
    )
