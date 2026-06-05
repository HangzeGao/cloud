"""
Evaluation and metrics module for cloud detection.

This module provides evaluation metrics (IoU, coverage) and result comparison utilities.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio

from ..configs.defaults import INVALID_LABEL_VALUE, NUM_CLASSES, logger
from ..core.raster_utils import mask2label


def intersection_over_union_and_coverage(
    pred: np.ndarray,
    true: np.ndarray,
    n_classes: int = NUM_CLASSES,
) -> Tuple[float, list, list]:
    """
    Calculate Intersection over Union (IoU) and coverage metrics.

    .. deprecated::
        Use intersection_over_union() for more detailed results including per-class IoU.

    Args:
        pred: Predicted labels array
        true: Ground truth labels array
        n_classes: Number of classes

    Returns:
        Tuple of:
        - mIoU: Mean IoU across all classes (excluding background)
        - true_coverage_list: List of true class coverages per class (excluding background)
        - pred_coverage_list: List of predicted class coverages per class (excluding background)
    """
    mIoU, _, true_coverage_list, pred_coverage_list = intersection_over_union(
        pred, true, n_classes, ignore_invalid=True
    )
    return mIoU, true_coverage_list, pred_coverage_list


def intersection_over_union(
    pred: np.ndarray,
    true: np.ndarray,
    n_classes: int = NUM_CLASSES,
    ignore_invalid: bool = True,
) -> Tuple[float, list, list, list]:
    """
    Calculate Intersection over Union (IoU) and coverage metrics.

    Args:
        pred: Predicted labels array
        true: Ground truth labels array
        n_classes: Number of classes
        ignore_invalid: Whether to ignore invalid pixels (value=255)

    Returns:
        Tuple of:
        - mIoU: Mean IoU across all classes (excluding background)
        - iou_list: List of IoU values per class (excluding background)
        - true_coverage_list: List of true class coverages per class (excluding background)
        - pred_coverage_list: List of predicted class coverages per class (excluding background)

    Raises:
        ValueError: If pred and true shapes don't match
    """
    if pred.shape != true.shape:
        raise ValueError(
            f"pred and true must have the same shape, "
            f"got pred={pred.shape}, true={true.shape}. "
            "Use a chip prediction with the matching chip label, "
            "or use a restored full-image prediction."
        )

    total_pixels = true.size

    # Filter out invalid pixels if requested
    if ignore_invalid:
        valid_pixel_mask = true != INVALID_LABEL_VALUE
        true = true[valid_pixel_mask]
        pred = pred[valid_pixel_mask]
        total_pixels = len(true)

    iou_list = []
    true_coverage_list = []
    pred_coverage_list = []

    for cls in range(n_classes):
        # Skip background class (0) in metrics
        if cls == 0:
            continue

        # Create binary masks for current class
        true_cls = true == cls
        pred_cls = pred == cls

        # Calculate intersection and union
        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()

        # IoU with smoothing to avoid division by zero
        iou = intersection / (union + 1e-8)
        iou_list.append(min(iou, 1.0))

        # Calculate coverages
        true_coverage = true_cls.sum() / total_pixels if total_pixels > 0 else 0
        pred_coverage = pred_cls.sum() / total_pixels if total_pixels > 0 else 0
        true_coverage_list.append(true_coverage)
        pred_coverage_list.append(pred_coverage)

    # Mean IoU (excluding background)
    mIoU = np.mean(iou_list) if iou_list else 0.0

    return mIoU, iou_list, true_coverage_list, pred_coverage_list


def read_prediction_and_aligned_true(
    data_path: Path,
    pred_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read prediction and align it with ground truth dataset.

    Handles cases where prediction and ground truth have different resolutions
    or extents by using geospatial alignment.

    Args:
        data_path: Path to full image dataset with labels (bands 1-4=dataset, band 5=mask)
        pred_path: Path to prediction GeoTIFF

    Returns:
        Tuple of (prediction array, aligned ground truth array)

    Raises:
        ValueError: If alignment fails or CRS mismatch
    """
    with rasterio.open(data_path) as data_src, rasterio.open(pred_path) as pred_src:
        pred = pred_src.read(1).astype(np.uint8)

        # Check if shapes already match
        if pred_src.height == data_src.height and pred_src.width == data_src.width:
            true = mask2label(data_src.read(5).astype(np.uint8))
            return pred, true

        # Prediction is a chip/subset - check for transform
        if pred_src.transform.is_identity:
            raise ValueError(
                f"Prediction {pred_path} is smaller than the full image "
                f"but has no GeoTIFF transform. "
                "Regenerate chip predictions with predict_small_chips()/GeoTIFFTiler.predict(), "
                "or evaluate this prediction against its matching chip label_path."
            )

        # Check CRS compatibility
        if data_src.crs and pred_src.crs and data_src.crs != pred_src.crs:
            raise ValueError(
                f"Cannot align prediction to label because CRS differs: "
                f"pred={pred_src.crs}, true={data_src.crs}"
            )

        # Attempt geospatial alignment
        try:
            window = rasterio.windows.from_bounds(
                *pred_src.bounds, transform=data_src.transform
            )
            true_mask = data_src.read(
                5,
                window=window,
                out_shape=(pred_src.height, pred_src.width),
                boundless=False,
                resampling=rasterio.enums.Resampling.nearest,
            ).astype(np.uint8)
        except Exception as exc:
            raise ValueError(
                f"Prediction shape {pred.shape} does not match full label shape "
                f"({data_src.height}, {data_src.width}), "
                "and automatic geospatial alignment failed. "
                "For chip predictions, pass the matching chip label or "
                "save predictions with GeoTIFF transform."
            ) from exc

        true = mask2label(true_mask)

        if pred.shape != true.shape:
            raise ValueError(
                f"Aligned true mask still does not match prediction: "
                f"pred={pred.shape}, true={true.shape}"
            )

        return pred, true


def calculate_class_distribution(
    labels: np.ndarray,
    n_classes: int = NUM_CLASSES,
) -> dict:
    """
    Calculate the class distribution in a label array.

    Args:
        labels: Label array
        n_classes: Number of classes

    Returns:
        Dictionary with class counts and percentages
    """
    total = labels.size
    distribution = {}

    for cls in range(n_classes):
        count = np.sum(labels == cls)
        distribution[f"class_{cls}"] = {
            "count": int(count),
            "percentage": float(count / total) if total > 0 else 0.0,
        }

    return distribution


def evaluate_batch(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    n_classes: int = NUM_CLASSES,
) -> dict:
    """
    Evaluate a batch of predictions against ground truths.

    Args:
        predictions: Batch of predicted labels (B, H, W)
        ground_truths: Batch of ground truth labels (B, H, W)
        n_classes: Number of classes

    Returns:
        Dictionary with evaluation metrics
    """
    batch_size = predictions.shape[0]
    batch_ious = []
    batch_coverages = []

    for i in range(batch_size):
        mIoU, iou_list, true_cov, pred_cov = intersection_over_union(
            predictions[i], ground_truths[i], n_classes
        )
        batch_ious.append(mIoU)
        batch_coverages.append({
            "true": true_cov,
            "pred": pred_cov,
        })

    return {
        "mean_iou": float(np.mean(batch_ious)),
        "std_iou": float(np.std(batch_ious)),
        "per_sample_iou": batch_ious,
        "per_sample_coverage": batch_coverages,
    }
