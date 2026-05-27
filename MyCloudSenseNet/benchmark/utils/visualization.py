"""
Visualization module for cloud detection results.

This module provides functions for visualizing images, predictions, and evaluation results.
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from loguru import logger

from benchmark.utils.utils import create_rgb_composite, display_thumbnail
from benchmark.core.evaluation import read_prediction_and_aligned_true, intersection_over_union


def display_thumbnail_with_prediction(
    data_path: Path,
    pred_path: Optional[Path] = None,
    max_size: int = 512,
) -> plt.Figure:
    """
    Display a thumbnail of the data with optional prediction overlay.

    Args:
        data_path: Path to the data GeoTIFF
        pred_path: Optional path to prediction GeoTIFF
        max_size: Maximum thumbnail size

    Returns:
        Matplotlib figure
    """
    logger.info(f"Displaying thumbnail for {data_path.name}")

    # Read data thumbnail
    with rasterio.open(data_path) as src:
        thumbnail = display_thumbnail(src, max_size)

    # Extract bands
    b, g, r, nir = thumbnail[:4]
    mask = thumbnail[4] if thumbnail.shape[0] > 4 else None

    # Create RGB
    rgb = create_rgb_composite(np.stack([b, g, r]))

    # Setup figure
    if pred_path and pred_path.exists():
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    # RGB image
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    # True mask
    if mask is not None:
        axes[1].imshow(mask)
        axes[1].set_title("True Mask")
        axes[1].axis("off")

    # Prediction with metrics
    if pred_path and pred_path.exists():
        pred, true_for_metric = read_prediction_and_aligned_true(data_path, pred_path)

        # Read prediction thumbnail
        with rasterio.open(pred_path) as src2:
            pred_mask = display_thumbnail(src2, max_size)

        # Calculate metrics
        mIoU, iou_list, true_cov, pred_cov = intersection_over_union(pred, true_for_metric)

        # Update titles with metrics
        if len(axes) > 1 and mask is not None:
            axes[1].set_title(
                f"True Mask\n"
                f"shadow={true_cov[0]:.4f} | cloud={true_cov[1]:.4f}"
            )

        if len(axes) > 2:
            axes[2].imshow(pred_mask[0] if pred_mask.ndim == 3 else pred_mask)
            axes[2].set_title(
                f"Predicted Mask (mIoU={mIoU:.4f})\n"
                f"shadow={pred_cov[0]:.4f} | cloud={pred_cov[1]:.4f}"
            )
            axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    return fig


def plot_class_distribution(
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Plot the class distribution in a label array.

    Args:
        labels: Label array
        class_names: Optional list of class names

    Returns:
        Matplotlib figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    percentages = counts / labels.size * 100

    if class_names is None:
        class_names = [f"Class {u}" for u in unique]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(unique)), percentages)

    # Color bars
    colors = ['gray', 'blue', 'white']
    for bar, color in zip(bars, colors[:len(bars)]):
        bar.set_color(color)
        bar.set_edgecolor('black')

    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels(class_names[:len(unique)])
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Class Distribution")
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for i, (count, pct) in enumerate(zip(counts, percentages)):
        ax.text(i, pct + 2, f"{count}\n({pct:.1f}%)", ha='center', va='bottom')

    plt.tight_layout()
    return fig


def plot_iou_comparison(
    iou_scores: List[float],
    labels: Optional[List[str]] = None,
    title: str = "IoU Comparison",
) -> plt.Figure:
    """
    Plot IoU scores comparison.

    Args:
        iou_scores: List of IoU scores
        labels: Optional list of labels for each score
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(iou_scores))
    bars = ax.bar(x, iou_scores)

    # Color by score
    for bar, score in zip(bars, iou_scores):
        if score >= 0.8:
            bar.set_color('green')
        elif score >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    ax.set_ylim(0, 1)
    ax.set_ylabel("IoU Score")
    ax.set_title(title)

    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')

    # Add value labels
    for i, score in enumerate(iou_scores):
        ax.text(i, score + 0.02, f"{score:.3f}", ha='center', va='bottom')

    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_prediction_comparison_grid(
    data_paths: List[Path],
    pred_paths: List[Path],
    max_cols: int = 3,
    max_size: int = 256,
) -> plt.Figure:
    """
    Plot a grid comparing multiple predictions.

    Args:
        data_paths: List of data file paths
        pred_paths: List of prediction file paths
        max_cols: Maximum number of columns
        max_size: Maximum thumbnail size

    Returns:
        Matplotlib figure
    """
    n = len(data_paths)
    n_cols = min(n, max_cols)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=(5 * n_cols, 5 * n_rows * 3))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows * 3, n_cols)

    for idx, (data_path, pred_path) in enumerate(zip(data_paths, pred_paths)):
        row = (idx // n_cols) * 3
        col = idx % n_cols

        with rasterio.open(data_path) as src:
            thumbnail = display_thumbnail(src, max_size)
            b, g, r = thumbnail[:3]
            rgb = create_rgb_composite(np.stack([r, g, b]))
            mask = thumbnail[3] if thumbnail.shape[0] > 3 else None

        # RGB
        axes[row, col].imshow(rgb)
        axes[row, col].set_title(f"{data_path.stem}\nRGB")
        axes[row, col].axis("off")

        # True mask
        if mask is not None:
            axes[row + 1, col].imshow(mask)
            axes[row + 1, col].set_title("True Mask")
            axes[row + 1, col].axis("off")

        # Prediction
        if pred_path.exists():
            try:
                pred, true = read_prediction_and_aligned_true(data_path, pred_path)
                mIoU, _, _, _ = intersection_over_union(pred, true)

                with rasterio.open(pred_path) as src:
                    pred_thumb = display_thumbnail(src, max_size)

                axes[row + 2, col].imshow(pred_thumb[0] if pred_thumb.ndim == 3 else pred_thumb)
                axes[row + 2, col].set_title(f"Prediction\nmIoU={mIoU:.3f}")
                axes[row + 2, col].axis("off")
            except Exception as e:
                axes[row + 2, col].text(0.5, 0.5, f"Error: {str(e)[:30]}",
                                       ha='center', va='center', transform=axes[row + 2, col].transAxes)
                axes[row + 2, col].axis("off")

    # Hide unused subplots
    for idx in range(n, n_rows * n_cols):
        row = (idx // n_cols) * 3
        col = idx % n_cols
        for r in range(3):
            if row + r < axes.shape[0]:
                axes[row + r, col].axis("off")

    plt.tight_layout()
    return fig
