"""
Cloud detection benchmark package.

This package provides tools for training and evaluating cloud detection models
on satellite imagery.
"""

__all__ = [
    # Bit depth estimation
    'get_bit_depth_range',
    'get_num_bit_depths',
    'get_bit_depth_values',
    'get_bit_depth_tensor',
    'pretty_print_dict',
    # Data
    'CloudDataset',
    # Losses/Metrics
    'intersection_over_union',
    # Configuration
    'BANDS',
    'CHIP_SIZE',
    'DEFAULT_TTA_MODES',
    # TTA
    'normalize_tta_modes',
    'apply_tta',
    'undo_tta',
    'predict_with_tta',
    # Evaluation
    'intersection_over_union',
    'read_prediction_and_aligned_true',
    # Prediction
    'load_cloud_model',
    'predict_small_chips',
    'save_prediction_geotiff',
    'fuse_probability_scores',
    # Metadata I/O
    'load_chip_metadata',
    'get_chip_metadata',
    'save_chip_metadata',
    # Tiling
    'GeoTIFFTiler',
    'ChipGenerator',
    'ChipWriter',
    # Visualization
    'display_thumbnail_with_prediction',
]

# Bit depth estimation
from benchmark.bit_depth_estimators import (
    get_bit_depth_range,
    get_num_bit_depths,
    get_bit_depth_values,
    get_bit_depth_tensor,
    pretty_print_dict,
)

# Data
from benchmark.cloud_dataset import CloudDataset

# Losses/Metrics
from benchmark.losses import intersection_over_union

# Configuration
from benchmark.config import BANDS, CHIP_SIZE, DEFAULT_TTA_MODES

# TTA
from benchmark.tta import normalize_tta_modes, apply_tta, undo_tta, predict_with_tta

# Evaluation
from benchmark.evaluation import intersection_over_union, read_prediction_and_aligned_true

# Prediction
from benchmark.prediction import (
    load_cloud_model,
    predict_small_chips,
    save_prediction_geotiff,
    fuse_probability_scores,
)

# Metadata I/O
from benchmark.metadata_io import load_chip_metadata, get_chip_metadata, save_chip_metadata

# Tiling
from benchmark.tiler import GeoTIFFTiler, ChipGenerator, ChipWriter

# Visualization
from benchmark.visualization import display_thumbnail_with_prediction
