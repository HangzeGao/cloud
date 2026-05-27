"""
Metadata and prediction utilities for cloud detection benchmark.

.. deprecated::
    This module is kept for backward compatibility.
    New code should import directly from the specialized modules:
    - benchmark.config: Configuration constants
    - benchmark.utils: Utility functions
    - benchmark.tta: Test-Time Augmentation
    - benchmark.evaluation: Evaluation metrics
    - benchmark.prediction: Prediction utilities
    - benchmark.metadata_io: Metadata I/O
    - benchmark.tiler: GeoTIFF tiling
    - benchmark.visualization: Visualization
"""

import warnings

# Configuration
from benchmark.config import (
    BANDS,
    CHIP_SIZE,
    OVERLAP_RATIO,
    VALID_THRESHOLD,
    DATA_DIR,
    MODEL_NAME,
    DEFAULT_TTA_MODES,
    PREDICTION_PAD_DIVISOR,
)

# Utilities
from benchmark.utils import (
    mask2label,
    stretch,
    ensure_dir,
    round_up,
    is_label_valid,
    create_rgb_composite,
    calculate_bit_depth,
)

# TTA
from benchmark.tta import (
    normalize_tta_modes,
    apply_tta,
    undo_tta,
    predict_with_tta as predict_batch_probabilities,
)

# Evaluation
from benchmark.evaluation import (
    intersection_over_union,
    intersection_over_union_and_coverage,
    read_prediction_and_aligned_true,
)

# Prediction
from benchmark.prediction import (
    default_model_weights_path,
    load_cloud_model,
    maybe_fast_dev,
    build_prediction_dataloader,
    iter_chip_probability_batches,
    pad_prediction_batch,
    save_prediction_geotiff,
    crop_prediction_to_shape,
    save_chip_predictions,
    predict_small_chips,
    fuse_probability_scores,
)

# Metadata I/O
from benchmark.metadata_io import (
    load_chip_metadata,
    get_chip_metadata,
    required_chip_columns,
)

# Tiler
from benchmark.tiler import GeoTIFFTiler

# Visualization
from benchmark.visualization import display_thumbnail_with_prediction as display_thumbnail_more

# Compatibility aliases (already imported from evaluation module)

# Compatibility: original function name for fuse_probability_scores
# No change needed - same name

# Compatibility: original display function name
def display_thumbnail_more_compat(*args, **kwargs):
    """Backward-compatible display function."""
    return display_thumbnail_more(*args, **kwargs)


# Compatibility warning
warnings.warn(
    "The 'metadata' module is deprecated. "
    "Please import directly from specialized modules: "
    "benchmark.config, benchmark.utils, benchmark.prediction, etc.",
    DeprecationWarning,
    stacklevel=2,
)


__all__ = [
    # Configuration
    "BANDS",
    "CHIP_SIZE",
    "OVERLAP_RATIO",
    "VALID_THRESHOLD",
    "DATA_DIR",
    "MODEL_NAME",
    "DEFAULT_TTA_MODES",
    "PREDICTION_PAD_DIVISOR",
    # Utilities
    "mask2label",
    "stretch",
    "is_label_valid",
    "ensure_dir",
    "round_up",
    "calculate_bit_depth",
    "create_rgb_composite",
    # TTA
    "normalize_tta_modes",
    "apply_tta",
    "undo_tta",
    "predict_batch_probabilities",
    # Evaluation
    "intersection_over_union",
    "intersection_over_union_and_coverage",
    "read_prediction_and_aligned_true",
    # Prediction
    "default_model_weights_path",
    "load_cloud_model",
    "maybe_fast_dev",
    "build_prediction_dataloader",
    "iter_chip_probability_batches",
    "pad_prediction_batch",
    "save_prediction_geotiff",
    "crop_prediction_to_shape",
    "save_chip_predictions",
    "predict_small_chips",
    "fuse_probability_scores",
    # Metadata I/O
    "load_chip_metadata",
    "get_chip_metadata",
    "required_chip_columns",
    # Tiler
    "GeoTIFFTiler",
    # Visualization
    "display_thumbnail_more",
]
