"""
Cloud detection benchmark package.

This package provides tools for training and evaluating cloud detection models
on satellite imagery.
"""

__all__ = [
    # Core - Configuration
    "BANDS",
    "CHIP_SIZE",
    "OVERLAP_RATIO",
    "DATA_DIR",
    "DEFAULT_TTA_MODES",
    "NUM_CLASSES",
    # Core - Dataset and Model
    "CloudDataset",
    "CloudModel",
    # Core - Evaluation
    "intersection_over_union",
    "intersection_over_union_and_coverage",
    "read_prediction_and_aligned_true",
    # Core - Prediction
    "load_cloud_model",
    "predict_small_chips",
    "fuse_probability_scores",
    # Core - Metadata
    "load_chip_metadata",
    "get_chip_metadata",
    # Core - Tiling
    "GeoTIFFTiler",
    "ChipGenerator",
    "ChipWriter",
    # Utils - TTA
    "normalize_tta_modes",
    "apply_tta",
    "undo_tta",
    "predict_with_tta",
    # Utils - General
    "ensure_dir",
    "mask2label",
    "stretch",
    # Utils - Visualization
    "display_thumbnail_with_prediction",
    # Models
    "AdaptiveEncoderFactory",
    "get_bit_depth_range",
]

# Core imports
from benchmark.core.config import (
    BANDS,
    CHIP_SIZE,
    OVERLAP_RATIO,
    DATA_DIR,
    DEFAULT_TTA_MODES,
    NUM_CLASSES,
)
from benchmark.core.cloud_dataset import CloudDataset
from benchmark.core.cloud_model import CloudModel
from benchmark.core.evaluation import (
    intersection_over_union,
    intersection_over_union_and_coverage,
    read_prediction_and_aligned_true,
)
from benchmark.core.prediction import (
    load_cloud_model,
    predict_small_chips,
    fuse_probability_scores,
)
from benchmark.core.metadata_io import (
    load_chip_metadata,
    get_chip_metadata,
)
from benchmark.core.tiler import (
    GeoTIFFTiler,
    ChipGenerator,
    ChipWriter,
)

# Utils imports
from benchmark.utils.tta import (
    normalize_tta_modes,
    apply_tta,
    undo_tta,
    predict_with_tta,
)
from benchmark.utils.utils import (
    ensure_dir,
    mask2label,
    stretch,
)
# Visualization is imported separately to avoid circular imports
# from benchmark.utils.visualization import ...

# Models imports
from benchmark.models.adaptive_encoders import AdaptiveEncoderFactory
from benchmark.models.bit_depth_estimators import get_bit_depth_range
