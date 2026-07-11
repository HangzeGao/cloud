"""
Configuration module for cloud detection benchmark.

This module contains all configuration constants used throughout the benchmark package.
"""

import logging
import sys
from pathlib import Path
from typing import Final, Sequence, Tuple

# Setup logger - compatible with both loguru and standard logging
try:
    from loguru import logger
except ImportError:
    # Fallback to standard logging
    logger = logging.getLogger("cloud_benchmark")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Data Configuration
# ---------------------------------------------------------------------------

# Canonical model inputs: Blue, Green, Red. NIR is intentionally excluded so
# training matches deployment scenes where it may not be available.
BANDS: Final[list[str]] = ["B02", "B03", "B04"]

# Image chip size for tiling (pixels)
CHIP_SIZE: Final[int] = 512

# Overlap ratio for sliding window tiling (0.2 = 20% overlap)
OVERLAP_RATIO: Final[float] = 0.2

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

# Number of output classes (background, cloud). Shadows are folded into
# background during dataset preparation.
NUM_CLASSES: Final[int] = 2

# ---------------------------------------------------------------------------
# Prediction Configuration
# ---------------------------------------------------------------------------

# Padding divisor for prediction batching (should match model requirements)
PREDICTION_PAD_DIVISOR: Final[int] = 32

# Probability fusion method for overlapping predictions
FUSION_METHOD: Final[str] = "probability"

# ---------------------------------------------------------------------------
# Raster/GeoTIFF Configuration
# ---------------------------------------------------------------------------

# Default compression for output GeoTIFF files
DEFAULT_COMPRESSION: Final[str] = "deflate"

# Default predictor for output GeoTIFF files
DEFAULT_PREDICTOR: Final[int] = 2

# Default dtype for output predictions
OUTPUT_DTYPE: Final[str] = "uint8"

# Default dtype for intermediate probability storage
PROBABILITY_DTYPE: Final[str] = "float16"

# ---------------------------------------------------------------------------
# Label Encoding Configuration
# ---------------------------------------------------------------------------

# Mask value ranges for different classes (based on input format)
class LabelRanges:
    """Label value ranges for different cloud classes."""

    BACKGROUND: Tuple[int, int] = (0, 50)  # Background/clear
    SHADOW: Tuple[int, int] = (100, 200)  # Cloud shadow
    CLOUD: Tuple[int, int] = (250, 255)  # Cloud


# Invalid label value (to be ignored in evaluation)
INVALID_LABEL_VALUE: Final[int] = 255

# ---------------------------------------------------------------------------
# Visualization Configuration
# ---------------------------------------------------------------------------

# Default thumbnail size for visualization
THUMBNAIL_SIZE: Final[int] = 512

# Default percentile for contrast stretching
STRETCH_PERCENTILE: Final[int] = 2

# ---------------------------------------------------------------------------
# Processing Configuration
# ---------------------------------------------------------------------------

# Default number of workers for parallel processing
DEFAULT_MAX_WORKERS: Final[int] = 32
