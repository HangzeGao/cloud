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

# Default spectral bands to use (Sentinel-2: B02=Blue, B03=Green, B04=Red, B08=NIR)
BANDS: Final[list[str]] = ["B02", "B03", "B04", "B08"]

# Image chip size for tiling (pixels)
CHIP_SIZE: Final[int] = 512

# Overlap ratio for sliding window tiling (0.2 = 20% overlap)
OVERLAP_RATIO: Final[float] = 0.2

# Valid pixel threshold for training chips [min_ratio, max_ratio]
# Chips with valid pixel percentage outside this range will be filtered out
VALID_THRESHOLD: Final[list[float]] = [0.001, 0.999]

# Default dataset directory (relative to current working directory)
DATA_DIR: Final[Path] = Path.cwd().parent.resolve() / "dataset"

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

# Default model architecture name
MODEL_NAME: Final[str] = "unet"

# Default model weights path (relative to notebook directory)
DEFAULT_MODEL_WEIGHTS_PATH: Final[Path] = Path("unet/assets/cloud_model.pt")

# Number of output classes (background, shadow, cloud)
NUM_CLASSES: Final[int] = 3

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
DEFAULT_MAX_WORKERS: Final[int] = 4
