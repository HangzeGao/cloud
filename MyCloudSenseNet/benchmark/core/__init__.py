"""
Core module for cloud detection benchmark.

Contains the main business logic for:
- Configuration management (config)
- Dataset handling (cloud_dataset)
- Model definitions (cloud_model)
- Evaluation metrics (evaluation)
- Prediction pipeline (prediction)
- Metadata I/O (metadata_io)
- Tiling functionality (tiler)
"""

from benchmark.core.config import (
    BANDS,
    CHIP_SIZE,
    OVERLAP_RATIO,
    VALID_THRESHOLD,
    DATA_DIR,
    MODEL_NAME,
    DEFAULT_TTA_MODES,
    PREDICTION_PAD_DIVISOR,
    DEFAULT_COMPRESSION,
    DEFAULT_PREDICTOR,
    OUTPUT_DTYPE,
    NUM_CLASSES,
    LabelRanges,
    THUMBNAIL_SIZE,
    STRETCH_PERCENTILE,
    DEFAULT_MAX_WORKERS,
)

from benchmark.core.cloud_dataset import CloudDataset
from benchmark.core.cloud_model import CloudModel
from benchmark.core.evaluation import (
    intersection_over_union,
    intersection_over_union_and_coverage,
    read_prediction_and_aligned_true,
    calculate_class_distribution,
    evaluate_batch,
)
from benchmark.core.prediction import (
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
from benchmark.core.metadata_io import (
    load_chip_metadata,
    get_chip_metadata,
    save_chip_metadata,
    merge_chip_metadata,
    filter_valid_chips,
    get_chip_statistics,
    required_chip_columns,
)
from benchmark.core.tiler import (
    GeoTIFFTiler,
    ChipGenerator,
    ChipWriter,
    ChipWindow,
    ImageInfo,
    GeoTIFFReader,
)

__all__ = [
    # Config
    "BANDS",
    "CHIP_SIZE",
    "OVERLAP_RATIO",
    "VALID_THRESHOLD",
    "DATA_DIR",
    "MODEL_NAME",
    "DEFAULT_TTA_MODES",
    "PREDICTION_PAD_DIVISOR",
    "DEFAULT_COMPRESSION",
    "DEFAULT_PREDICTOR",
    "OUTPUT_DTYPE",
    "NUM_CLASSES",
    "LabelRanges",
    "THUMBNAIL_SIZE",
    "STRETCH_PERCENTILE",
    "DEFAULT_MAX_WORKERS",
    # Dataset and Model
    "CloudDataset",
    "CloudModel",
    # Evaluation
    "intersection_over_union",
    "intersection_over_union_and_coverage",
    "read_prediction_and_aligned_true",
    "calculate_class_distribution",
    "evaluate_batch",
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
    "save_chip_metadata",
    "merge_chip_metadata",
    "filter_valid_chips",
    "get_chip_statistics",
    "required_chip_columns",
    # Tiler
    "GeoTIFFTiler",
    "ChipGenerator",
    "ChipWriter",
    "ChipWindow",
    "ImageInfo",
    "GeoTIFFReader",
]
