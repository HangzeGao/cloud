"""
Prediction module for cloud detection.

This module handles model loading, batch prediction, and result saving for both
small chips and large image tiles.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from benchmark.cloud_model import CloudModel
from benchmark.cloud_dataset import CloudDataset
from benchmark.config import (
    BANDS,
    DEFAULT_TTA_MODES,
    MODEL_NAME,
    NUM_CLASSES,
    PREDICTION_PAD_DIVISOR,
    PROBABILITY_DTYPE,
    DEFAULT_COMPRESSION,
    DEFAULT_PREDICTOR,
    OUTPUT_DTYPE,
)
from benchmark.tta import predict_with_tta, normalize_tta_modes
from benchmark.utils import ensure_dir, round_up


def default_model_weights_path(model_name: str = MODEL_NAME) -> Path:
    """
    Get the default path for model weights.

    Args:
        model_name: Model architecture name

    Returns:
        Path to model weights file
    """
    return Path(f"benchmark/{model_name}/assets/cloud_model.pt")


def load_cloud_model(
    model_weights_path: Path,
    bands: List[str] = BANDS,
    model_name: str = MODEL_NAME,
) -> CloudModel:
    """
    Load a cloud detection model from weights.

    Args:
        model_weights_path: Path to model weights file
        bands: List of band names
        model_name: Model architecture name

    Returns:
        Loaded CloudModel in eval mode
    """
    logger.info(f"Loading model from {model_weights_path}")

    model = CloudModel(
        bands=bands,
        hparams={"weights": None},
        model_name=model_name,
    )

    # Determine device for loading
    device_type = getattr(model, "device_type", "cpu")
    map_location = (
        torch.device(device_type)
        if device_type in ("cuda", "mps")
        else torch.device("cpu")
    )

    # Load state dict
    state_dict = torch.load(model_weights_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def maybe_fast_dev(
    metadata: pd.DataFrame,
    model: CloudModel,
    fast_dev_run: bool,
) -> pd.DataFrame:
    """
    Limit metadata to a small subset for fast development runs.

    Args:
        metadata: Chip metadata DataFrame
        model: CloudModel (used to get batch_size)
        fast_dev_run: Whether to enable fast development mode

    Returns:
        Potentially truncated metadata DataFrame
    """
    if not fast_dev_run:
        return metadata
    return metadata.head(max(model.batch_size, 1))


def pad_prediction_batch(
    batch: List[Dict[str, Any]],
    pad_divisor: int = PREDICTION_PAD_DIVISOR,
) -> Dict[str, Any]:
    """
    Pad a batch of chips to consistent size for batch prediction.

    Args:
        batch: List of chip dictionaries with 'chip' and 'chip_id' keys
        pad_divisor: Divisor to round up to (for model requirements)

    Returns:
        Dictionary with padded batch tensor and metadata
    """
    import rasterio

    chips = [torch.as_tensor(item["chip"], dtype=torch.float32) for item in batch]
    shapes = [(int(chip.shape[-2]), int(chip.shape[-1])) for chip in chips]

    max_h = round_up(max(h for h, _ in shapes), pad_divisor)
    max_w = round_up(max(w for _, w in shapes), pad_divisor)

    padded_chips = []
    for chip in chips:
        h, w = chip.shape[-2], chip.shape[-1]
        padded_chips.append(F.pad(chip, (0, max_w - w, 0, max_h - h)))

    return {
        "chip_id": [item["chip_id"] for item in batch],
        "chip": torch.stack(padded_chips, dim=0),
        "shape": shapes,
    }


def build_prediction_dataloader(
    model: CloudModel,
    x_paths: pd.DataFrame,
    bands: List[str],
    device_type: str,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader for prediction.

    Args:
        model: CloudModel with batch_size and num_workers attributes
        x_paths: DataFrame with chip paths
        bands: List of band names
        device_type: Device type ('cuda', 'mps', or 'cpu')

    Returns:
        Configured DataLoader
    """
    dataset = CloudDataset(x_paths=x_paths.reset_index(drop=True), bands=bands)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=model.batch_size,
        num_workers=model.num_workers,
        shuffle=False,
        pin_memory=device_type == "cuda",
        collate_fn=pad_prediction_batch,
    )


def iter_chip_probability_batches(
    model: CloudModel,
    x_paths: pd.DataFrame,
    bands: List[str] = BANDS,
    tta_modes: Optional[Sequence[str]] = DEFAULT_TTA_MODES,
):
    """
    Iterate over chip batches and yield probability predictions.

    Args:
        model: CloudModel for prediction
        x_paths: DataFrame with chip metadata
        bands: List of band names
        tta_modes: TTA modes to use

    Yields:
        Tuples of (chip_ids, probabilities, shapes) where:
        - chip_ids: List of chip IDs
        - probabilities: NumPy array of shape (B, num_classes, H, W)
        - shapes: List of (H, W) tuples for original chip sizes
    """
    device_type = getattr(model, "device_type", "cpu")

    # Move model to device
    if device_type in ("cuda", "mps"):
        model = model.to(device_type)

    dataloader = build_prediction_dataloader(model, x_paths, bands, device_type)

    with torch.no_grad():
        for batch in dataloader:
            x = batch["chip"]
            if device_type in ("cuda", "mps"):
                x = x.to(device_type)

            probs = predict_with_tta(model, x, tta_modes=tta_modes)

            yield batch["chip_id"], probs.detach().cpu().numpy(), batch["shape"]


def save_prediction_geotiff(
    pred: np.ndarray,
    reference_path: Path,
    output_path: Path,
) -> None:
    """
    Save a prediction as a GeoTIFF with geographic metadata.

    Args:
        pred: Prediction array (H, W) with class labels
        reference_path: Path to reference image for metadata
        output_path: Output path for prediction GeoTIFF
    """
    import rasterio

    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()

    profile.update(
        count=1,
        dtype=OUTPUT_DTYPE,
        height=pred.shape[0],
        width=pred.shape[1],
        compress=DEFAULT_COMPRESSION,
        predictor=DEFAULT_PREDICTOR,
    )
    profile.pop("nodata", None)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred.astype(np.uint8), 1)


def crop_prediction_to_shape(
    pred: np.ndarray,
    prob: np.ndarray,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop prediction and probability to original chip shape.

    Args:
        pred: Prediction array (H, W)
        prob: Probability array (num_classes, H, W)
        shape: Original (height, width) of the chip

    Returns:
        Tuple of (cropped_pred, cropped_prob)
    """
    height, width = shape
    return pred[:height, :width], prob[:, :height, :width]


def save_chip_predictions(
    model: CloudModel,
    x_paths: pd.DataFrame,
    pred_out_dir: Path,
    bands: List[str] = BANDS,
    tta_modes: Optional[Sequence[str]] = DEFAULT_TTA_MODES,
    save_probabilities: bool = False,
) -> int:
    """
    Save predictions for all chips in x_paths.

    Args:
        model: CloudModel for prediction
        x_paths: DataFrame with chip metadata
        pred_out_dir: Output directory for predictions
        bands: List of band names
        tta_modes: TTA modes to use
        save_probabilities: Whether to save probability arrays as .npy files

    Returns:
        Number of predictions saved
    """
    ensure_dir(pred_out_dir)

    prob_out_dir = pred_out_dir / "probabilities"
    if save_probabilities:
        ensure_dir(prob_out_dir)

    metadata_by_chip_id = x_paths.set_index("chip_id")
    count = 0

    for chip_ids, probs, shapes in iter_chip_probability_batches(
        model, x_paths, bands=bands, tta_modes=tta_modes
    ):
        preds = np.argmax(probs, axis=1).astype(np.uint8)

        for chip_id, pred, prob, shape in zip(chip_ids, preds, probs, shapes):
            pred, prob = crop_prediction_to_shape(pred, prob, shape)
            row = metadata_by_chip_id.loc[chip_id]

            # Save prediction GeoTIFF
            ref_path = row[f"{bands[0]}_path"]
            save_prediction_geotiff(pred, ref_path, pred_out_dir / f"{chip_id}.tif")

            # Save probability if requested
            if save_probabilities:
                np.save(prob_out_dir / f"{chip_id}.npy", prob.astype(PROBABILITY_DTYPE))

            count += 1

    logger.info(f"Saved {count} predictions to {pred_out_dir}")
    return count


def predict_small_chips(
    features: Path | pd.DataFrame,
    pred_out_dir: Path,
    model_weights_path: Path,
    bands: List[str] = BANDS,
    model_name: str = MODEL_NAME,
    fast_dev_run: bool = False,
    tta_modes: Optional[Sequence[str]] = DEFAULT_TTA_MODES,
    save_probabilities: bool = False,
) -> pd.DataFrame:
    """
    Predict on small pre-chipped images.

    Args:
        features: Path to features directory or CSV, or DataFrame
        pred_out_dir: Output directory for predictions
        model_weights_path: Path to model weights
        bands: List of band names
        model_name: Model architecture name
        fast_dev_run: Whether to use fast development mode
        tta_modes: TTA modes to use
        save_probabilities: Whether to save probability arrays

    Returns:
        DataFrame with chip metadata
    """
    from benchmark.metadata_io import load_chip_metadata

    # Load metadata
    x_paths = load_chip_metadata(features, bands=bands)

    # Load model
    model = load_cloud_model(model_weights_path, bands=bands, model_name=model_name)

    # Fast dev mode
    x_paths = maybe_fast_dev(x_paths, model, fast_dev_run)

    logger.info(f"Found {len(x_paths)} small chips")
    logger.info("Generating small-chip predictions in batches")

    # Run prediction
    save_chip_predictions(
        model=model,
        x_paths=x_paths,
        pred_out_dir=Path(pred_out_dir),
        bands=bands,
        tta_modes=tta_modes,
        save_probabilities=save_probabilities,
    )

    return x_paths


def fuse_probability_scores(
    prob_full: np.ndarray,
    weight_full: np.ndarray,
    fusion_method: str = "probability",
) -> np.ndarray:
    """
    Fuse overlapping probability predictions.

    Args:
        prob_full: Accumulated probability scores (num_classes, H, W)
        weight_full: Accumulated weights for normalization (H, W)
        fusion_method: Fusion method name (only "probability" supported)

    Returns:
        Fused prediction labels (H, W)

    Raises:
        ValueError: If fusion_method is not supported
    """
    if fusion_method != "probability":
        raise ValueError(f"fusion_method must be 'probability', got {fusion_method}")

    # Avoid division by zero
    weight_full = np.maximum(weight_full, 1)

    # Normalize probabilities and get argmax
    normalized_probs = prob_full / weight_full[None, :, :]
    return np.argmax(normalized_probs, axis=0).astype(np.uint8)
