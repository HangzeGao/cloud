"""GeoTIFF prediction orchestration.

This module composes the low-level tiler from ``benchmark.core`` with the
batch prediction helpers from ``benchmark.inference.predict``.
"""

from __future__ import annotations

from typing import Optional

import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..configs import ModelConfig
from ..configs.defaults import DEFAULT_MODEL_WEIGHTS_PATH, NUM_CLASSES, logger
from ..core.geotiff_tiler import GeoTIFFTiler
from ..utils.utils import ensure_dir
from .predict import crop, fuse, iter_predict, load_model, run, save_geotiff


def predict_full_geotiff(
    tiler: GeoTIFFTiler,
    pred_out_dir: Path,
    model_weights_path: Path = DEFAULT_MODEL_WEIGHTS_PATH,
    config: Optional[ModelConfig] = None,
    device: str = "auto",
    save_chip_masks: bool = False,
    save_probabilities: bool = False,
) -> Path:
    """Predict chips from a prepared tiler and fuse them into one full mask."""
    if tiler.info.metadata is None:
        raise ValueError(
            "No chip metadata available. Call get_chips() first or load existing metadata."
        )

    logger.info("Loading model")
    model = load_model(
        model_weights_path=model_weights_path,
        config=config,
        device=device,
    )
    bands = model.config.bands
    x_paths = tiler.info.metadata
    logger.info(f"Predicting on {len(x_paths)} chips")

    pred_label_dir = pred_out_dir / tiler.df_row.filename
    ensure_dir(pred_label_dir)

    prob_out_dir = pred_label_dir / "probabilities"
    if save_probabilities:
        ensure_dir(prob_out_dir)

    height, width = tiler.info.height, tiler.info.width
    num_classes = getattr(model, "num_classes", NUM_CLASSES)
    prob_full = np.zeros((num_classes, height, width), dtype=np.float32)
    weight_full = np.zeros((height, width), dtype=np.float32)
    metadata_by_chip_id = x_paths.set_index("chip_id")

    logger.info("Predicting and fusing chips")
    total_batches = math.ceil(len(x_paths) / max(model.config.batch_size, 1))
    for chip_ids, preds, probs, shapes in tqdm(
        iter_predict(model, x_paths),
        total=total_batches,
        desc="Predicting",
    ):
        for chip_id, pred, prob, shape in zip(chip_ids, preds, probs, shapes):
            row = metadata_by_chip_id.loc[chip_id]
            x1, y1 = int(row["x_start"]), int(row["y_start"])
            x2, y2 = int(row["x_end"]), int(row["y_end"])
            chip_h, chip_w = y2 - y1, x2 - x1

            pred, prob = crop(pred, prob, shape)
            prob_full[:, y1:y2, x1:x2] += prob[:, :chip_h, :chip_w]
            weight_full[y1:y2, x1:x2] += 1

            if save_chip_masks:
                save_geotiff(
                    pred,
                    row[f"{bands[0]}_path"],
                    pred_label_dir / f"{chip_id}.tif",
                )

            if save_probabilities:
                np.save(prob_out_dir / f"{chip_id}.npy", prob.astype("float16"))

    pred_full = fuse(prob_full, weight_full)
    output_path = pred_out_dir / f"{tiler.df_row.filename}_PredictedMask.tif"
    tiler.save_full_prediction(pred_full, output_path)

    logger.info(f"Saved prediction to {output_path}")
    return output_path


def predict_geotiff_chips(
    tiler: GeoTIFFTiler,
    out_dir: Path,
    model_weights_path: Path = DEFAULT_MODEL_WEIGHTS_PATH,
    config: Optional[ModelConfig] = None,
    device: str = "auto",
    save_probs: bool = False,
) -> int:
    """Predict each chip from a prepared tiler and save chip masks."""
    if tiler.info.metadata is None:
        raise ValueError(
            "No chip metadata available. Call get_chips() first or load existing metadata."
        )

    logger.info("Loading model")
    model = load_model(
        model_weights_path=model_weights_path,
        config=config,
        device=device,
    )
    x_paths = tiler.info.metadata
    logger.info(f"Found {len(x_paths)} chips")

    return run(
        model=model,
        x_paths=x_paths,
        output_dir=out_dir,
        save_probs=save_probs,
    )
