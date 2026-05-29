"""
GeoTIFF tiling module for cloud detection.

This module provides functionality for tiling large GeoTIFF images into smaller chips
for processing, and then reassembling the results.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from benchmark.core.config import (
    BANDS,
    CHIP_SIZE,
    OVERLAP_RATIO,
    DEFAULT_MAX_WORKERS,
    DEFAULT_COMPRESSION,
    DEFAULT_PREDICTOR,
    OUTPUT_DTYPE,
    NUM_CLASSES,
    logger,
    DEFAULT_MODEL_WEIGHTS_PATH,
)
from benchmark.utils.utils import (
    ensure_dir,
    mask2label,
    is_label_valid,
    create_rgb_composite,
    display_thumbnail,
    calculate_bit_depth,
)
from benchmark.core.prediction import (
    load_model,
    save_geotiff,
    iter_predict,
    crop,
    fuse,
    run,
)


@dataclass
class ChipWindow:
    """Represents a single chip window within a larger image."""

    chip_idx: int
    chip_id: str
    chip_size: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    transform: rasterio.Affine

    @property
    def width(self) -> int:
        """Get the actual width of the chip window."""
        return self.x_end - self.x_start

    @property
    def height(self) -> int:
        """Get the actual height of the chip window."""
        return self.y_end - self.y_start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chip_idx": self.chip_idx,
            "chip_id": self.chip_id,
            "chip_size": self.chip_size,
            "x_start": self.x_start,
            "x_end": self.x_end,
            "y_start": self.y_start,
            "y_end": self.y_end,
            "transform": self.transform,
        }


@dataclass
class ImageInfo:
    """Container for image metadata and data."""

    data: np.ndarray
    label: np.ndarray
    bit_depth: int
    meta: Dict[str, Any]
    transform: rasterio.Affine
    crs: Any
    height: int
    width: int
    windows: Optional[List[ChipWindow]] = None
    metadata: Optional[pd.DataFrame] = None


class GeoTIFFReader:
    """Handles reading and basic processing of GeoTIFF files."""

    def __init__(self, file_path: Path):
        """
        Initialize reader with a GeoTIFF file path.

        Args:
            file_path: Path to the GeoTIFF file
        """
        self.file_path = Path(file_path)
        self.info: Optional[ImageInfo] = None

    def read(
        self,
        display_thumbnail: bool = False,
    ) -> ImageInfo:
        """
        Read the GeoTIFF file and extract image info.

        Args:
            display_thumbnail: Whether to display a thumbnail preview

        Returns:
            ImageInfo with all metadata and image data

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with rasterio.open(self.file_path) as src:
            # Read first 4 bands (B02, B03, B04, B08)
            data = src.read((1, 2, 3, 4))
            # Read 5th band as mask/label
            mask = src.read(5)

            bit_depth = calculate_bit_depth(data)
            label = mask2label(mask)

            self.info = ImageInfo(
                data=data,
                label=label,
                bit_depth=bit_depth,
                meta=src.meta.copy(),
                transform=src.transform,
                crs=src.crs,
                height=src.height,
                width=src.width,
            )

            if display_thumbnail:
                self._display_thumbnail(src)

        return self.info

    def _display_thumbnail(self, src: rasterio.DatasetReader) -> None:
        """Display a thumbnail preview of the image."""
        from matplotlib import pyplot as plt

        thumbnail = display_thumbnail(src)

        # Extract bands for RGB
        b, g, r, nir = thumbnail[:4]
        mask = thumbnail[4] if thumbnail.shape[0] > 4 else None

        rgb = create_rgb_composite(np.stack([b, g, r]))

        fig, axes = plt.subplots(1, 2 if mask is not None else 1, figsize=(12, 6))
        if mask is None:
            axes = [axes]

        name = self.file_path.stem
        fig.suptitle(name, fontsize=14, fontweight="bold")

        axes[0].imshow(rgb)
        axes[0].set_title("RGB Image")
        axes[0].axis("off")

        if mask is not None:
            axes[1].imshow(mask)
            axes[1].set_title("Mask Image")
            axes[1].axis("off")

        plt.tight_layout()
        plt.show()


class ChipGenerator:
    """Generates chip windows and extracts chips from large images."""

    def __init__(
        self,
        image_info: ImageInfo,
        base_name: str,
    ):
        """
        Initialize chip generator.

        Args:
            image_info: ImageInfo with image data and metadata
            base_name: Base name for generated chip IDs
        """
        self.info = image_info
        self.base_name = base_name

    def generate_windows(
        self,
        chip_size: int = CHIP_SIZE,
        overlap_ratio: float = OVERLAP_RATIO,
    ) -> List[ChipWindow]:
        """
        Generate overlapping chip windows for the image.

        Args:
            chip_size: Size of each chip in pixels
            overlap_ratio: Overlap ratio between adjacent chips (0-1)

        Returns:
            List of ChipWindow objects
        """
        stride = int(chip_size * (1 - overlap_ratio))
        height, width = self.info.height, self.info.width

        windows = []
        chip_idx = 0

        # Row-wise iteration
        y_start = 0
        while y_start < height:
            y_end = min(y_start + chip_size, height)

            # Column-wise iteration
            x_start = 0
            while x_start < width:
                x_end = min(x_start + chip_size, width)

                chip_id = f"{self.base_name}_chip_{chip_idx:04d}"
                transform = self._calculate_transform(
                    x_start, y_start, x_end, y_end, chip_size
                )

                windows.append(
                    ChipWindow(
                        chip_idx=chip_idx,
                        chip_id=chip_id,
                        chip_size=chip_size,
                        x_start=x_start,
                        x_end=x_end,
                        y_start=y_start,
                        y_end=y_end,
                        transform=transform,
                    )
                )

                x_start += stride
                chip_idx += 1
            y_start += stride

        self.info.windows = windows

        logger.info(
            f"Generated {len(windows)} chips: "
            f"size={chip_size}, overlap={overlap_ratio*100:.0f}%, stride={stride}"
        )

        return windows

    def _calculate_transform(
        self,
        x_start: int,
        y_start: int,
        x_end: int,
        y_end: int,
        chip_size: int,
    ) -> rasterio.Affine:
        """
        Calculate the geospatial transform for a chip window.

        Args:
            x_start: Starting X coordinate
            y_start: Starting Y coordinate
            x_end: Ending X coordinate
            y_end: Ending Y coordinate
            chip_size: Size of the chip

        Returns:
            Affine transform for the chip
        """
        orig_transform = self.info.transform
        x_min = orig_transform.xoff + x_start * orig_transform.a
        y_max = orig_transform.yoff + y_start * orig_transform.e
        x_max = orig_transform.xoff + x_end * orig_transform.a
        y_min = orig_transform.yoff + y_end * orig_transform.e

        return rasterio.transform.from_bounds(
            x_min, y_min, x_max, y_max, chip_size, chip_size
        )

    def extract_chip(
        self,
        window: ChipWindow,
        is_for_training: bool = True,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract image data and label for a single chip window.

        Args:
            window: ChipWindow defining the region to extract
            is_for_training: Whether this is for training (applies label validation)

        Returns:
            Tuple of (chip_data, chip_label) or None if invalid
        """
        x1, x2 = window.x_start, window.x_end
        y1, y2 = window.y_start, window.y_end

        chip_label = self.info.label[y1:y2, x1:x2]

        # Validate label for training
        if is_for_training and not is_label_valid(chip_label):
            return None

        chip_data = self.info.data[:, y1:y2, x1:x2]
        return chip_data, chip_label


class ChipWriter:
    """Writes chips to disk as individual GeoTIFF files."""

    def __init__(
        self,
        bands: List[str] = BANDS,
        compression: str = DEFAULT_COMPRESSION,
        predictor: int = DEFAULT_PREDICTOR,
    ):
        """
        Initialize chip writer.

        Args:
            bands: List of band names
            compression: Compression algorithm for output files
            predictor: Predictor type for compression
        """
        self.bands = bands
        self.compression = compression
        self.predictor = predictor

    def write_chip(
        self,
        chip_data: np.ndarray,
        chip_label: np.ndarray,
        window: ChipWindow,
        output_dirs: Dict[str, Path],
        crs: Any,
    ) -> Dict[str, str]:
        """
        Write a single chip to disk.

        Args:
            chip_data: Image data array (bands, H, W)
            chip_label: Label array (H, W)
            window: ChipWindow with metadata
            output_dirs: Dictionary with 'img' and 'label' output paths
            crs: Coordinate reference system

        Returns:
            Dictionary with paths to written files
        """
        chip_id = window.chip_id
        chip_size = window.chip_size

        # Create output directories
        img_out_dir = output_dirs["img"]
        label_out_dir = output_dirs["label"]
        ensure_dir(img_out_dir)
        ensure_dir(label_out_dir)

        chip_dir = img_out_dir / chip_id
        ensure_dir(chip_dir)

        # GeoTIFF metadata
        meta = {
            "driver": "GTiff",
            "height": chip_size,
            "width": chip_size,
            "count": 1,
            "dtype": "uint16",
            "crs": crs,
            "transform": window.transform,
            "compress": self.compression,
            "predictor": self.predictor,
        }

        paths = {}

        # Write each band
        for idx, band in enumerate(self.bands):
            if idx < chip_data.shape[0]:
                path = chip_dir / f"{band}.tif"
                if not path.exists():
                    with rasterio.open(path, "w", **meta) as dst:
                        dst.write(chip_data[idx], 1)
                paths[f"{band}_path"] = str(path.resolve())

        # Write label
        label_path = label_out_dir / f"{chip_id}.tif"
        if not label_path.exists():
            with rasterio.open(label_path, "w", **meta) as dst:
                dst.write(chip_label, 1)
        paths["label_path"] = str(label_path.resolve())

        return paths


class GeoTIFFTiler:
    """
    Main class for tiling large GeoTIFF images.

    Coordinates reading, window generation, extraction, and writing of chips.
    """

    def __init__(
        self,
        df_row: pd.Series,
        is_for_training: bool = True,
        display_thumbnail: bool = False,
    ):
        """
        Initialize tiler with image metadata.

        Args:
            df_row: Pandas Series with 'path', 'filename', and optionally 'location', 'datetime'
            is_for_training: Whether this is for training (applies label validation)
            display_thumbnail: Whether to display a thumbnail preview
        """
        self.df_row = df_row
        self.is_for_training = is_for_training

        # Initialize components
        self.reader = GeoTIFFReader(df_row.path)
        self.info = self.reader.read(display_thumbnail=display_thumbnail)

        self.generator = ChipGenerator(self.info, df_row.filename)
        self.writer = ChipWriter()

    def get_chips(
        self,
        csv_out_dir: Path,
        img_out_dir: Path,
        label_out_dir: Path,
        max_workers: int = DEFAULT_MAX_WORKERS,
        chip_size: int = CHIP_SIZE,
        overlap_ratio: float = OVERLAP_RATIO,
    ) -> pd.DataFrame:
        """
        Extract and save all chips from the image.

        Args:
            csv_out_dir: Output directory for metadata CSV
            img_out_dir: Output directory for image chips
            label_out_dir: Output directory for label chips
            max_workers: Number of parallel workers
            chip_size: Size of each chip
            overlap_ratio: Overlap ratio between chips

        Returns:
            DataFrame with chip metadata
        """
        # Generate windows
        windows = self.generator.generate_windows(chip_size, overlap_ratio)

        # Prepare output directories
        ensure_dir(csv_out_dir)
        img_chip_out_dir = img_out_dir / self.df_row.filename
        label_chip_out_dir = label_out_dir / self.df_row.filename
        output_dirs = {"img": img_chip_out_dir, "label": label_chip_out_dir}

        # Process chips in parallel
        metadata = []
        process_func = partial(
            self._process_single_chip,
            output_dirs=output_dirs,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_func, window): window for window in windows}

            for future in tqdm(
                futures.keys(),
                total=len(windows),
                desc="Processing chips",
            ):
                result = future.result()
                if result:
                    metadata.append(result)

        # Save metadata
        chip_df = pd.DataFrame(metadata)
        self.info.metadata = chip_df
        chip_df.to_csv(csv_out_dir / f"{self.df_row.filename}_metadata.csv", index=False)

        logger.info(
            f"Completed {self.df_row.filename}: "
            f"{len(metadata)}/{len(windows)} valid chips"
        )

        return chip_df

    def _process_single_chip(
        self,
        window: ChipWindow,
        output_dirs: Dict[str, Path],
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single chip window.

        Args:
            window: ChipWindow to process
            output_dirs: Output directory paths

        Returns:
            Dictionary with chip metadata or None if invalid
        """
        # Extract chip data
        result = self.generator.extract_chip(window, self.is_for_training)
        if result is None:
            return None

        chip_data, chip_label = result

        # Write chip to disk
        paths = self.writer.write_chip(
            chip_data,
            chip_label,
            window,
            output_dirs,
            self.info.crs,
        )

        # Build metadata row
        row = {
            "chip_id": window.chip_id,
            "x_start": window.x_start,
            "x_end": window.x_end,
            "y_start": window.y_start,
            "y_end": window.y_end,
            **paths,
        }

        # Add optional location/datetime if available
        if hasattr(self.df_row, "location"):
            row["location"] = self.df_row.location
        if hasattr(self.df_row, "datetime"):
            row["datetime"] = self.df_row.datetime

        return row

    def predict(
        self,
        pred_out_dir: Path,
        save_chip_masks: bool = False,
        save_probabilities: bool = False,
    ) -> Path:
        """
        Predict on all chips and fuse results into a full image.

        Args:
            pred_out_dir: Output directory for predictions
            save_chip_masks: Whether to save individual chip masks
            save_probabilities: Whether to save probability arrays

        Returns:
            Path to the output prediction GeoTIFF
        """
        if self.info.metadata is None:
            raise ValueError(
                "No chip metadata available. "
                "Call get_chips() first or load existing metadata."
            )

        logger.info("Loading model")
        model = load_model(DEFAULT_MODEL_WEIGHTS_PATH)

        # Prepare for prediction
        x_paths = self.info.metadata
        logger.info(f"Predicting on {len(x_paths)} chips")

        # Output directories
        pred_label_dir = pred_out_dir / self.df_row.filename
        ensure_dir(pred_label_dir)

        prob_out_dir = pred_label_dir / "probabilities"
        if save_probabilities:
            ensure_dir(prob_out_dir)

        # Initialize accumulation arrays
        height, width = self.info.height, self.info.width
        num_classes = getattr(model, "num_classes", NUM_CLASSES)
        prob_full = np.zeros((num_classes, height, width), dtype=np.float32)
        weight_full = np.zeros((height, width), dtype=np.float32)

        metadata_by_chip_id = x_paths.set_index("chip_id")

        # Predict and accumulate
        logger.info("Predicting and fusing chips")
        for chip_ids, preds, probs, shapes in tqdm(
            iter_predict(model, x_paths),
            total=math.ceil(len(x_paths) / max(model.config.batch_size, 1)),
            desc="Predicting",
        ):
            for chip_id, pred, prob, shape in zip(chip_ids, preds, probs, shapes):
                row = metadata_by_chip_id.loc[chip_id]
                x1, y1 = int(row["x_start"]), int(row["y_start"])
                x2, y2 = int(row["x_end"]), int(row["y_end"])
                chip_h, chip_w = y2 - y1, x2 - x1

                pred, prob = crop(pred, prob, shape)

                # Accumulate probabilities
                prob_full[:, y1:y2, x1:x2] += prob[:, :chip_h, :chip_w]
                weight_full[y1:y2, x1:x2] += 1

                # Save individual chip if requested
                if save_chip_masks:
                    save_geotiff(
                        pred,
                        row[f"{BANDS[0]}_path"],
                        pred_label_dir / f"{chip_id}.tif",
                    )

                if save_probabilities:
                    np.save(prob_out_dir / f"{chip_id}.npy", prob.astype("float16"))

        # Fuse and save
        pred_full = fuse(prob_full, weight_full)

        output_path = pred_out_dir / f"{self.df_row.filename}_PredictedMask.tif"
        self._save_full_prediction(pred_full, output_path)

        logger.info(f"Saved prediction to {output_path}")
        return output_path

    def _save_full_prediction(self, pred: np.ndarray, output_path: Path) -> None:
        """
        Save the full fused prediction as a GeoTIFF.

        Args:
            pred: Fused prediction array
            output_path: Output file path
        """
        meta = self.info.meta.copy()
        meta.update(
            {
                "count": 1,
                "dtype": OUTPUT_DTYPE,
                "height": self.info.height,
                "width": self.info.width,
                "crs": self.info.crs,
                "transform": self.info.transform,
                "compress": DEFAULT_COMPRESSION,
                "predictor": DEFAULT_PREDICTOR,
            }
        )
        meta.pop("nodata", None)

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(pred, 1)

    def predict_chips(
        self,
        out_dir: Path,
    ) -> int:
        """
        Predict on chips and save individual predictions (no fusion).

        Args:
            out_dir: Output directory for predictions

        Returns:
            Number of predictions saved
        """
        if self.info.metadata is None:
            raise ValueError(
                "No chip metadata available. "
                "Call get_chips() first or load existing metadata."
            )

        logger.info("Loading model")
        model = load_model(DEFAULT_MODEL_WEIGHTS_PATH)

        x_paths = self.info.metadata
        logger.info(f"Found {len(x_paths)} chips")

        return run(
            model=model,
            x_paths=x_paths,
            output_dir=out_dir,
        )
