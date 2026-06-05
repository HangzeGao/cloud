"""GeoTIFF tiling orchestration."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from .chip_generator import ChipGenerator
from .chip_writer import ChipWriter
from ..configs.defaults import (
    CHIP_SIZE,
    DEFAULT_COMPRESSION,
    DEFAULT_MAX_WORKERS,
    DEFAULT_PREDICTOR,
    OUTPUT_DTYPE,
    OVERLAP_RATIO,
    logger,
)
from .geotiff_reader import GeoTIFFReader
from .raster_utils import label_class_stats
from .tiling_types import ChipWindow


class GeoTIFFTiler:
    """Coordinate reading, chip generation and chip writing."""

    def __init__(
        self,
        df_row: pd.Series,
        display_thumbnail: bool = False,
    ):
        self.df_row = df_row
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
        windows = self.generator.generate_windows(chip_size, overlap_ratio)

        csv_out_dir.mkdir(exist_ok=True, parents=True)
        output_dirs = {
            "img": img_out_dir / self.df_row.filename,
            "label": label_out_dir / self.df_row.filename,
        }

        metadata = []
        process_func = partial(self._process_single_chip, output_dirs=output_dirs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_func, window): window for window in windows}
            for future in tqdm(futures.keys(), total=len(windows), desc="Processing chips"):
                result = future.result()
                if result:
                    metadata.append(result)

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
        result = self.generator.extract_chip(window)
        if result is None:
            return None

        chip_data, chip_label = result
        paths = self.writer.write_chip(
            chip_data,
            chip_label,
            window,
            output_dirs,
            self.info.crs,
        )

        row = {
            "chip_id": window.chip_id,
            "x_start": window.x_start,
            "x_end": window.x_end,
            "y_start": window.y_start,
            "y_end": window.y_end,
            **label_class_stats(chip_label),
            **paths,
        }

        if hasattr(self.df_row, "location"):
            row["location"] = self.df_row.location
        if hasattr(self.df_row, "datetime"):
            row["datetime"] = self.df_row.datetime

        return row

    def save_full_prediction(self, pred: np.ndarray, output_path: Path) -> None:
        """Save a full-scene prediction using the source GeoTIFF profile."""
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
