"""Write extracted chips to GeoTIFF files."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rasterio

from ..configs.defaults import BANDS, DEFAULT_COMPRESSION, DEFAULT_PREDICTOR
from .tiling_types import ChipWindow


class ChipWriter:
    """Write image bands and labels for a chip window."""

    def __init__(
        self,
        bands: List[str] = BANDS,
        compression: str = DEFAULT_COMPRESSION,
        predictor: int = DEFAULT_PREDICTOR,
    ):
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
        chip_id = window.chip_id
        img_out_dir = output_dirs["img"]
        label_out_dir = output_dirs["label"]
        img_out_dir.mkdir(exist_ok=True, parents=True)
        label_out_dir.mkdir(exist_ok=True, parents=True)

        chip_dir = img_out_dir / chip_id
        chip_dir.mkdir(exist_ok=True, parents=True)

        meta = {
            "driver": "GTiff",
            "height": window.chip_size,
            "width": window.chip_size,
            "count": 1,
            "dtype": "uint16",
            "crs": crs,
            "transform": window.transform,
            "compress": self.compression,
            "predictor": self.predictor,
        }

        paths = {}
        for idx, band in enumerate(self.bands):
            if idx < chip_data.shape[0]:
                path = chip_dir / f"{band}.tif"
                if not path.exists():
                    with rasterio.open(path, "w", **meta) as dst:
                        dst.write(chip_data[idx], 1)
                paths[f"{band}_path"] = str(path.resolve())

        label_path = label_out_dir / f"{chip_id}.tif"
        label_meta = {**meta, "dtype": "uint8", "nodata": 255}
        with rasterio.open(label_path, "w", **label_meta) as dst:
            dst.write(chip_label.astype(np.uint8), 1)
        paths["label_path"] = str(label_path.resolve())

        return paths
