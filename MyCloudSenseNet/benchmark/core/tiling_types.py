"""Shared dataset structures for GeoTIFF tiling."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import rasterio


@dataclass
class ChipWindow:
    """A single chip window within a larger image."""

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
        return self.x_end - self.x_start

    @property
    def height(self) -> int:
        return self.y_end - self.y_start

    def to_dict(self) -> Dict[str, Any]:
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
    """Image pixels, labels and geospatial metadata used across the tiling flow."""

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
