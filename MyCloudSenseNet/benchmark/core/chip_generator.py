"""Chip-window generation and chip extraction."""

from typing import List, Optional, Tuple

import numpy as np
import rasterio

from ..configs.defaults import CHIP_SIZE, OVERLAP_RATIO, logger
from .tiling_types import ChipWindow, ImageInfo


class ChipGenerator:
    """Generate chip windows and extract image/label arrays."""

    def __init__(
        self,
        image_info: ImageInfo,
        base_name: str,
    ):
        self.info = image_info
        self.base_name = base_name

    def generate_windows(
        self,
        chip_size: int = CHIP_SIZE,
        overlap_ratio: float = OVERLAP_RATIO,
    ) -> List[ChipWindow]:
        stride = int(chip_size * (1 - overlap_ratio))
        height, width = self.info.height, self.info.width
        windows = []
        chip_idx = 0

        y_start = 0
        while y_start < height:
            y_end = min(y_start + chip_size, height)
            x_start = 0

            while x_start < width:
                x_end = min(x_start + chip_size, width)
                windows.append(
                    ChipWindow(
                        chip_idx=chip_idx,
                        chip_id=f"{self.base_name}_chip_{chip_idx:04d}",
                        chip_size=chip_size,
                        x_start=x_start,
                        x_end=x_end,
                        y_start=y_start,
                        y_end=y_end,
                        transform=self._calculate_transform(
                            x_start,
                            y_start,
                            x_end,
                            y_end,
                            chip_size,
                        ),
                    )
                )
                x_start += stride
                chip_idx += 1

            y_start += stride

        self.info.windows = windows
        logger.info(
            f"Generated {len(windows)} chips: "
            f"size={chip_size}, overlap={overlap_ratio * 100:.0f}%, stride={stride}"
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
        orig_transform = self.info.transform
        x_min = orig_transform.xoff + x_start * orig_transform.a
        y_max = orig_transform.yoff + y_start * orig_transform.e
        x_max = orig_transform.xoff + x_end * orig_transform.a
        y_min = orig_transform.yoff + y_end * orig_transform.e

        return rasterio.transform.from_bounds(
            x_min,
            y_min,
            x_max,
            y_max,
            chip_size,
            chip_size,
        )

    def extract_chip(
        self,
        window: ChipWindow,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        x1, x2 = window.x_start, window.x_end
        y1, y2 = window.y_start, window.y_end
        chip_label = self.info.label[y1:y2, x1:x2]

        chip_data = self.info.data[:, y1:y2, x1:x2]
        return chip_data, chip_label
