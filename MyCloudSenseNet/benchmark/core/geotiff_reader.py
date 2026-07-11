"""GeoTIFF reading and thumbnail display helpers."""

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

from .tiling_types import ImageInfo
from .raster_utils import (
    calculate_bit_depth,
    create_rgb_composite,
    display_thumbnail,
    mask2label,
)


class GeoTIFFReader:
    """Read source GeoTIFF files into an ImageInfo object."""

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.info: Optional[ImageInfo] = None

    def read(
        self,
        display_thumbnail_preview: bool = False,
        **kwargs,
    ) -> ImageInfo:
        display_thumbnail_preview = kwargs.get("display_thumbnail", display_thumbnail_preview)

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with rasterio.open(self.file_path) as src:
            # GF1-style source TIFFs contain B/G/R/NIR/label. Keep only B/G/R
            # for the common deployment-compatible model input.
            data = src.read((1, 2, 3))
            mask = src.read(5)
            normalization_stats = {}
            for index, band in enumerate(("B02", "B03", "B04")):
                values = data[index].ravel()
                step = max(1, values.size // 200_000)
                sample = values[::step]
                normalization_stats[band] = {
                    "p2": float(np.percentile(sample, 2)),
                    "p98": float(np.percentile(sample, 98)),
                }

            self.info = ImageInfo(
                data=data,
                label=mask2label(mask),
                bit_depth=calculate_bit_depth(data),
                meta=src.meta.copy(),
                transform=src.transform,
                crs=src.crs,
                height=src.height,
                width=src.width,
                normalization_stats=normalization_stats,
            )

            if display_thumbnail_preview:
                self._display_thumbnail(src)

        return self.info

    def _display_thumbnail(self, src: rasterio.DatasetReader) -> None:
        from matplotlib import pyplot as plt

        thumbnail = display_thumbnail(src)
        b, g, r = thumbnail[:3]
        mask = thumbnail[4] if thumbnail.shape[0] > 4 else None
        rgb = create_rgb_composite(np.stack([b, g, r]))

        fig, axes = plt.subplots(1, 2 if mask is not None else 1, figsize=(12, 6))
        if mask is None:
            axes = [axes]

        fig.suptitle(self.file_path.stem, fontsize=14, fontweight="bold")
        axes[0].imshow(rgb)
        axes[0].set_title("RGB Image")
        axes[0].axis("off")

        if mask is not None:
            axes[1].imshow(mask)
            axes[1].set_title("Mask Image")
            axes[1].axis("off")

        plt.tight_layout()
        plt.show()
