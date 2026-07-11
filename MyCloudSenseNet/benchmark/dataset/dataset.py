from typing import Optional, List
import numpy as np
import pandas as pd
import rasterio
import torch


def normalize_by_minmax(data, max_pixel=1):
    min_val = float(np.nanmin(data))
    max_val = float(np.nanmax(data))
    range_val = max_val - min_val
    if range_val == 0:
        return data
    data = (data - min_val) / range_val
    data *= max_pixel
    return data


def estimate_bit_depth_from_range(
    data,
    min_bit_depth: int = 1,
    max_bit_depth: Optional[int] = None,
) -> int:
    """Estimate bit depth from finite input values."""
    data = np.asarray(data)
    finite_data = data[np.isfinite(data)]
    if finite_data.size == 0:
        return min_bit_depth

    data_min = float(finite_data.min())
    data_max = float(finite_data.max())
    if data_max <= 0:
        return min_bit_depth

    if data_min < 0:
        data_max = data_max - data_min

    bit_depth = int(np.ceil(np.log2(data_max + 1)))
    bit_depth = max(bit_depth, min_bit_depth)
    if max_bit_depth is not None:
        bit_depth = min(bit_depth, max_bit_depth)
    return bit_depth


def normalize_by_bit_depth(
    data,
    input_bit_depth: Optional[int] = None,
    target_range: tuple = (0, 1),
    clip: bool = True,
):
    """
    基于位深度的归一化。
    当 input_bit_depth 为 None 时，根据输入数据范围自动估计位深度。

    Args:
        data: 输入数据
        input_bit_depth: 输入数据的位深度；为 None 时自动估计
        target_range: 归一化目标范围，默认(0, 1)
        clip: 是否将结果裁剪到 target_range

    Returns:
        归一化后的数据
    """
    if input_bit_depth is None:
        input_bit_depth = estimate_bit_depth_from_range(data)

    source_max = 2 ** input_bit_depth - 1
    normalized = data.astype("float32") / source_max

    min_val, max_val = target_range
    normalized = normalized * (max_val - min_val) + min_val
    if clip:
        normalized = np.clip(normalized, min_val, max_val)

    return normalized

def percentile_stretch_normalize(data, low_percentile=2, high_percentile=98, target_range=(0, 1)):
    """
    基于百分位数的拉伸归一化
    对异常值鲁棒，适用于不同位深度图像

    Args:
        data: 输入数据
        low_percentile: 低百分位数 (默认2%)
        high_percentile: 高百分位数 (默认98%)
        target_range: 归一化目标范围
    """
    data = data.astype("float32")

    lo = np.percentile(data, low_percentile)
    hi = np.percentile(data, high_percentile)

    stretched = (data - lo) / (hi - lo + 1e-8)
    stretched = np.clip(stretched, 0, 1)

    min_val, max_val = target_range
    return stretched * (max_val - min_val) + min_val


class CloudDataset(torch.utils.data.Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(
        self,
        x_paths: pd.DataFrame,
        bands: List[str],
        y_paths: Optional[pd.DataFrame] = None,
        bit_depth: Optional[int] = None,
        bit_depth_classes: Optional[List[int]] = None,
        transforms: Optional = None,
    ):
        """
        Instantiate the CloudDataset class.
        """
        self.data = x_paths
        self.bands = bands
        self.label = y_paths
        self.bit_depth = bit_depth
        self.bit_depth_classes = tuple(bit_depth_classes or [8, 10, 12, 14, 16])
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Loads an n-channel image from a chip-level dataframe
        img = self.data.loc[idx]
        band_arrs = []
        band_bit_depths = []
        for band in self.bands:
            with rasterio.open(img[f"{band}_path"]) as b:
                band_arr = b.read(1).astype("float32")
                band_bit_depth = self._get_band_bit_depth(img, band, band_arr)
                band_bit_depths.append(band_bit_depth)
                stats = self._get_percentile_stats(img, band)
                if stats is None:
                    band_arr = normalize_by_bit_depth(band_arr, band_bit_depth)
                else:
                    lo, hi = stats
                    band_arr = np.clip((band_arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            band_arrs.append(band_arr)
        x_arr = np.stack(band_arrs, axis=-1)

        y_arr = None
        if self.label is not None:
            label_path = self.label.loc[idx].label_path
            with rasterio.open(label_path) as lp:
                y_arr = lp.read(1).astype("float32")

        if self.transforms:
            if y_arr is None:
                x_arr = self.transforms(image=x_arr)["image"]
            else:
                transformed = self.transforms(image=x_arr, mask=y_arr)
                x_arr = transformed["image"]
                y_arr = transformed["mask"]

        if isinstance(x_arr, np.ndarray):
            x_arr = np.transpose(x_arr, [2, 0, 1])

        sample_bit_depth = self._nearest_supported_bit_depth(max(band_bit_depths))
        item = {
            "chip_id": str(img.chip_id),
            "chip": x_arr,
            "bit_depth": torch.tensor(sample_bit_depth, dtype=torch.long),
        }
        if y_arr is not None:
            item["label"] = y_arr

        return item

    def _get_band_bit_depth(self, row, band: str, band_arr: np.ndarray) -> int:
        if self.bit_depth is not None:
            return int(self.bit_depth)

        for column in ("bit_depth", "input_bit_depth", f"{band}_bit_depth", f"{band}_depth"):
            if column in row and pd.notna(row[column]):
                return int(row[column])

        return estimate_bit_depth_from_range(band_arr)

    def _get_percentile_stats(self, row, band: str):
        p2_column = f"{band}_p2"
        p98_column = f"{band}_p98"
        if p2_column not in row or p98_column not in row:
            return None
        if pd.isna(row[p2_column]) or pd.isna(row[p98_column]):
            return None
        return float(row[p2_column]), float(row[p98_column])

    def _nearest_supported_bit_depth(self, bit_depth: int) -> int:
        return min(self.bit_depth_classes, key=lambda candidate: abs(candidate - int(bit_depth)))
