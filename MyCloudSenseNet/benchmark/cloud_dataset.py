from typing import Optional, List

import albumentations as A
import numpy as np
import pandas as pd
import rasterio
import torch


def normalize_by_minmax(data, max_pixel=1):
    min_val = float(np.nanmin(data))
    max_val = float(np.nanmax(data))
    range_val = max_val - min_val
    if range_val == 0:
        return data.copy()
    data = (data.copy() - min_val) / range_val
    data *= max_pixel
    return data

def normalize_by_bit_depth(data, input_bit_depth: int = 10, target_range: tuple = (0, 1)):
    """
    基于固定位深度的归一化
    保留绝对辐射强度信息，适用于跨位深度迁移

    Args:
        data: 输入数据
        input_bit_depth: 输入数据的位深度 (GF1=10)
        target_range: 归一化目标范围，默认(0, 1)

    Returns:
        归一化后的数据
    """
    max_val = 2 ** input_bit_depth - 1  # 10-bit: 1023
    normalized = data.astype("float32") / max_val

    min_val, max_val = target_range
    normalized = normalized * (max_val - min_val) + min_val

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
        bit_depth: Optional[int] = 10,
        transforms: Optional[A.Compose] = None,
    ):
        """
        Instantiate the CloudDataset class.
        """
        self.data = x_paths
        self.bands = bands
        self.label = y_paths
        self.bit_depth = bit_depth
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # Loads an n-channel image from a chip-level dataframe
        img = self.data.loc[idx]
        band_arrs = []
        for band in self.bands:
            with rasterio.open(img[f"{band}_path"]) as b:
                band_arr = b.read(1).astype("float32")
                # band_arr = normalize_by_bit_depth(band_arr, self.bit_depth)
                band_arr = normalize_by_minmax(band_arr)
            band_arrs.append(band_arr)
        x_arr = np.stack(band_arrs, axis=-1)

        # Apply data augmentations, if provided
        if self.transforms:
            x_arr = self.transforms(image=x_arr)["image"]
        x_arr = np.transpose(x_arr, [2, 0, 1])

        # Prepare dictionary for item
        item = {"chip_id": str(img.chip_id), "chip": x_arr}

        # Load label if available
        if self.label is not None:
            label_path = self.label.loc[idx].label_path
            with rasterio.open(label_path) as lp:
                y_arr = lp.read(1).astype("float32")
            # Apply same data augmentations to the label
            if self.transforms:
                y_arr = self.transforms(image=y_arr)["image"]
            item["label"] = y_arr

        return item
