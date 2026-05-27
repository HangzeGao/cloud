import os
import re
from io import BytesIO

import cv2
import rasterio
import numpy as np
import pandas as pd
import tarfile
from pathlib import Path

from matplotlib import pyplot as plt
from tqdm import tqdm
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject, Resampling
from typing import List
from loguru import logger
from PIL import Image
import torch

try:
    from MyCloudSenseNet.benchmark.cloud_dataset import CloudDataset
    from MyCloudSenseNet.benchmark.cloud_model import CloudModel
except ImportError:
    from unet.benchmark.segformer.cloud_dataset import CloudDataset
    from unet.benchmark.segformer.cloud_model import CloudModel

import warnings
warnings.filterwarnings('ignore')

# DATASET_NAME = "GF1_WFV1_E88.6_N28.0_20141208_L2A0000845193"
# DATASET_NAME = "GF1_WFV1_E73.7_N56.3_20130712_L2A0000356200"
# DATASET_NAME = "GF1_WFV3_E103.3_N18.9_20140523_L2A0000356148"
# DATASET_NAME = "GF1_WFV3_E121.7_N27.2_20160801_L2A0001735665"
# DATASET_NAME = "GF1_WFV4_W152.9_N21.9_20160806_L2A0001748198"
# DATASET_NAME = "GF1_WFV4_E121.8_N18.5_20130508_L2A0000017773"
DATASET_NAME = "GF1_WFV4_W97.5_N38.5_20140517_L2A0000244685"

BANDS = ["B02", "B03", "B04", "B08"]

CHIP_SIZE = 512


def geotiff_extract_tar(path, save_dir=None):
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            if member.isfile() and "_thumb" in member.name:
                tar.extract(member.name, path=save_dir)
        return None


def geotiff_read_tar(path, save_dir=None):
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".tiff"):
                return BytesIO(tar.extractfile(member.name).read())
        return None


def geotiff_read_combine(img_input_path, mask_input_path=None, output_path=None):
    fp = img_input_path
    if ".tar" in img_input_path.suffixes:
        fp = geotiff_read_tar(img_input_path)
    with rasterio.open(fp) as src:
        data = src.read()
        profile = src.profile
        count = src.count
    if mask_input_path:
        with rasterio.open(mask_input_path) as src2:
            if src2.height != src.height or src2.width != src.width:
                raise ValueError(
                    "Error: The width and height of the Mask do not match those of the original image, concatenation is impossible!")

            data2 = src2.read()
            combined_data = np.concatenate([data, data2], axis=0)

            profile.update(count=count + 1)
            if output_path:
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(combined_data)


def geotiff_rotate_crop(input_path, output_path):
    with rasterio.open(input_path) as src:
        data = src.read()
        profile = src.profile
        transform = src.transform
        count, height, width = src.count, src.height, src.width

    r = data[2]
    g = data[1]
    b = data[0]
    rgb = np.dstack((r, g, b))
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = np.column_stack(np.where(binary > 0))
    rect = cv2.minAreaRect(coords)
    rotate_angle = rect[-1]

    if rotate_angle < -45:
        rotate_angle = -(90 + rotate_angle)
    else:
        rotate_angle = -rotate_angle

    cx, cy = transform * (width / 2.0, height / 2.0)
    rotate_mat = Affine.translation(cx, cy) * Affine.rotation(-rotate_angle) * Affine.translation(-cx, -cy)
    new_transform = rotate_mat * transform
    rotated_data = np.zeros_like(data)

    reproject(
        source=data,
        destination=rotated_data,
        src_transform=transform,
        src_crs=src.crs,
        dst_transform=new_transform,
        dst_crs=src.crs,
        resampling=Resampling.bilinear,
    )

    gray = rotated_data[0]
    mask = np.abs(gray) > 5
    y_idx, x_idx = np.where(mask)
    x1, x2 = x_idx.min(), x_idx.max()
    y1, y2 = y_idx.min(), y_idx.max()

    cropped_data = rotated_data[:, y1:y2, x1:x2]
    cropped_transform = new_transform * Affine.translation(x1, y1)

    profile.update({
        "height": cropped_data.shape[1],
        "width": cropped_data.shape[2],
        "transform": cropped_transform,
        "compress": "deflate",
    })
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cropped_data)


def geotiff_stretch(arr, p=2, a_min=0, a_max=1):
    lo = np.percentile(arr, p)
    hi = np.percentile(arr, 100 - p)
    return np.clip((arr - lo) / (hi - lo), a_min, a_max)


def geotiff_map_class(mask):
    class_mask = np.zeros_like(mask, dtype=np.uint8)
    class_mask[(mask >= 0) & (mask <= 50)] = 0  # background
    class_mask[(mask >= 100) & (mask <= 200)] = 1  # shadow
    class_mask[(mask >= 250) & (mask <= 255)] = 2  # cloud
    return class_mask


def geotiff_show_rgb_thumbnail(data_path, pred_path=None, max_size=512):
    with rasterio.open(data_path) as src:
        h, w = src.height, src.width
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        data = src.read(out_shape=(src.count, new_h, new_w),
                        resampling=rasterio.enums.Resampling.bilinear)

    b = data[0]
    g = data[1]
    r = data[2]
    nir = data[3]
    mask = data[4]
    rgb = np.dstack((r, g, b))

    rgb = geotiff_stretch(rgb)
    mask = geotiff_map_class(mask)

    if pred_path:
        with rasterio.open(pred_path) as src2:
            pred_mask = src2.read(indexes=(1),
                                  out_shape=(src2.count, new_h, new_w),
                                  resampling=rasterio.enums.Resampling.bilinear).astype(np.uint8)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.title("RGB Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title("Mask Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask)
    plt.title("Predicted Mask Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def geotiff_preprocess(target=None):
    root_dir = Path("C:/Users/gaoha/Downloads/GF1_WHU")
    output_dir = Path("../data/GF1_WHU")
    os.makedirs(output_dir, exist_ok=True)

    for tar_file in root_dir.glob("*.tar.gz"):
        dataset_name = tar_file.stem.rsplit(".", 1)[0]
        if target and dataset_name == target:
            img_input_path = root_dir / tar_file
            mask_input_path = root_dir / f"{dataset_name}_ReferenceMask.tif"
            output_path = output_dir / f"{dataset_name}.tif"
            geotiff_read_combine(img_input_path, mask_input_path, output_path)
            geotiff_rotate_crop(output_path, output_path)
            geotiff_show_rgb_thumbnail(output_path)


def get_location_datetime(in_path: Path):
    filename = Path(in_path).stem
    pattern = r"([EW])(\d+\.?\d*)_([NS])(\d+\.?\d*)"
    match = re.search(pattern, filename)
    location = "E00.0-N00.0"
    if match:
        lon_dir, lon_val, lat_dir, lat_val = match.groups()
        location = f"{lon_dir}{float(lon_val)}-{lat_dir}{float(lat_val)}"
    pattern = r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})"
    match = re.search(pattern, filename)
    datetime = "2020-01-01"
    if match:
        year, month, day = match.groups()
        datetime = f"{year}-{month}-{day}"
    return filename, location, datetime


def geotiff_compress(in_dir: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    metadata = []

    tif_files = list(in_dir.glob("*.tif")) + list(in_dir.glob("*.tiff"))
    if not tif_files:
        return

    for in_path in tif_files:
        try:
            logger.debug(f"Compressing：{in_path.name}")
            out_path = out_dir / in_path.name

            with rasterio.open(in_path) as src:
                data = src.read()
                profile = src.profile
                profile.update(
                    compress="deflate",
                    predictor=2,
                )

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)

        except Exception as e:
            logger.debug(f"Failed to compress {in_path.name}: {str(e)}, skip")
        finally:
            filename, location, datetime = get_location_datetime(in_path)
            metadata.append({
                "name": filename,
                "location": location,
                "datetime": datetime,
                "path": str(out_path.resolve()),
            })

    if metadata:
        csv_save_path = out_dir / "tiff_metadata.csv"
        pd.DataFrame(metadata).to_csv(csv_save_path, index=False)
        logger.info(f"done!")


def geotiff_read(path):
    with rasterio.open(path) as src:
        # 读取影像数组数据 (波段数, 高度, 宽度)
        # data = src.read()
        data = src.read((1, 2, 3, 4))
        mask = geotiff_map_class(src.read((5)))
        # 影像完整元数据（包含驱动、宽高、数据类型、坐标系、变换矩阵等）
        meta = src.meta.copy()
        # 地理变换矩阵：用于 像素坐标 ↔ 地理坐标 互转
        transform = src.transform
        # 坐标参考系统（投影信息，如 WGS84 / UTM / 墨卡托等）
        crs = src.crs
        # 影像高度（行数）、宽度（列数）
        h, w = src.height, src.width
        # 正则匹配经纬度和日期
        filename, location, datetime = get_location_datetime(path)
    return filename, data, mask, meta, transform, crs, h, w, location, datetime


def geotiff_generate_chip_windows(name, height, width, chip_size, overlap_ratio=0.2):
    stride = int(chip_size * (1 - overlap_ratio))
    windows = []
    chip_idx = 0

    # Row-wise
    y_start = 0
    while y_start < height:
        y_end = min(y_start + chip_size, height)

        # Column-wise
        x_start = 0
        while x_start < width:
            x_end = min(x_start + chip_size, width)

            chip_id = f"{name}_chip_{chip_idx:04d}"
            windows.append({
                "chip_id": chip_id,
                "x_start": x_start,
                "x_end": x_end,
                "y_start": y_start,
                "y_end": y_end
            })

            x_start += stride
            chip_idx += 1
        y_start += stride

    logger.info(
        f"Generating overlapping windows: Total {len(windows)} chips | Overlap ratio {overlap_ratio * 100:.0f}% | Stride {stride}")
    return windows


def geotiff_save_chip(chip_id, chip_data, chip_mask, img_out_dir, mask_out_dir, crs, orig_transform, chip_size, x_start,
                      y_start, x_end, y_end):
    chip_dir = img_out_dir / chip_id
    chip_dir.mkdir(exist_ok=True, parents=True)
    mask_out_dir.mkdir(exist_ok=True, parents=True)

    x_min = orig_transform.xoff + x_start * orig_transform.a
    y_max = orig_transform.yoff + y_start * orig_transform.e
    x_max = orig_transform.xoff + x_end * orig_transform.a
    y_min = orig_transform.yoff + y_end * orig_transform.e

    transform = from_bounds(x_min, y_min, x_max, y_max, chip_size, chip_size)
    meta = {
        "driver": "GTiff", "height": chip_size, "width": chip_size,
        "count": 1, "dtype": "uint16", "crs": crs, "transform": transform,
        "compress": "deflate", "predictor": 2,
    }

    paths = {}
    for idx, band in enumerate(BANDS):
        path = chip_dir / f"{band}.tif"
        with rasterio.open(path, "w", **meta) as dst:
            dst.write(chip_data[idx], 1)
        paths[f"{band}_path"] = str(path.resolve())

    path = mask_out_dir / f"{chip_id}.tif"
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(chip_mask, 1)
    paths[r"label_path"] = str(path.resolve())
    return paths


def geotiff_generate_metadata(img_in_path, img_out_dir, mask_out_dir):
    img_out_dir.mkdir(exist_ok=True, parents=True)
    mask_out_dir.mkdir(exist_ok=True, parents=True)

    name, data, mask, meta, transform, crs, h, w, location, datetime = geotiff_read(img_in_path)

    img_chip_out_dir = img_out_dir / name
    mask_chip_out_dir = mask_out_dir / name

    chip_size = CHIP_SIZE
    valid_threshold = [0.01, 0.999]

    short_name = name[:3] + "_" + name[-6:]

    windows = geotiff_generate_chip_windows(short_name, h, w, chip_size)
    metadata = []

    for i, win in enumerate(tqdm(windows, desc="slicing")):
        cid = win["chip_id"]
        x1, x2 = win["x_start"], win["x_end"]
        y1, y2 = win["y_start"], win["y_end"]

        chip_mask = mask[y1:y2, x1:x2]
        # valid_pixel = np.sum((chip_mask > 0.01) & (chip_mask <= 2.0))
        # total_pixel = chip_mask.size
        # valid_percent = valid_pixel / total_pixel
        # if valid_percent < valid_threshold[0] or valid_percent > valid_threshold[1]:
        #     continue

        chip_data = data[:, y1:y2, x1:x2]
        paths = geotiff_save_chip(cid, chip_data, chip_mask, img_chip_out_dir, mask_chip_out_dir, crs, transform,
                                  chip_size, x1, y1, x2, y2)

        metadata.append({
            "chip_id": cid,
            "location": location,
            "datetime": datetime,
            "x_start": x1,
            "y_start": y1,
            "x_end": x2,
            "y_end": y2,
            **paths,
        })

    pd.DataFrame(metadata).to_csv(img_out_dir / f"{name}_metadata.csv", index=False)
    logger.info("Process completed!")


def geotiff_restore_metadata(metadata_path, original_img_path, pred_mask_dir, output_path, fusion_method="vote"):
    logger.info("Reading metadata and original image information")
    df_meta = pd.read_csv(metadata_path)
    with rasterio.open(original_img_path) as src:
        orig_h, orig_w = src.height, src.width
        orig_meta = src.meta.copy()
        orig_crs = src.crs
        orig_transform = src.transform

    # pred_full：Used to store the restored prediction results
    # weight_full：Used to record the number of times each pixel is covered (for overlay blending)
    pred_full = np.zeros((orig_h, orig_w), dtype=np.float32)
    weight_full = np.zeros((orig_h, orig_w), dtype=np.float32)

    logger.info("Starting restoring predicted chips to a large image")
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="restoring"):
        x1, y1 = int(row["x_start"]), int(row["y_start"])
        x2, y2 = int(row["x_end"]), int(row["y_end"])
        chip_pred_path = Path(pred_mask_dir) / f"{row['chip_id']}.tif"
        with rasterio.open(chip_pred_path) as src:
            chip_pred = src.read(1)

        chip_h, chip_w = y2 - y1, x2 - x1
        pred_full[y1:y2, x1:x2] += chip_pred[:chip_h, :chip_w]
        weight_full[y1:y2, x1:x2] += 1

    weight_full[weight_full == 0] = 1

    if fusion_method == "average":
        pred_full = pred_full / weight_full
    elif fusion_method == "vote":
        pred_full = np.round(pred_full / weight_full).astype(np.int32)

    logger.info("Saving prediction results as GeoTIFF with geographic coordinates")
    orig_meta.update({
        "count": 1,
        "dtype": np.uint8,  # pred_full.dtype,
        "height": orig_h,
        "width": orig_w,
        "crs": orig_crs,
        "transform": orig_transform,
        "compress": "deflate",
    })

    with rasterio.open(output_path, "w", **orig_meta) as dst:
        dst.write(pred_full, 1)

    logger.info("Restoration completed!")


def get_metadata(features_dir: os.PathLike, bands: List[str] = BANDS):
    chip_ids = (pth.name for pth in features_dir.iterdir() if not pth.name.startswith("."))
    rows = []
    for chip_id in chip_ids:
        row = {"chip_id": chip_id}
        for band in bands:
            row[f"{band}_path"] = features_dir / chip_id / f"{band}.tif"
        rows.append(row)
    return pd.DataFrame(rows)


def make_predictions(
        model: CloudModel,
        x_paths: pd.DataFrame,
        bands: List[str],
        predictions_dir: os.PathLike,
):
    predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(exist_ok=True, parents=True)
    device_type = getattr(model, "device_type", "cpu")
    if device_type in ("cuda", "mps"):
        model = model.to(device_type)
    model.eval()

    test_dataset = CloudDataset(x_paths=x_paths, bands=bands)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=model.batch_size,
        num_workers=model.num_workers,
        shuffle=False,
        pin_memory=device_type == "cuda",
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(test_dataloader):
            logger.debug(f"Predicting batch {batch_index} of {len(test_dataloader)}")
            x = batch["chip"].to(device_type) if device_type in ("cuda", "mps") else batch["chip"]
            preds = model.forward(x)
            # preds = torch.softmax(preds, dim=1)[:, 1]
            # preds = (preds > 0.5).detach().numpy().astype("uint8")
            preds = torch.argmax(preds, dim=1).detach().cpu().numpy().astype("uint8")
            for chip_id, pred in zip(batch["chip_id"], preds):
                chip_pred_path = predictions_dir / f"{chip_id}.tif"
                chip_pred_im = Image.fromarray(pred)
                chip_pred_im.save(chip_pred_path)


def main(
        model_weights_path: Path = Path("benchmark/segformer/assets/cloud_model.pt"),
        test_features_dir: Path = Path(f"../data/images/{DATASET_NAME}"),
        predictions_dir: Path = Path(f"../data/predictions/{DATASET_NAME}"),
        bands: List[str] = ["B02", "B03", "B04", "B08"],
        fast_dev_run: bool = False,
):
    if not test_features_dir.exists():
        raise ValueError(
            f"The directory for test feature images must exist and {test_features_dir} does not exist"
        )
    predictions_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Loading model")
    model = CloudModel(bands=bands, hparams={"weights": None})
    model.load_state_dict(torch.load(model_weights_path))

    logger.info("Loading test metadata")
    test_metadata = get_metadata(test_features_dir, bands=bands)
    if fast_dev_run:
        test_metadata = test_metadata.head()
    logger.info(f"Found {len(test_metadata)} chips")

    logger.info("Generating predictions in batches")
    make_predictions(model, test_metadata, bands, predictions_dir)

    logger.info(f"""Saved {len(list(predictions_dir.glob("*.tif")))} predictions""")


if __name__ == "__main__":
    # geotiff_generate_metadata(f"data/GF1_WHU/compress/{DATASET_NAME}.tif", Path("data/images/GF1_WHU"), Path("data/masks/GF1_WHU"))
    # main()
    # geotiff_restore_metadata(f"data/metadata/{DATASET_NAME}_metadata.csv", f"data/GF1_WHU/{DATASET_NAME}.tif", f"data/predictions/{DATASET_NAME}", f"data/predictions/{DATASET_NAME}_PredictedMask.tif")
    geotiff_show_rgb_thumbnail(f"../data/GF1_WHU/compress/{DATASET_NAME}.tif",
                               f"../data/predictions/{DATASET_NAME}_PredictedMask.tif")
