import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from loguru import logger
from pathlib import Path
import numpy as np
from functools import partial
import concurrent
import concurrent.futures
from typing import Dict, Any, List, Optional, Sequence, Tuple
import rasterio
import torch
import torch.nn.functional as F

from MyCloudSenseNet.benchmark.cloud_dataset import CloudDataset
from MyCloudSenseNet.benchmark.cloud_model import CloudModel

BANDS = ["B02", "B03", "B04", "B08"]
CHIP_SIZE = 512
OVERLAP_RATIO = 0.2
VALID_THRESHOLD = [0.001, 0.999]
DATA_DIR = Path.cwd().parent.resolve() / "data"
MODEL_NAME = "unet"
DEFAULT_TTA_MODES = ("none", "hflip", "vflip", "hvflip")
PREDICTION_PAD_DIVISOR = 32


def mask2label(mask):
    label = np.zeros_like(mask, dtype=np.uint8)
    label[(mask >= 0) & (mask <= 50)] = 0  # background
    label[(mask >= 100) & (mask <= 200)] = 1  # shadow
    label[(mask >= 250) & (mask <= 255)] = 2  # cloud
    return label

def stretch(arr, p=2, a_min=0, a_max=1):
    lo = np.percentile(arr, p)
    hi = np.percentile(arr, 100 - p)
    return np.clip((arr - lo) / (hi - lo), a_min, a_max)


def is_label_valid(label: np.ndarray, valid_threshold: List[float] = VALID_THRESHOLD) -> bool:
    valid_pixel = np.sum((label > 0.01) & (label <= 2.0))
    total_pixel = label.size
    valid_percent = valid_pixel / total_pixel
    return valid_threshold[0] <= valid_percent <= valid_threshold[1]


def normalize_tta_modes(tta_modes: Optional[Sequence[str]] = DEFAULT_TTA_MODES) -> Tuple[str, ...]:
    if not tta_modes:
        return ("none",)

    valid_modes = {"none", "hflip", "vflip", "hvflip"}
    modes = tuple(tta_modes)
    unknown = sorted(set(modes) - valid_modes)
    if unknown:
        raise ValueError(f"Unknown TTA modes: {unknown}. Choose from {sorted(valid_modes)}")
    if "none" not in modes:
        modes = ("none", *modes)
    return modes


def apply_tta(image: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "hflip":
        return torch.flip(image, dims=(-1,))
    if mode == "vflip":
        return torch.flip(image, dims=(-2,))
    if mode == "hvflip":
        return torch.flip(image, dims=(-2, -1))
    return image


def undo_tta(pred: torch.Tensor, mode: str) -> torch.Tensor:
    return apply_tta(pred, mode)


def predict_batch_probabilities(
        model: CloudModel,
        x: torch.Tensor,
        tta_modes: Optional[Sequence[str]] = DEFAULT_TTA_MODES,
) -> torch.Tensor:
    modes = normalize_tta_modes(tta_modes)
    prob_sum = None

    for mode in modes:
        logits = model(apply_tta(x, mode))
        logits = undo_tta(logits, mode)
        probs = F.softmax(logits, dim=1)
        prob_sum = probs if prob_sum is None else prob_sum + probs

    return prob_sum / len(modes)


def load_cloud_model(
        model_weights_path: Path = Path(f"benchmark/{MODEL_NAME}/assets/cloud_model.pt"),
        bands: List[str] = BANDS,
        model_name: str = MODEL_NAME,
) -> CloudModel:
    logger.info("Loading model")
    model = CloudModel(bands=bands, hparams={"weights": None}, model_name=model_name)
    device_type = getattr(model, "device_type", "cpu")
    map_location = torch.device(device_type) if device_type in ("cuda", "mps") else torch.device("cpu")
    model.load_state_dict(torch.load(model_weights_path, map_location=map_location))
    model.eval()
    return model


def iter_chip_probability_batches(
        model: CloudModel,
        x_paths: pd.DataFrame,
        bands: List[str] = BANDS,
        tta_modes: Optional[Sequence[str]] = DEFAULT_TTA_MODES,
):
    device_type = getattr(model, "device_type", "cpu")
    if device_type in ("cuda", "mps"):
        model = model.to(device_type)

    dataset = CloudDataset(x_paths=x_paths.reset_index(drop=True), bands=bands)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=model.batch_size,
        num_workers=model.num_workers,
        shuffle=False,
        pin_memory=device_type == "cuda",
        collate_fn=pad_prediction_batch,
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            x = batch["chip"]
            if device_type in ("cuda", "mps"):
                x = x.to(device_type)

            probs = predict_batch_probabilities(model, x, tta_modes=tta_modes)
            yield batch["chip_id"], probs.detach().cpu().numpy(), batch["shape"]


def round_up(value: int, divisor: int) -> int:
    if divisor <= 1:
        return value
    return ((value + divisor - 1) // divisor) * divisor


def pad_prediction_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    chips = [torch.as_tensor(item["chip"], dtype=torch.float32) for item in batch]
    shapes = [(int(chip.shape[-2]), int(chip.shape[-1])) for chip in chips]
    max_h = round_up(max(height for height, _ in shapes), PREDICTION_PAD_DIVISOR)
    max_w = round_up(max(width for _, width in shapes), PREDICTION_PAD_DIVISOR)

    padded_chips = []
    for chip in chips:
        height, width = chip.shape[-2], chip.shape[-1]
        padded_chips.append(F.pad(chip, (0, max_w - width, 0, max_h - height)))

    return {
        "chip_id": [item["chip_id"] for item in batch],
        "chip": torch.stack(padded_chips, dim=0),
        "shape": shapes,
    }


def save_chip_predictions(
        model: CloudModel,
        x_paths: pd.DataFrame,
        pred_out_dir: Path,
        bands: List[str] = BANDS,
        tta_modes: Optional[Sequence[str]] = DEFAULT_TTA_MODES,
        save_probabilities: bool = False,
) -> None:
    pred_out_dir.mkdir(exist_ok=True, parents=True)
    prob_out_dir = pred_out_dir / "probabilities"
    if save_probabilities:
        prob_out_dir.mkdir(exist_ok=True, parents=True)

    metadata_by_chip_id = x_paths.set_index("chip_id")
    for chip_ids, probs, shapes in iter_chip_probability_batches(model, x_paths, bands=bands, tta_modes=tta_modes):
        preds = np.argmax(probs, axis=1).astype("uint8")
        for chip_id, pred, prob, shape in zip(chip_ids, preds, probs, shapes):
            height, width = shape
            pred = pred[:height, :width]
            prob = prob[:, :height, :width]
            row = metadata_by_chip_id.loc[chip_id]
            save_prediction_geotiff(pred, row[f"{bands[0]}_path"], pred_out_dir / f"{chip_id}.tif")
            if save_probabilities:
                np.save(prob_out_dir / f"{chip_id}.npy", prob.astype("float16"))

    logger.info(f"""Saved {len(list(pred_out_dir.glob("*.tif")))} predictions""")


def save_prediction_geotiff(pred: np.ndarray, reference_path: Path | str, output_path: Path) -> None:
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()

    profile.update(
        count=1,
        dtype="uint8",
        height=pred.shape[0],
        width=pred.shape[1],
        compress="deflate",
        predictor=2,
    )
    profile.pop("nodata", None)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred.astype("uint8"), 1)


def get_chip_metadata(features_dir: Path, bands: List[str] = BANDS) -> pd.DataFrame:
    features_dir = Path(features_dir)
    if not features_dir.exists():
        raise ValueError(f"features_dir does not exist: {features_dir}")

    rows = []
    if all((features_dir / f"{band}.tif").exists() for band in bands):
        rows.append({
            "chip_id": features_dir.name,
            **{f"{band}_path": str((features_dir / f"{band}.tif").resolve()) for band in bands},
        })
        return pd.DataFrame(rows)

    chip_dirs = [path for path in features_dir.iterdir() if path.is_dir() and not path.name.startswith(".")]
    for chip_dir in sorted(chip_dirs):
        if not all((chip_dir / f"{band}.tif").exists() for band in bands):
            logger.warning(f"Skipping {chip_dir.name}: missing one or more band files")
            continue

        row = {"chip_id": chip_dir.name}
        for band in bands:
            row[f"{band}_path"] = str((chip_dir / f"{band}.tif").resolve())
        rows.append(row)

    if not rows:
        raise ValueError(
            f"No valid chip folders found in {features_dir}. "
            f"Expected either {', '.join(f'{band}.tif' for band in bands)} directly, "
            "or subfolders containing those files."
        )

    return pd.DataFrame(rows)


def load_chip_metadata(
        features: Path | pd.DataFrame,
        bands: List[str] = BANDS,
) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features.copy()

    features = Path(features)
    if features.suffix.lower() == ".csv":
        metadata = pd.read_csv(features)
    else:
        metadata = get_chip_metadata(features, bands=bands)

    required_columns = {"chip_id", *(f"{band}_path" for band in bands)}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"Chip metadata is missing required columns: {sorted(missing)}")

    return metadata


def predict_small_chips(
        features: Path | pd.DataFrame,
        pred_out_dir: Path,
        model_weights_path: Path = Path(f"benchmark/{MODEL_NAME}/assets/cloud_model.pt"),
        bands: List[str] = BANDS,
        model_name: str = MODEL_NAME,
        fast_dev_run: bool = False,
        tta_modes: Optional[Sequence[str]] = DEFAULT_TTA_MODES,
        save_probabilities: bool = False,
) -> pd.DataFrame:
    x_paths = load_chip_metadata(features, bands=bands)
    model = load_cloud_model(model_weights_path, bands=bands, model_name=model_name)

    if fast_dev_run:
        x_paths = x_paths.head(max(model.batch_size, 1))

    logger.info(f"Found {len(x_paths)} small chips")
    logger.info("Generating small-chip predictions in batches")
    save_chip_predictions(
        model=model,
        x_paths=x_paths,
        pred_out_dir=Path(pred_out_dir),
        bands=bands,
        tta_modes=tta_modes,
        save_probabilities=save_probabilities,
    )
    return x_paths


class GeoTIFFTiler:
    def __init__(self, df_row: pd.Series, is_for_training: bool = True, display_thumbnail: bool = False):
        self.df_row = df_row
        self.info = None
        self.is_for_training = is_for_training
        self._read(display_thumbnail)

    def _read(self, display_thumbnail: bool):
        with rasterio.open(self.df_row.path) as src:
            data = src.read((1, 2, 3, 4))
            mask = src.read((5))
            self.info = {
                "data": data,
                "bit": int(math.log2(data.max() + 1)),
                "label": mask2label(mask),
                "meta": src.meta.copy(),
                "transform": src.transform,
                "crs": src.crs,
                "height": src.height,
                "width": src.width,
            }
            if display_thumbnail:
                self._display_thumbnail(src)

    def _display_thumbnail(self, src):
        max_size = 512
        h, w = src.height, src.width
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        thumbnail = src.read(out_shape=(src.count, new_h, new_w),
                             resampling=rasterio.enums.Resampling.bilinear)

        name = Path(self.df_row.path).stem
        b, g, r, nir, mask = thumbnail[:5]
        rgb = stretch(np.dstack((r, g, b)))

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        logger.debug(name)
        fig.suptitle(name, fontsize=14, fontweight="bold")

        ax[0].imshow(rgb)
        ax[0].set_title("RGB Image")

        ax[1].imshow(mask)
        ax[1].set_title("Mask Image")

        plt.tight_layout()
        plt.show()

    def calculate_transform(self, x_start: int, y_start: int, x_end: int, y_end: int, chip_size: int) -> rasterio.Affine:
        orig_transform = self.info["transform"]
        x_min = orig_transform.xoff + x_start * orig_transform.a
        y_max = orig_transform.yoff + y_start * orig_transform.e
        x_max = orig_transform.xoff + x_end * orig_transform.a
        y_min = orig_transform.yoff + y_end * orig_transform.e

        return rasterio.transform.from_bounds(x_min, y_min, x_max, y_max, chip_size, chip_size)

    def generate_chip_windows(self, chip_size=CHIP_SIZE, overlap_ratio=OVERLAP_RATIO):
        stride = int(chip_size * (1 - overlap_ratio))
        height, width = self.info.get("height", 0), self.info.get("width", 0)
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

                chip_id = f"{self.df_row.filename}_chip_{chip_idx:04d}"
                windows.append({
                    "chip_idx": chip_idx,
                    "chip_id": chip_id,
                    "chip_size": chip_size,
                    "x_start": x_start,
                    "x_end": x_end,
                    "y_start": y_start,
                    "y_end": y_end,
                    "transform": self.calculate_transform(x_start, y_start, x_end, y_end, chip_size),
                })

                x_start += stride
                chip_idx += 1
            y_start += stride

        self.info["windows"] = windows

        logger.info(f"Generating overlapping windows: Total {len(windows)} chips | Overlap ratio {overlap_ratio * 100:.0f}% | Stride {stride}")

    def save_chip(self, chip_data: np.ndarray, chip_label: np.ndarray, chip_info: Dict[str, Any], output_dirs: Dict[str, Path]) -> Dict[str, str]:
        chip_id = chip_info["chip_id"]
        chip_size = chip_info["chip_size"]
        chip_dir = output_dirs["img"] / chip_id
        chip_dir.mkdir(exist_ok=True, parents=True)
        label_out_dir = output_dirs["label"]
        label_out_dir.mkdir(exist_ok=True, parents=True)

        meta = {
            "driver": "GTiff",
            "height": chip_size,
            "width": chip_size,
            "count": 1,
            "dtype": "uint16",
            "crs": chip_info["crs"],
            "transform": chip_info["transform"],
            "compress": "deflate",
            "predictor": 2,
        }

        paths = {}
        for idx, band in enumerate(BANDS):
            path = chip_dir / f"{band}.tif"
            if not path.exists():
                with rasterio.open(path, "w", **meta) as dst:
                    dst.write(chip_data[idx], 1)
            paths[f"{band}_path"] = str(path.resolve())

        path = label_out_dir / f"{chip_id}.tif"
        if not path.exists():
            with rasterio.open(path, "w", **meta) as dst:
                dst.write(chip_label, 1)
        paths[r"label_path"] = str(path.resolve())
        return paths

    def get_single_chip(self, window: Dict[str, Any], full_data: np.ndarray, full_label: np.ndarray, output_dirs: Dict[str, Path]) -> Dict[str, Any]:
        x1, x2 = window["x_start"], window["x_end"]
        y1, y2 = window["y_start"], window["y_end"]

        chip_label = full_label[y1:y2, x1:x2]

        if self.is_for_training and not is_label_valid(chip_label):
            return None

        chip_data = full_data[:, y1:y2, x1:x2]

        chip_info = {
            "chip_id": window["chip_id"],
            "chip_size": window["chip_size"],
            "transform": window["transform"],
            "crs": self.info["crs"],
            "x_start": x1, "x_end": x2,
            "y_start": y1, "y_end": y2
        }

        paths = self.save_chip(chip_data, chip_label, chip_info, output_dirs)

        return {
            "chip_id": window["chip_id"],
            "location": self.df_row.location,
            "datetime": self.df_row.datetime,
            "x_start": x1, "x_end": x2,
            "y_start": y1, "y_end": y2,
            **paths,
        }

    def get_chips(self, csv_out_dir: Path, img_out_dir: Path, label_out_dir: Path, max_workers: int = 4) -> None:
        chip_size = CHIP_SIZE
        # if not self.is_for_training:
        #     chip_size = CHIP_SIZE * 2
        self.generate_chip_windows(chip_size=chip_size)
        csv_out_dir.mkdir(exist_ok=True, parents=True)
        img_out_dir.mkdir(exist_ok=True, parents=True)
        label_out_dir.mkdir(exist_ok=True, parents=True)

        filename = self.df_row.filename
        img_chip_out_dir = img_out_dir / filename
        label_chip_out_dir = label_out_dir / filename

        img_chip_out_dir.mkdir(exist_ok=True, parents=True)
        label_chip_out_dir.mkdir(exist_ok=True, parents=True)

        output_dirs = {
            "img": img_chip_out_dir,
            "label": label_chip_out_dir
        }

        metadata = []
        windows = self.info["windows"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            process_func = partial(
                self.get_single_chip,
                full_data=self.info["data"],
                full_label=self.info["label"],
                output_dirs=output_dirs
            )
            futures = {executor.submit(process_func, win): win for win in windows}

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(windows), desc="Processing chips"):
                result = future.result()
                if result:
                    metadata.append(result)

        chip_df = pd.DataFrame(metadata)
        self.info["metadata"] = chip_df
        chip_df.to_csv(csv_out_dir / f"{filename}_metadata.csv", index=False)
        logger.info(f"Process {filename} completed! Generated {len(metadata)} valid chips.")

    def predict_chips(
            self,
            pred_out_dir: Path,
            fast_dev_run: bool = False,
            tta_modes: Optional[Sequence[str]] = DEFAULT_TTA_MODES,
            save_probabilities: bool = False,
    ):
        logger.info("Loading model weights")
        model_weights_path: Path = Path(f"benchmark/{MODEL_NAME}/assets/cloud_model.pt")
        model = load_cloud_model(model_weights_path, bands=BANDS, model_name=MODEL_NAME)

        logger.info("Loading metadata")
        x_paths = self.info["metadata"]
        if fast_dev_run:
            x_paths = x_paths.head(max(model.batch_size, 1))
        logger.info(f"Found {len(x_paths)} chips")

        logger.info("Generating predictions in batches")
        save_chip_predictions(
            model=model,
            x_paths=x_paths,
            pred_out_dir=pred_out_dir,
            bands=BANDS,
            tta_modes=tta_modes,
            save_probabilities=save_probabilities,
        )

    def predict(
            self,
            pred_out_dir: Path = DATA_DIR / "predictions",
            fusion_method: str = "probability",
            fast_dev_run: bool = False,
            tta_modes: Optional[Sequence[str]] = DEFAULT_TTA_MODES,
            save_chip_masks: bool = True,
            save_probabilities: bool = False,
    ):
        logger.info("Starting predicting chips")
        pred_label_dir = pred_out_dir / self.df_row.filename
        pred_label_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Loading model weights")
        model_weights_path: Path = Path(f"benchmark/{MODEL_NAME}/assets/cloud_model.pt")
        model = load_cloud_model(model_weights_path, bands=BANDS, model_name=MODEL_NAME)

        logger.info("Loading metadata")
        x_paths = self.info["metadata"]
        if fast_dev_run:
            x_paths = x_paths.head(max(model.batch_size, 1))
        logger.info(f"Found {len(x_paths)} chips")

        height, width = self.info["height"], self.info["width"]
        meta, crs, transform = self.info["meta"], self.info["crs"], self.info["transform"]
        num_classes = getattr(model, "num_classes", 3)

        prob_full = np.zeros((num_classes, height, width), dtype=np.float32)
        weight_full = np.zeros((height, width), dtype=np.float32)

        prob_out_dir = pred_label_dir / "probabilities"
        if save_probabilities:
            prob_out_dir.mkdir(exist_ok=True, parents=True)

        metadata_by_chip_id = x_paths.set_index("chip_id")
        logger.info("Generating TTA predictions and restoring chips to a large image")
        for chip_ids, probs, shapes in tqdm(
                iter_chip_probability_batches(model, x_paths, bands=BANDS, tta_modes=tta_modes),
                total=math.ceil(len(x_paths) / max(model.batch_size, 1)),
                desc="predicting/restoring",
        ):
            preds = np.argmax(probs, axis=1).astype("uint8")
            for chip_id, pred, prob, shape in zip(chip_ids, preds, probs, shapes):
                row = metadata_by_chip_id.loc[chip_id]
                x1, y1 = int(row["x_start"]), int(row["y_start"])
                x2, y2 = int(row["x_end"]), int(row["y_end"])
                chip_h, chip_w = y2 - y1, x2 - x1
                height, width = shape
                pred = pred[:height, :width]
                prob = prob[:, :height, :width]

                prob_full[:, y1:y2, x1:x2] += prob[:, :chip_h, :chip_w]
                weight_full[y1:y2, x1:x2] += 1

                if save_chip_masks:
                    save_prediction_geotiff(pred, row[f"{BANDS[0]}_path"], pred_label_dir / f"{chip_id}.tif")
                if save_probabilities:
                    np.save(prob_out_dir / f"{chip_id}.npy", prob.astype("float16"))

        weight_full[weight_full == 0] = 1

        if fusion_method == "average":
            logger.warning("fusion_method='average' now averages class probabilities before argmax.")
            pred_full = np.argmax(prob_full / weight_full[None, :, :], axis=0).astype(np.uint8)
        elif fusion_method == "vote":
            logger.warning("fusion_method='vote' is kept for compatibility; probability fusion is used with TTA.")
            pred_full = np.argmax(prob_full / weight_full[None, :, :], axis=0).astype(np.uint8)
        elif fusion_method == "probability":
            pred_full = np.argmax(prob_full / weight_full[None, :, :], axis=0).astype(np.uint8)
        else:
            raise ValueError("fusion_method must be one of: 'probability', 'vote', 'average'")

        logger.info("Saving prediction results as GeoTIFF with geographic coordinates")
        meta.update({
            "count": 1,
            "dtype": np.uint8,  # pred_full.dtype,
            "height": height,
            "width": width,
            "crs": crs,
            "transform": transform,
            "compress": "deflate",
            "predictor": 2,
        })

        pred_out_path = pred_out_dir / f"{self.df_row.filename}_PredictedMask.tif"
        with rasterio.open(pred_out_path, "w", **meta) as dst:
            dst.write(pred_full, 1)

        logger.info("Restoration completed!")


def intersection_over_union_and_coverage(pred, true, n_classes=3):
    """
    Calculates intersection and union for a batch of images.
    Calculates coverage of each class for a batch of images.
    """
    if pred.shape != true.shape:
        raise ValueError(
            f"pred and true must have the same shape, got pred={pred.shape}, true={true.shape}. "
            "Use a chip prediction with the matching chip label, or use a restored full-image prediction."
        )

    total_pixels = true.size
    valid_pixel_mask = (true != 255)  # valid pixel mask
    true = true[valid_pixel_mask]
    pred = pred[valid_pixel_mask]

    iou_list = []
    true_coverage_list = []
    pred_coverage_list = []
    for cls in range(n_classes):
        # skip background
        if cls == 0:
            continue

        # Prediction/ground truth mask of the current category
        true_cls = (true == cls)
        pred_cls = (pred == cls)

        # Intersection and union totals
        intersection = np.logical_and(true_cls, pred_cls)
        union = np.logical_or(true_cls, pred_cls)

        iou = intersection.sum() / (union.sum() + 1e-8)
        iou_list.append(min(iou, 1))

        true_coverage = true_cls.sum() / total_pixels
        pred_coverage = pred_cls.sum() / total_pixels
        true_coverage_list.append(true_coverage)
        pred_coverage_list.append(pred_coverage)

    mIoU = np.mean(iou_list)

    return mIoU, true_coverage_list, pred_coverage_list


def read_prediction_and_aligned_true(data_path: Path, pred_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with rasterio.open(data_path) as data_src, rasterio.open(pred_path) as pred_src:
        pred = pred_src.read(1).astype(np.uint8)

        if pred_src.height == data_src.height and pred_src.width == data_src.width:
            true = mask2label(data_src.read(5).astype(np.uint8))
            return pred, true

        if pred_src.transform.is_identity:
            raise ValueError(
                f"Prediction {pred_path} is smaller than the full image but has no GeoTIFF transform. "
                "Regenerate chip predictions with predict_small_chips()/GeoTIFFTiler.predict(), "
                "or evaluate this prediction against its matching chip label_path."
            )

        if data_src.crs and pred_src.crs and data_src.crs != pred_src.crs:
            raise ValueError(
                f"Cannot align prediction to label because CRS differs: pred={pred_src.crs}, true={data_src.crs}"
            )

        try:
            window = rasterio.windows.from_bounds(*pred_src.bounds, transform=data_src.transform)
            true_mask = data_src.read(
                5,
                window=window,
                out_shape=(pred_src.height, pred_src.width),
                boundless=False,
                resampling=rasterio.enums.Resampling.nearest,
            ).astype(np.uint8)
        except Exception as exc:
            raise ValueError(
                f"Prediction shape {pred.shape} does not match full label shape "
                f"({data_src.height}, {data_src.width}), and automatic geospatial alignment failed. "
                "For chip predictions, pass the matching chip label or save predictions with GeoTIFF transform."
            ) from exc

        true = mask2label(true_mask)
        if pred.shape != true.shape:
            raise ValueError(
                f"Aligned true mask still does not match prediction: pred={pred.shape}, true={true.shape}"
            )
        return pred, true


def display_thumbnail_more(data_path, pred_path=None, max_size=512):
    logger.info(f"Displaying thumbnail for {data_path.name}")
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

    rgb = stretch(rgb)

    fig, ax = plt.subplots(1, 3, figsize=(24, 8))


    ax[0].imshow(rgb)
    ax[0].set_title("RGB Image")

    ax[1].imshow(mask)
    ax[1].set_title("True Mask Image")

    if pred_path:
        pred, true_for_metric = read_prediction_and_aligned_true(data_path, pred_path)
        with rasterio.open(pred_path) as src2:
            pred_mask = src2.read(indexes=(1),
                                  out_shape=(src2.count, new_h, new_w),
                                  resampling=rasterio.enums.Resampling.bilinear).astype(np.uint8)

        iou, true_cov, pred_cov = intersection_over_union_and_coverage(pred, true_for_metric)

        ax[1].set_title(f"True Mask Image\nshadow coverage={true_cov[0]:04f} | cloud coverage={true_cov[1]:04f} ")
        ax[2].imshow(pred_mask)
        ax[2].set_title(f"Predicted Mask Image with iou={iou:04f}\nshadow coverage={pred_cov[0]:04f} | cloud coverage={pred_cov[1]:04f}")

    plt.tight_layout()
    plt.show()
