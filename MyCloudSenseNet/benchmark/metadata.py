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
from PIL import Image

from MyCloudSenseNet.benchmark.cloud_dataset import CloudDataset
from MyCloudSenseNet.benchmark.cloud_model import CloudModel

BANDS = ["B02", "B03", "B04", "B08"]
CHIP_SIZE = 512
OVERLAP_RATIO = 0.2
VALID_THRESHOLD = [0.001, 0.999]
DATA_DIR = Path.cwd().parent.resolve() / "data"
MODEL_NAME = "unet"
DEFAULT_TTA_MODES = ("none", "hflip", "vflip", "hvflip")


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
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if batch_index % 30 == 0:
                logger.debug(f"Predicting batch {batch_index} of {len(dataloader)}")

            x = batch["chip"]
            if device_type in ("cuda", "mps"):
                x = x.to(device_type)

            probs = predict_batch_probabilities(model, x, tta_modes=tta_modes)
            yield batch["chip_id"], probs.detach().cpu().numpy()


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

    for chip_ids, probs in iter_chip_probability_batches(model, x_paths, bands=bands, tta_modes=tta_modes):
        preds = np.argmax(probs, axis=1).astype("uint8")
        for chip_id, pred, prob in zip(chip_ids, preds, probs):
            Image.fromarray(pred).save(pred_out_dir / f"{chip_id}.tif")
            if save_probabilities:
                np.save(prob_out_dir / f"{chip_id}.npy", prob.astype("float16"))

    logger.info(f"""Saved {len(list(pred_out_dir.glob("*.tif")))} predictions""")


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
        for chip_ids, probs in tqdm(
                iter_chip_probability_batches(model, x_paths, bands=BANDS, tta_modes=tta_modes),
                total=math.ceil(len(x_paths) / max(model.batch_size, 1)),
                desc="predicting/restoring",
        ):
            preds = np.argmax(probs, axis=1).astype("uint8")
            for chip_id, pred, prob in zip(chip_ids, preds, probs):
                row = metadata_by_chip_id.loc[chip_id]
                x1, y1 = int(row["x_start"]), int(row["y_start"])
                x2, y2 = int(row["x_end"]), int(row["y_end"])
                chip_h, chip_w = y2 - y1, x2 - x1

                prob_full[:, y1:y2, x1:x2] += prob[:, :chip_h, :chip_w]
                weight_full[y1:y2, x1:x2] += 1

                if save_chip_masks:
                    Image.fromarray(pred).save(pred_label_dir / f"{chip_id}.tif")
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
