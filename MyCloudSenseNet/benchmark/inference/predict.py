"""Prediction module for cloud detection - v2.0

推理模块 - 支持新架构模型和自适应TTA
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..configs import Configs, ModelConfig
from ..configs.defaults import (
    PREDICTION_PAD_DIVISOR,
    PROBABILITY_DTYPE,
    DEFAULT_COMPRESSION,
    DEFAULT_PREDICTOR,
    OUTPUT_DTYPE,
    logger,
)
from ..utils.utils import ensure_dir, round_up


def load_model(
    model_weights_path: Path,
    config: Optional[ModelConfig] = None,
    device: str = "auto",
) -> CloudModel:
    """
    加载云检测模型

    Args:
        model_weights_path: 模型权重路径
        config: 模型配置（默认使用 balanced）
        device: 设备 ('cuda', 'mps', 'cpu', 'auto')

    Returns:
        加载好的模型（eval模式）
    """
    logger.info(f"Loading model from {model_weights_path}")
    from ..models.cloud_model import CloudModel

    config = copy.deepcopy(config) if config is not None else Configs.balanced()
    config.validate()

    model = CloudModel(
        config=config,
    )

    if device == "auto":
        device = model.device_type

    map_location = torch.device(device)
    checkpoint = torch.load(model_weights_path, map_location=map_location)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model = model.to(device)
    model.device_type = str(device)
    model.gpu = str(device).startswith("cuda")

    logger.info(f"Model loaded on {device}")
    return model


def predict(
    model: CloudModel,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    单样本/批次预测

    Args:
        model: CloudModel
        x: 输入 [B, C, H, W]

    Returns:
        (probabilities, info)
    """
    model.eval()
    from ..utils.tta import predict_with_tta

    with torch.no_grad():
        x = x.to(model.device_type)

        probs, info = predict_with_tta(
            model=lambda img: model(img),
            x=x,
            tta_strategy=model.config.tta_strategy if model.config.use_tta else "none",
            confidence_threshold=model.config.tta_threshold,
        )

    return probs, info


def pad_batch(
    batch: List[Dict[str, Any]],
    pad_divisor: int = PREDICTION_PAD_DIVISOR,
) -> Dict[str, Any]:
    """
    将批次填充到统一尺寸

    Args:
        batch: 批次列表
        pad_divisor: 填充除数

    Returns:
        填充后的批次字典
    """
    chips = [torch.as_tensor(item["chip"], dtype=torch.float32) for item in batch]
    shapes = [(int(chip.shape[-2]), int(chip.shape[-1])) for chip in chips]

    max_h = round_up(max(h for h, _ in shapes), pad_divisor)
    max_w = round_up(max(w for _, w in shapes), pad_divisor)

    padded_chips = []
    for chip in chips:
        h, w = chip.shape[-2], chip.shape[-1]
        padded_chips.append(F.pad(chip, (0, max_w - w, 0, max_h - h)))

    return {
        "chip_id": [item["chip_id"] for item in batch],
        "chip": torch.stack(padded_chips, dim=0),
        "shape": shapes,
    }


def build_dataloader(
    model: CloudModel,
    x_paths: pd.DataFrame,
) -> torch.utils.data.DataLoader:
    """
    构建推理数据加载器

    Args:
        model: CloudModel
        x_paths: 数据路径DataFrame

    Returns:
        DataLoader
    """
    from ..dataset import CloudDataset

    dataset = CloudDataset(x_paths=x_paths.reset_index(drop=True), bands=model.config.bands)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=model.config.batch_size,
        num_workers=model.config.num_workers,
        shuffle=False,
        pin_memory=model.gpu,
        collate_fn=pad_batch,
    )


def iter_predict(
    model: CloudModel,
    x_paths: pd.DataFrame,
):
    """
    迭代预测批次

    Args:
        model: CloudModel
        x_paths: 数据路径

    Yields:
        (chip_ids, predictions, probabilities, shapes)
    """
    dataloader = build_dataloader(model, x_paths)

    with torch.no_grad():
        for batch in dataloader:
            x = batch["chip"].to(model.device_type)

            probs, info = predict(model, x)
            probs_np = probs.cpu().numpy()
            preds = np.argmax(probs_np, axis=1).astype(np.uint8)

            yield batch["chip_id"], preds, probs_np, batch["shape"]


def save_geotiff(
    pred: np.ndarray,
    reference_path: Path,
    output_path: Path,
) -> None:
    """
    保存预测结果为GeoTIFF

    Args:
        pred: 预测数组 [H, W]
        reference_path: 参考图像路径（用于地理信息）
        output_path: 输出路径
    """
    import rasterio

    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()

    profile.update(
        count=1,
        dtype=OUTPUT_DTYPE,
        height=pred.shape[0],
        width=pred.shape[1],
        compress=DEFAULT_COMPRESSION,
        predictor=DEFAULT_PREDICTOR,
    )
    profile.pop("nodata", None)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred.astype(np.uint8), 1)


def crop(
    pred: np.ndarray,
    prob: np.ndarray,
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    裁剪到原始尺寸

    Args:
        pred: 预测 [H, W]
        prob: 概率 [C, H, W]
        shape: 目标尺寸 (h, w)

    Returns:
        (cropped_pred, cropped_prob)
    """
    h, w = shape
    return pred[:h, :w], prob[:, :h, :w]


def fuse(
    prob_full: np.ndarray,
    weight_full: np.ndarray,
) -> np.ndarray:
    """
    融合重叠区域的概率

    Args:
        prob_full: 累积概率 [C, H, W]
        weight_full: 权重 [H, W]

    Returns:
        融合后的标签 [H, W]
    """
    weight_full = np.maximum(weight_full, 1)
    normalized = prob_full / weight_full[None, :, :]
    return np.argmax(normalized, axis=0).astype(np.uint8)


def run(
    model: CloudModel,
    x_paths: pd.DataFrame,
    output_dir: Path,
    save_probs: bool = False,
) -> int:
    """
    运行推理并保存结果

    Args:
        model: CloudModel
        x_paths: 输入数据路径
        output_dir: 输出目录
        save_probs: 是否保存概率文件

    Returns:
        预测数量
    """
    ensure_dir(output_dir)
    bands = model.config.bands

    if save_probs:
        prob_dir = output_dir / "probabilities"
        ensure_dir(prob_dir)

    metadata = x_paths.set_index("chip_id")
    count = 0

    for chip_ids, preds, probs, shapes in iter_predict(model, x_paths):
        for chip_id, pred, prob, shape in zip(chip_ids, preds, probs, shapes):
            pred, prob = crop(pred, prob, shape)

            row = metadata.loc[chip_id]
            ref_path = row[f"{bands[0]}_path"]
            save_geotiff(pred, ref_path, output_dir / f"{chip_id}.tif")

            if save_probs:
                np.save(prob_dir / f"{chip_id}.npy", prob.astype(PROBABILITY_DTYPE))

            count += 1

    logger.info(f"Saved {count} predictions to {output_dir}")
    return count


def simple(
    model_path: Path,
    input_paths: pd.DataFrame,
    output_dir: Path,
    config: Optional[ModelConfig] = None,
    device: str = "auto",
    save_probs: bool = False,
) -> int:
    """
    简单推理接口 - 一键预测

    Args:
        model_path: 模型权重路径
        input_paths: 输入数据路径
        output_dir: 输出目录
        config: 模型配置（默认balanced）

    Returns:
        预测数量
    """
    model = load_model(model_path, config=config, device=device)
    return run(
        model,
        input_paths,
        output_dir,
        save_probs=save_probs,
    )
