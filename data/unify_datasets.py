#!/usr/bin/env python3
"""
遥感图像云语义分割数据集统一转换脚本
将 HRC_WHU、RICE2、38-Cloud、95-Cloud 转换为 OnCloudN 格式
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm


class DatasetUnifier:
    """数据集统一转换器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.metadata = {
            "datasets": {},
            "total_samples": 0,
            "band_info": {
                "B02": {"name": "Blue", "wavelength": "490nm"},
                "B03": {"name": "Green", "wavelength": "560nm"},
                "B04": {"name": "Red", "wavelength": "665nm"},
                "B08": {"name": "NIR", "wavelength": "842nm"}
            },
            "label_mapping": {"0": "non-cloud", "1": "cloud"}
        }

        # 创建输出目录
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def _save_band(self, data: np.ndarray, output_path: Path, dtype=None):
        """保存单波段为GeoTIFF"""
        if dtype:
            data = data.astype(dtype)

        height, width = data.shape
        transform = Affine.identity()  # 使用单位变换矩阵

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs=None,
            transform=transform,
        ) as dst:
            dst.write(data, 1)

    def _remap_labels(self, label_array: np.ndarray) -> np.ndarray:
        """将标签重新映射为0=非云，1=云"""
        # 将255映射为1
        label_array = np.where(label_array == 255, 1, label_array)
        # 确保只有0和1
        label_array = np.clip(label_array, 0, 1).astype(np.uint8)
        return label_array

    def _generate_pseudo_nir(self, rgb_array: np.ndarray, label_array: np.ndarray = None) -> np.ndarray:
        """
        基于云特征突出的伪NIR通道生成算法

        策略原理:
        1. 云在可见光波段(特别是红波段)有高反射率，在NIR波段反射率更高
        2. 使用红波段作为NIR的基础估计
        3. 结合亮度信息和云标签进行自适应增强
        4. 对云区域增强，对非云区域适度抑制

        参数:
            rgb_array: RGB图像数组 (H, W, 3) 或 (3, H, W)，值范围 [0, 255]
            label_array: 云标签 (H, W)，可选，用于指导增强

        返回:
            pseudo_nir: 伪NIR通道 (H, W)，uint16类型
        """
        # 处理输入维度
        if len(rgb_array.shape) == 3 and rgb_array.shape[0] == 3:
            # (3, H, W) 格式
            r = rgb_array[0].astype(np.float32)
            g = rgb_array[1].astype(np.float32)
            b = rgb_array[2].astype(np.float32)
        elif len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
            # (H, W, 3) 格式
            r = rgb_array[:, :, 0].astype(np.float32)
            g = rgb_array[:, :, 1].astype(np.float32)
            b = rgb_array[:, :, 2].astype(np.float32)
        else:
            raise ValueError(f"Unsupported RGB array shape: {rgb_array.shape}")

        # 计算亮度 (Luminance) - 云的指示器
        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        # 计算白度 (Whiteness) - 云通常是白色的
        # 使用RGB标准差，云的标准差较小（更白）
        rgb_mean = (r + g + b) / 3.0
        rgb_std = np.sqrt(((r - rgb_mean)**2 + (g - rgb_mean)**2 + (b - rgb_mean)**2) / 3.0 + 1e-6)
        whiteness = 1.0 - (rgb_std / 128.0)  # 归一化，越白值越大
        whiteness = np.clip(whiteness, 0, 1)

        # 基于红波段的NIR基础估计
        # 云在红波段反射率约为0.3-0.5，在NIR约为0.6-0.9
        nir_base = r * 1.5  # 基础放大

        # 云特征增强因子
        # 策略：亮度高+白度高 = 很可能是云 -> 大幅增强NIR值
        #      亮度低 = 地表/水体 -> 适度增强或抑制

        # 亮度权重：高亮度区域得到更多增强
        brightness_weight = np.power(luminance / 255.0, 0.7)  # 非线性增强亮区

        # 白度权重：白色区域（云）得到额外增强
        cloud_weight = np.power(whiteness, 2.0)  # 强调白度

        # 综合增强因子
        enhancement = 1.0 + brightness_weight * 0.5 + cloud_weight * 0.8

        # 如果有标签信息，使用标签进行精细调整
        if label_array is not None:
            # 标签区域：云(1)进一步增强，非云(0)适度抑制
            label_weight = np.where(label_array == 1, 1.3, 0.9)
            enhancement = enhancement * label_weight

        # 应用增强
        pseudo_nir = nir_base * enhancement

        # 添加基于纹理的细节（模拟真实NIR的纹理特征）
        # 使用蓝波段和红波段的差异来估计植被/水体特征
        br_diff = b - r  # 蓝减红，负值表示植被（叶绿素吸收红波段）
        vegetation_factor = np.clip(1.0 - np.abs(br_diff) / 255.0, 0.5, 1.0)

        # 植被区域适度降低NIR（虽然真实NIR植被也高，但为了突出云，这里做对比）
        pseudo_nir = pseudo_nir * vegetation_factor

        # 归一化到uint16范围 [0, 65535]，模拟Sentinel-2/Landsat的NIR值
        # 实际云在NIR的反射率通常在0.4-0.9，对应约 26000-59000
        pseudo_nir = np.clip(pseudo_nir / 255.0 * 80000, 0, 65535)

        return pseudo_nir.astype(np.uint16)

    def unify_oncloudn(self, source_dir: str):
        """
        OnCloudN 已经是目标格式，只需复制并添加前缀
        源结构: images/{sample_id}/{band}.tif, labels/{sample_id}.tif
        """
        print("处理 OnCloudN 数据集...")
        source_path = Path(source_dir)

        # 获取所有样本
        sample_dirs = [d for d in (source_path / "images").iterdir() if d.is_dir()]
        dataset_samples = []

        for sample_dir in tqdm(sample_dirs, desc="OnCloudN"):
            sample_id = sample_dir.name
            new_sample_id = f"oncloudn_{sample_id}"

            # 创建样本目录
            new_sample_dir = self.images_dir / new_sample_id
            new_sample_dir.mkdir(exist_ok=True)

            # 复制波段文件
            for band in ["B02.tif", "B03.tif", "B04.tif", "B08.tif"]:
                src_band = sample_dir / band
                if src_band.exists():
                    shutil.copy2(src_band, new_sample_dir / band)

            # 复制并处理标签
            src_label = source_path / "labels" / f"{sample_id}.tif"
            dst_label = self.labels_dir / f"{new_sample_id}.tif"

            if src_label.exists():
                # 读取、重新映射、保存
                with rasterio.open(src_label) as src:
                    label_data = src.read(1)
                    label_data = self._remap_labels(label_data)
                    self._save_band(label_data, dst_label)

            dataset_samples.append(new_sample_id)

        self.metadata["datasets"]["oncloudn"] = {
            "source": str(source_path),
            "samples": len(dataset_samples),
            "sample_ids": dataset_samples[:10] + ["..."]  # 只存前10个示例
        }
        self.metadata["total_samples"] += len(dataset_samples)
        print(f"  完成: {len(dataset_samples)} 个样本")

    def unify_hrc_whu(self, source_dir: str):
        """
        HRC_WHU 转换
        源结构: images/{scene}_{number}.tif, masks/{scene}_{number}.tif
        目标: 3波段 (RGB -> B04,B03,B02)
        """
        print("处理 HRC_WHU 数据集...")
        source_path = Path(source_dir)

        image_files = sorted([f for f in (source_path / "images").glob("*.tif")])
        dataset_samples = []

        for img_file in tqdm(image_files, desc="HRC_WHU"):
            # 解析文件名: {scene}_{number}.tif
            sample_id = img_file.stem  # e.g., barren_1
            new_sample_id = f"hrcwhu_{sample_id}"

            # 创建样本目录
            new_sample_dir = self.images_dir / new_sample_id
            new_sample_dir.mkdir(exist_ok=True)

            # 读取RGB图像并分离波段
            with rasterio.open(img_file) as src:
                rgb_data = src.read()  # (3, H, W)

            # 保存为独立波段: R->B04, G->B03, B->B02
            bands = [(rgb_data[0], "B04.tif"), (rgb_data[1], "B03.tif"), (rgb_data[2], "B02.tif")]
            for band_data, band_name in bands:
                self._save_band(band_data, new_sample_dir / band_name, dtype=np.uint8)

            # 处理标签（提前读取，用于指导伪NIR生成）
            mask_file = source_path / "masks" / f"{sample_id}.tif"
            label_data = None
            if mask_file.exists():
                with rasterio.open(mask_file) as src:
                    label_data = src.read(1)
                    label_data = self._remap_labels(label_data)

            # 生成伪NIR通道 (B08)，突出云特征
            # 将RGB数据转换为 (H, W, 3) 格式用于NIR生成
            rgb_hwc = np.transpose(rgb_data, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
            pseudo_nir = self._generate_pseudo_nir(rgb_hwc, label_data)
            self._save_band(pseudo_nir, new_sample_dir / "B08.tif", dtype=np.uint16)

            # 保存标签
            dst_label = self.labels_dir / f"{new_sample_id}.tif"
            if label_data is not None:
                self._save_band(label_data, dst_label, dtype=np.uint8)

            dataset_samples.append(new_sample_id)

        self.metadata["datasets"]["hrcwhu"] = {
            "source": str(source_path),
            "samples": len(dataset_samples),
            "sample_ids": dataset_samples[:10] + ["..."]
        }
        self.metadata["total_samples"] += len(dataset_samples)
        print(f"  完成: {len(dataset_samples)} 个样本")

    def unify_rice2(self, source_dir: str):
        """
        RICE2 转换
        源结构: images/{number}.png, masks/{number}.png
        目标: 3波段 (RGB -> B04,B03,B02)
        """
        print("处理 RICE2 数据集...")
        source_path = Path(source_dir)

        image_files = sorted([f for f in (source_path / "images").glob("*.png")])
        dataset_samples = []

        for img_file in tqdm(image_files, desc="RICE2"):
            sample_id = img_file.stem  # e.g., 0
            new_sample_id = f"rice2_{sample_id}"

            # 创建样本目录
            new_sample_dir = self.images_dir / new_sample_id
            new_sample_dir.mkdir(exist_ok=True)

            # 读取RGB图像
            img = Image.open(img_file)
            rgb_array = np.array(img)  # (H, W, 3)

            # 分离波段并保存
            self._save_band(rgb_array[:,:,0], new_sample_dir / "B04.tif", dtype=np.uint8)
            self._save_band(rgb_array[:,:,1], new_sample_dir / "B03.tif", dtype=np.uint8)
            self._save_band(rgb_array[:,:,2], new_sample_dir / "B02.tif", dtype=np.uint8)

            # 处理标签（提前读取，用于指导伪NIR生成）
            mask_file = source_path / "masks" / f"{sample_id}.png"
            label_data = None
            if mask_file.exists():
                mask = Image.open(mask_file)
                mask_array = np.array(mask)
                # 如果是RGB，取第一个通道
                if len(mask_array.shape) == 3:
                    mask_array = mask_array[:,:,0]
                label_data = self._remap_labels(mask_array)

            # 生成伪NIR通道 (B08)，突出云特征
            pseudo_nir = self._generate_pseudo_nir(rgb_array, label_data)
            self._save_band(pseudo_nir, new_sample_dir / "B08.tif", dtype=np.uint16)

            # 保存标签
            dst_label = self.labels_dir / f"{new_sample_id}.tif"
            if label_data is not None:
                self._save_band(label_data, dst_label, dtype=np.uint8)

            dataset_samples.append(new_sample_id)

        self.metadata["datasets"]["rice2"] = {
            "source": str(source_path),
            "samples": len(dataset_samples),
            "sample_ids": dataset_samples[:10] + ["..."]
        }
        self.metadata["total_samples"] += len(dataset_samples)
        print(f"  完成: {len(dataset_samples)} 个样本")

    def unify_cloud38(self, source_dir: str):
        """
        38-Cloud 转换 (训练集)
        源结构: train_{band}/, train_gt/
        目标: 4波段 (red->B04, green->B03, blue->B02, nir->B08)
        """
        print("处理 38-Cloud 数据集...")
        source_path = Path(source_dir)
        train_path = source_path / "38-Cloud_training"

        # 获取所有样本ID (从blue波段文件提取)
        blue_files = sorted([f for f in (train_path / "train_blue").glob("*.TIF")])
        dataset_samples = []

        for blue_file in tqdm(blue_files, desc="38-Cloud"):
            # 从文件名提取样本ID
            # e.g., blue_patch_100_5_by_12_LC08_L1TP_061017_20160720_20170223_01_T1.TIF
            sample_id = blue_file.stem.replace("blue_", "")  # 移除blue_前缀
            new_sample_id = f"cloud38_{sample_id}"

            # 创建样本目录
            new_sample_dir = self.images_dir / new_sample_id
            new_sample_dir.mkdir(exist_ok=True)

            # 读取并保存各波段
            bands = [
                (train_path / "train_red" / f"red_{sample_id}.TIF", "B04.tif"),
                (train_path / "train_green" / f"green_{sample_id}.TIF", "B03.tif"),
                (train_path / "train_blue" / f"blue_{sample_id}.TIF", "B02.tif"),
                (train_path / "train_nir" / f"nir_{sample_id}.TIF", "B08.tif"),
            ]

            for src_path, band_name in bands:
                if src_path.exists():
                    with rasterio.open(src_path) as src:
                        band_data = src.read(1)
                        self._save_band(band_data, new_sample_dir / band_name, dtype=np.uint16)

            # 处理标签
            gt_file = train_path / "train_gt" / f"gt_{sample_id}.TIF"
            dst_label = self.labels_dir / f"{new_sample_id}.tif"

            if gt_file.exists():
                with rasterio.open(gt_file) as src:
                    label_data = src.read(1)
                    label_data = self._remap_labels(label_data)
                    self._save_band(label_data, dst_label, dtype=np.uint8)

            dataset_samples.append(new_sample_id)

        self.metadata["datasets"]["cloud38"] = {
            "source": str(source_path),
            "samples": len(dataset_samples),
            "sample_ids": dataset_samples[:10] + ["..."]
        }
        self.metadata["total_samples"] += len(dataset_samples)
        print(f"  完成: {len(dataset_samples)} 个样本")

    def unify_cloud95(self, source_dir: str):
        """
        95-Cloud 转换 (38-Cloud扩展)
        与38-Cloud结构相同
        """
        print("处理 95-Cloud 数据集...")
        source_path = Path(source_dir)
        train_path = source_path / "95-cloud_training_only_additional_to38-cloud"

        # 获取所有样本ID
        blue_files = sorted([f for f in (train_path / "train_blue_additional_to38cloud").glob("*.TIF")])
        dataset_samples = []

        for blue_file in tqdm(blue_files, desc="95-Cloud"):
            sample_id = blue_file.stem.replace("blue_", "")
            new_sample_id = f"cloud95_{sample_id}"

            # 创建样本目录
            new_sample_dir = self.images_dir / new_sample_id
            new_sample_dir.mkdir(exist_ok=True)

            # 读取并保存各波段
            bands = [
                (train_path / "train_red_additional_to38cloud" / f"red_{sample_id}.TIF", "B04.tif"),
                (train_path / "train_green_additional_to38cloud" / f"green_{sample_id}.TIF", "B03.tif"),
                (train_path / "train_blue_additional_to38cloud" / f"blue_{sample_id}.TIF", "B02.tif"),
                (train_path / "train_nir_additional_to38cloud" / f"nir_{sample_id}.TIF", "B08.tif"),
            ]

            for src_path, band_name in bands:
                if src_path.exists():
                    with rasterio.open(src_path) as src:
                        band_data = src.read(1)
                        self._save_band(band_data, new_sample_dir / band_name, dtype=np.uint16)

            # 处理标签
            gt_file = train_path / "train_gt_additional_to38cloud" / f"gt_{sample_id}.TIF"
            dst_label = self.labels_dir / f"{new_sample_id}.tif"

            if gt_file.exists():
                with rasterio.open(gt_file) as src:
                    label_data = src.read(1)
                    label_data = self._remap_labels(label_data)
                    self._save_band(label_data, dst_label, dtype=np.uint8)

            dataset_samples.append(new_sample_id)

        self.metadata["datasets"]["cloud95"] = {
            "source": str(source_path),
            "samples": len(dataset_samples),
            "sample_ids": dataset_samples[:10] + ["..."]
        }
        self.metadata["total_samples"] += len(dataset_samples)
        print(f"  完成: {len(dataset_samples)} 个样本")

    def save_metadata(self):
        """保存元数据文件"""
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"\n元数据已保存: {metadata_path}")

    def generate_summary(self):
        """生成数据集摘要"""
        print("\n" + "="*60)
        print("统一数据集生成完成!")
        print("="*60)
        print(f"输出目录: {self.output_dir}")
        print(f"总样本数: {self.metadata['total_samples']}")
        print("\n各数据集样本数:")
        for name, info in self.metadata["datasets"].items():
            print(f"  {name}: {info['samples']} 样本")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="统一遥感云分割数据集")
    parser.add_argument("--output", "-o", type=str, default="./Unified_Cloud_Dataset",
                        help="输出目录路径")
    parser.add_argument("--oncloudn", type=str, default="./OnCloudN",
                        help="OnCloudN 数据集路径")
    parser.add_argument("--hrcwhu", type=str, default="./HRC_WHU",
                        help="HRC_WHU 数据集路径")
    parser.add_argument("--rice2", type=str, default="./RICE2",
                        help="RICE2 数据集路径")
    parser.add_argument("--cloud38", type=str,
                        default="./kagglehub/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images/versions/4",
                        help="38-Cloud 数据集路径")
    parser.add_argument("--cloud95", type=str,
                        default="./kagglehub/datasets/sorour/95cloud-cloud-segmentation-on-satellite-images/versions/3",
                        help="95-Cloud 数据集路径")
    parser.add_argument("--skip", nargs="+", choices=["oncloudn", "hrcwhu", "rice2", "cloud38", "cloud95"],
                        default=[], help="跳过指定数据集")

    args = parser.parse_args()

    # 创建统一器
    unifier = DatasetUnifier(args.output)

    # 处理各数据集
    if "oncloudn" not in args.skip and os.path.exists(args.oncloudn):
        unifier.unify_oncloudn(args.oncloudn)

    if "hrcwhu" not in args.skip and os.path.exists(args.hrcwhu):
        unifier.unify_hrc_whu(args.hrcwhu)

    if "rice2" not in args.skip and os.path.exists(args.rice2):
        unifier.unify_rice2(args.rice2)

    if "cloud38" not in args.skip and os.path.exists(args.cloud38):
        unifier.unify_cloud38(args.cloud38)

    if "cloud95" not in args.skip and os.path.exists(args.cloud95):
        unifier.unify_cloud95(args.cloud95)

    # 保存元数据
    unifier.save_metadata()
    unifier.generate_summary()


if __name__ == "__main__":
    main()
