#!/usr/bin/env python3
"""
统一云分割数据集加载器
支持加载按OnCloudN格式组织的多个数据集
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class UnifiedCloudDataset(Dataset):
    """
    统一云分割数据集加载器

    支持功能:
    - 按数据集名称筛选样本
    - 数据增强
    - 波段选择
    - 归一化
    - 划分训练/验证/测试集
    """

    # 波段名称映射
    BAND_NAMES = {
        "B02": "Blue",
        "B03": "Green",
        "B04": "Red",
        "B08": "NIR"
    }

    # 可用波段 (按OnCloudN命名)
    AVAILABLE_BANDS = ["B02", "B03", "B04", "B08"]

    # 数据集特定的归一化参数 (基于物理意义的反射率缩放)
    # Sentinel-2 和 Landsat 使用 10000 作为反射率缩放因子 (即 0.01 = 100)
    # 详见: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/resolutions/radiometric
    DATASET_NORM_PARAMS = {
        # 多光谱卫星数据: uint16, TOA反射率 × 10000
        "oncloudn": {"type": "reflectance", "divisor": 10000.0, "clip": (0, 1), "dtype": "uint16"},
        "cloud38": {"type": "reflectance", "divisor": 10000.0, "clip": (0, 1), "dtype": "uint16"},
        "cloud95": {"type": "reflectance", "divisor": 10000.0, "clip": (0, 1), "dtype": "uint16"},
        # RGB数据集: uint8, DN值，需要映射到近似反射率范围
        "hrcwhu": {"type": "dn_to_reflectance", "divisor": 255.0, "clip": (0, 1), "dtype": "uint8"},
        "rice2": {"type": "dn_to_reflectance", "divisor": 255.0, "clip": (0, 1), "dtype": "uint8"},
    }

    # 波段特定的典型值范围 (用于异常检测和替代归一化)
    BAND_STATISTICS = {
        "B02": {"mean": 0.12, "std": 0.08, "min": 0, "max": 0.6},    # Blue
        "B03": {"mean": 0.15, "std": 0.09, "min": 0, "max": 0.7},    # Green
        "B04": {"mean": 0.16, "std": 0.10, "min": 0, "max": 0.8},    # Red
        "B08": {"mean": 0.25, "std": 0.15, "min": 0, "max": 1.0},    # NIR
    }

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        bands: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        normalize: str = "auto",
        norm_params: Optional[Dict] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        compute_stats: bool = False
    ):
        """
        参数:
            data_dir: 统一数据集根目录
            split: 'train', 'val', 或 'test'
            bands: 要加载的波段列表，默认 [B02, B03, B04, B08]
            datasets: 要加载的子数据集列表，默认全部
                      可选: oncloudn, hrcwhu, rice2, cloud38, cloud95
            transform: 数据增强变换函数
            normalize: 归一化策略
                      - "auto": 自动根据数据集选择 (默认)
                      - "reflectance": 除以10000，裁剪到[0,1] (卫星标准)
                      - "minmax": Min-Max归一化到[0,1]
                      - "standard": 标准分数 (x-mean)/std
                      - "bandwise_standard": 波段独立标准分数
                      - "percentile": 1-99百分位数归一化
                      - "none": 不归一化
            norm_params: 自定义归一化参数，覆盖默认参数
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
            compute_stats: 是否预计算数据集统计信息（用于标准分数归一化）
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.bands = bands or ["B02", "B03", "B04", "B08"]
        self.datasets_filter = datasets
        self.transform = transform
        self.normalize_mode = normalize
        self.custom_norm_params = norm_params or {}
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # 验证波段名称
        for band in self.bands:
            if band not in self.AVAILABLE_BANDS:
                raise ValueError(f"未知波段: {band}. 可用波段: {self.AVAILABLE_BANDS}")

        # 验证归一化模式
        valid_modes = ["auto", "reflectance", "minmax", "standard", "bandwise_standard", "percentile", "none"]
        if normalize not in valid_modes:
            raise ValueError(f"未知归一化模式: {normalize}. 可用: {valid_modes}")

        # 加载元数据
        self.metadata_path = self.data_dir / "metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"未找到元数据文件: {self.metadata_path}")

        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # 获取样本列表
        self.samples = self._get_samples()

        # 划分数据集
        self.sample_ids = self._split_samples(random_seed)

        # 预计算统计信息（用于标准分数归一化）
        self.dataset_stats = {}
        if compute_stats or normalize in ["standard", "bandwise_standard"]:
            self._compute_dataset_statistics()

        print(f"[{split}] 加载 {len(self.sample_ids)} 个样本 (归一化: {normalize})")

    def _get_samples(self) -> List[Dict]:
        """获取所有可用样本"""
        images_dir = self.data_dir / "images"
        labels_dir = self.data_dir / "labels"

        samples = []

        # 遍历所有样本目录
        for sample_dir in images_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            sample_id = sample_dir.name

            # 检查数据集筛选
            if self.datasets_filter:
                dataset_prefix = sample_id.split('_')[0]
                if dataset_prefix not in self.datasets_filter:
                    continue

            # 检查标签文件是否存在
            label_path = labels_dir / f"{sample_id}.tif"
            if not label_path.exists():
                continue

            # 检查所需波段是否都存在
            available_bands = [f.name for f in sample_dir.glob("*.tif")]
            missing_bands = [b for b in self.bands if f"{b}.tif" not in available_bands]
            if missing_bands:
                continue  # 跳过缺少波段的样本

            # 检查样本所属的数据集
            dataset_name = sample_id.split('_')[0]

            samples.append({
                "id": sample_id,
                "dataset": dataset_name,
                "image_dir": sample_dir,
                "label_path": label_path
            })

        return samples

    def _split_samples(self, random_seed: int) -> List[str]:
        """划分训练/验证/测试集"""
        # 按数据集分层划分
        dataset_groups = {}
        for sample in self.samples:
            dataset = sample["dataset"]
            if dataset not in dataset_groups:
                dataset_groups[dataset] = []
            dataset_groups[dataset].append(sample["id"])

        train_ids, val_ids, test_ids = [], [], []

        random.seed(random_seed)
        for dataset, ids in dataset_groups.items():
            ids = sorted(ids)
            random.shuffle(ids)

            n = len(ids)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            # test_ratio 用于计算测试集大小，确保三个比例之和不超过1
            n_test = int(n * self.test_ratio)
            # 如果三个比例之和小于1，剩余样本分配给测试集
            remaining = n - n_train - n_val - n_test
            if remaining > 0:
                n_test += remaining

            train_ids.extend(ids[:n_train])
            val_ids.extend(ids[n_train:n_train+n_val])
            test_ids.extend(ids[n_train+n_val:n_train+n_val+n_test])

        if self.split == "train":
            return train_ids
        elif self.split == "val":
            return val_ids
        elif self.split == "test":
            return test_ids
        else:
            raise ValueError(f"未知的split: {self.split}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        返回:
            image: 图像数组 (C, H, W)
            label: 标签数组 (H, W)
            info: 样本信息字典
        """
        sample_id = self.sample_ids[idx]

        # 找到样本信息
        sample_info = next(s for s in self.samples if s["id"] == sample_id)
        image_dir = sample_info["image_dir"]
        label_path = sample_info["label_path"]

        # 加载波段
        band_data = []
        for band in self.bands:
            band_path = image_dir / f"{band}.tif"
            with rasterio.open(band_path) as src:
                data = src.read(1)
                band_data.append(data)

        image = np.stack(band_data, axis=0)  # (C, H, W)

        # 加载标签
        with rasterio.open(label_path) as src:
            label = src.read(1)

        # 归一化
        if self.normalize_mode != "none":
            image = self._normalize_image(image, sample_info["dataset"])

        # 数据增强
        if self.transform:
            image, label = self.transform(image, label)

        info = {
            "id": sample_id,
            "dataset": sample_info["dataset"],
            "bands": self.bands
        }

        return image, label, info

    def _compute_dataset_statistics(self, max_samples: int = 100):
        """预计算数据集统计信息用于标准分数归一化"""
        print("正在计算数据集统计信息...")

        # 按数据集和波段分组收集统计
        stats_by_dataset_band = defaultdict(lambda: defaultdict(list))

        for i, sample_id in enumerate(self.sample_ids):
            if i >= max_samples:
                break

            sample_info = next(s for s in self.samples if s["id"] == sample_id)
            dataset = sample_info["dataset"]
            image_dir = sample_info["image_dir"]

            for band in self.bands:
                band_path = image_dir / f"{band}.tif"
                if band_path.exists():
                    with rasterio.open(band_path) as src:
                        data = src.read(1).astype(np.float32)
                        # 排除nodata值
                        data = data[data > 0]
                        if len(data) > 0:
                            stats_by_dataset_band[dataset][band].append({
                                'mean': data.mean(),
                                'std': data.std(),
                                'min': data.min(),
                                'max': data.max(),
                                'p1': np.percentile(data, 1),
                                'p99': np.percentile(data, 99)
                            })

        # 聚合统计
        for dataset, band_stats in stats_by_dataset_band.items():
            self.dataset_stats[dataset] = {}
            for band, stats_list in band_stats.items():
                if len(stats_list) > 0:
                    self.dataset_stats[dataset][band] = {
                        'mean': np.mean([s['mean'] for s in stats_list]),
                        'std': np.mean([s['std'] for s in stats_list]),
                        'min': np.min([s['min'] for s in stats_list]),
                        'max': np.max([s['max'] for s in stats_list]),
                        'p1': np.mean([s['p1'] for s in stats_list]),
                        'p99': np.mean([s['p99'] for s in stats_list])
                    }

        print("统计信息计算完成")

    def _normalize_image(self, image: np.ndarray, dataset: str) -> np.ndarray:
        """
        多策略归一化

        参数:
            image: 图像数组 (C, H, W)，原始DN值
            dataset: 数据集名称

        返回:
            归一化后的图像 (C, H, W)，float32
        """
        # 获取归一化参数
        norm_params = self.custom_norm_params.get(dataset) or \
                     self.DATASET_NORM_PARAMS.get(dataset, {"type": "minmax", "divisor": 1.0})

        image = image.astype(np.float32)

        # 根据归一化模式处理
        if self.normalize_mode == "auto":
            # 自动模式：基于数据集类型选择
            norm_type = norm_params.get("type", "reflectance")
            if norm_type == "reflectance":
                # 卫星数据：TOA反射率归一化
                image = image / norm_params.get("divisor", 10000.0)
                image = np.clip(image, 0, 1)
            elif norm_type == "dn_to_reflectance":
                # RGB数据：DN值映射到近似反射率
                # 假设DN 0-255映射到反射率 0-0.3（典型地表范围）
                image = image / 255.0 * 0.3
            else:
                image = image / norm_params.get("divisor", 255.0)
                image = np.clip(image, 0, 1)

        elif self.normalize_mode == "reflectance":
            # 强制使用卫星标准反射率归一化
            image = image / 10000.0
            image = np.clip(image, 0, 1)

        elif self.normalize_mode == "minmax":
            # Min-Max归一化到[0, 1]
            image_min = image.min(axis=(1, 2), keepdims=True)
            image_max = image.max(axis=(1, 2), keepdims=True)
            image = (image - image_min) / (image_max - image_min + 1e-8)

        elif self.normalize_mode == "bandwise_minmax":
            # 波段独立Min-Max
            for c in range(image.shape[0]):
                band_min = image[c].min()
                band_max = image[c].max()
                image[c] = (image[c] - band_min) / (band_max - band_min + 1e-8)

        elif self.normalize_mode == "standard":
            # 标准分数 (x - mean) / std，使用全局统计
            for c, band in enumerate(self.bands):
                band_stats = self.BAND_STATISTICS.get(band, {"mean": 0.15, "std": 0.1})
                mean = band_stats["mean"]
                std = band_stats["std"]
                image[c] = (image[c] - mean) / std

        elif self.normalize_mode == "bandwise_standard":
            # 波段独立标准分数，使用数据集统计
            for c, band in enumerate(self.bands):
                if dataset in self.dataset_stats and band in self.dataset_stats[dataset]:
                    stats = self.dataset_stats[dataset][band]
                    mean = stats['mean']
                    std = stats['std']
                else:
                    # 回退到典型值
                    band_stats = self.BAND_STATISTICS.get(band, {"mean": 0.15, "std": 0.1})
                    mean = band_stats["mean"] * 10000  # 转换为DN值
                    std = band_stats["std"] * 10000
                image[c] = (image[c] - mean) / (std + 1e-8)

        elif self.normalize_mode == "percentile":
            # 1-99百分位数归一化（对异常值鲁棒）
            for c in range(image.shape[0]):
                p1, p99 = np.percentile(image[c], [1, 99])
                image[c] = (image[c] - p1) / (p99 - p1 + 1e-8)
            image = np.clip(image, 0, 1)

        # "none" 模式：直接返回原始值

        return image

    def get_normalization_info(self) -> Dict:
        """获取当前归一化配置信息"""
        info = {
            "mode": self.normalize_mode,
            "bands": self.bands,
            "dataset_params": {},
            "computed_stats": {}
        }

        for dataset in self.DATASET_NORM_PARAMS:
            info["dataset_params"][dataset] = self.DATASET_NORM_PARAMS[dataset]

        if self.dataset_stats:
            info["computed_stats"] = self.dataset_stats

        return info

    def denormalize(self, image: np.ndarray, dataset: str = "oncloudn") -> np.ndarray:
        """
        反归一化，将归一化后的图像转回原始DN值范围

        参数:
            image: 归一化后的图像 (C, H, W)
            dataset: 数据集名称

        返回:
            反归一化后的图像
        """
        norm_params = self.DATASET_NORM_PARAMS.get(dataset, {"divisor": 10000.0})

        if self.normalize_mode in ["auto", "reflectance"]:
            if norm_params.get("type") == "reflectance":
                return image * norm_params["divisor"]
            else:
                return image * 255.0
        elif self.normalize_mode == "minmax":
            # Min-Max无法精确反归一化，需要原始min/max
            raise NotImplementedError("Min-Max归一化无法精确反归一化，请保存原始范围")
        elif self.normalize_mode in ["standard", "bandwise_standard"]:
            for c, band in enumerate(self.bands):
                if dataset in self.dataset_stats and band in self.dataset_stats[dataset]:
                    stats = self.dataset_stats[dataset][band]
                    mean, std = stats['mean'], stats['std']
                else:
                    band_stats = self.BAND_STATISTICS.get(band, {"mean": 0.15, "std": 0.1})
                    mean = band_stats["mean"] * 10000
                    std = band_stats["std"] * 10000
                image[c] = image[c] * std + mean
            return image
        else:
            return image

    def get_class_distribution(self) -> Dict[int, int]:
        """获取类别分布统计"""
        class_counts = {0: 0, 1: 0}

        for sample_id in self.sample_ids:
            sample_info = next(s for s in self.samples if s["id"] == sample_id)
            with rasterio.open(sample_info["label_path"]) as src:
                label = src.read(1)
                unique, counts = np.unique(label, return_counts=True)
                for val, count in zip(unique, counts):
                    if val in class_counts:
                        class_counts[val] += count

        return class_counts

    def print_statistics(self):
        """打印数据集统计信息"""
        print("="*60)
        print(f"数据集: {self.split}")
        print(f"样本数: {len(self)}")
        print(f"波段: {self.bands}")
        print(f"数据源: {self._get_dataset_distribution()}")

        class_dist = self.get_class_distribution()
        total = sum(class_dist.values())
        print(f"类别分布:")
        for cls, count in class_dist.items():
            print(f"  类别 {cls}: {count} ({count/total*100:.2f}%)")
        print("="*60)

    def _get_dataset_distribution(self) -> Dict[str, int]:
        """获取各数据集的样本数"""
        dist = {}
        for sample_id in self.sample_ids:
            dataset = sample_id.split('_')[0]
            dist[dataset] = dist.get(dataset, 0) + 1
        return dist


class CloudAugmentation:
    """
    云分割数据增强
    
    支持随机裁剪、翻转、亮度/对比度调整、高斯噪声等
    支持 3通道 (RGB) 或 4通道 (RGB+NIR) 输入
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.use_nir = config.get('use_nir', True)
        
        if self.enabled:
            print(f"[CloudAugmentation] Enabled with use_nir={self.use_nir}")
    
    def __call__(self, image, mask, mode='train'):
        """
        应用数据增强
        
        Args:
            image: numpy.ndarray [C, H, W] 或 torch.Tensor [C, H, W]
            mask: numpy.ndarray [H, W] 或 torch.Tensor [H, W]
            mode: 'train' or 'val'
            
        Returns:
            image: torch.Tensor [C, H, W]
            mask: torch.Tensor [H, W]
        """
        import torch
        import torchvision.transforms.functional as TF
        
        # 统一转换为 torch.Tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        
        # 根据 use_nir 调整通道数
        if not self.use_nir and image.shape[0] == 4:
            image = image[:3]  # 只保留RGB
        
        if not self.enabled or mode != 'train':
            return image, mask
        
        # 训练模式的数据增强
        # 1. 随机裁剪
        crop_size = self.config.get('random_crop_size', None)
        if crop_size is not None:
            i, j, h, w = T.RandomCrop.get_params(image, output_size=crop_size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask.unsqueeze(0), i, j, h, w).squeeze(0)
        
        # 2. 随机水平翻转
        if torch.rand(1) < self.config.get('horizontal_flip', 0.5):
            image = TF.hflip(image)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        
        # 3. 随机垂直翻转
        if torch.rand(1) < self.config.get('vertical_flip', 0.5):
            image = TF.vflip(image)
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
        
        # 4. 亮度调整 (只应用于RGB通道)
        brightness = self.config.get('brightness', 0)
        if brightness > 0:
            brightness_factor = torch.empty(1).uniform_(1 - brightness, 1 + brightness).item()
            if image.shape[0] == 4:
                rgb = image[:3]
                nir = image[3:]
                rgb = TF.adjust_brightness(rgb, brightness_factor)
                image = torch.cat([rgb, nir], dim=0)
            else:
                image = TF.adjust_brightness(image, brightness_factor)
        
        # 5. 对比度调整 (只应用于RGB通道)
        contrast = self.config.get('contrast', 0)
        if contrast > 0:
            contrast_factor = torch.empty(1).uniform_(1 - contrast, 1 + contrast).item()
            if image.shape[0] == 4:
                rgb = image[:3]
                nir = image[3:]
                rgb = TF.adjust_contrast(rgb, contrast_factor)
                image = torch.cat([rgb, nir], dim=0)
            else:
                image = TF.adjust_contrast(image, contrast_factor)
        
        # 6. 高斯噪声
        noise_std = self.config.get('gaussian_noise', 0)
        if noise_std > 0:
            noise = torch.randn_like(image) * noise_std
            image = image + noise
            image = torch.clamp(image, 0, 1)
        
        return image, mask


def _unified_dataset_collate(batch):
    """
    自定义 collate 函数，将 UnifiedCloudDataset 的 tuple 输出转换为 dict
    
    输入: list of (image, mask, info) tuples
    输出: dict with batched tensors
    """
    import torch
    
    images = []
    masks = []
    filenames = []
    datasets = []
    
    for image, mask, info in batch:
        # 统一转换为 tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        
        images.append(image)
        masks.append(mask)
        filenames.append(info['id'])
        datasets.append(info['dataset'])
    
    # Stack into batch
    return {
        'image': torch.stack(images),
        'mask': torch.stack(masks),
        'filename': filenames,
        'dataset': datasets,
    }


def create_mixed_dataloaders(config: dict, dev_run: bool = False):
    """
    创建多数据集混合数据加载器
    
    支持:
    - 多数据集混合训练
    - 自动统一通道和位宽
    - 从训练集分层采样 val/test
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader, Subset
    
    data_cfg = config['data']
    train_cfg = config['training']
    
    # 统一配置
    use_nir = data_cfg.get('use_nir', True)
    target_channels = 4 if use_nir else 3
    
    # 数据增强
    aug_config = train_cfg.get('augmentation', {'enabled': False})
    aug_config['use_nir'] = use_nir
    transform = CloudAugmentation(aug_config)
    
    # 获取数据集配置
    datasets = data_cfg.get('datasets', [])
    if not datasets:
        raise ValueError("No datasets configured for multi-dataset mode")
    
    # 提取数据集名称列表
    dataset_names = [ds['name'].lower().replace('_', '') for ds in datasets]
    print(f"\n[MixedDataLoader] Loading datasets: {dataset_names}")
    print(f"  Target: {target_channels}ch (use_nir={use_nir})")
    
    # 获取 val/test 分割配置
    split_cfg = data_cfg.get('val_test_split', {})
    train_ratio = split_cfg.get('train_ratio', 0.75)
    val_ratio = split_cfg.get('val_ratio', 0.15)
    test_ratio = split_cfg.get('test_ratio', 0.10)
    seed = split_cfg.get('seed', 42)
    
    # 统一数据集目录
    unified_data_dir = data_cfg.get('unified_data_dir', './Data/Unified_Cloud_Dataset')
    
    # 归一化方法
    normalize = data_cfg.get('normalize', 'percentile')
    
    # 确定波段
    bands = ["B02", "B03", "B04", "B08"] if use_nir else ["B02", "B03", "B04"]
    
    print(f"  Normalize: {normalize}, Bands: {bands}")
    
    # 创建三个数据集
    train_dataset = UnifiedCloudDataset(
        data_dir=unified_data_dir,
        split='train',
        bands=bands,
        datasets=dataset_names if dataset_names else None,
        transform=transform,
        normalize=normalize,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
    )
    
    val_dataset = UnifiedCloudDataset(
        data_dir=unified_data_dir,
        split='val',
        bands=bands,
        datasets=dataset_names if dataset_names else None,
        transform=None,  # 验证集不使用数据增强
        normalize=normalize,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
    )
    
    test_dataset = UnifiedCloudDataset(
        data_dir=unified_data_dir,
        split='test',
        bands=bands,
        datasets=dataset_names if dataset_names else None,
        transform=None,
        normalize=normalize,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
    )
    
    # Dev run: 限制数据集大小
    if dev_run:
        train_samples = min(int(len(train_dataset) * 0.1), len(train_dataset))
        val_samples = min(int(len(val_dataset) * 0.1), len(val_dataset))
        test_samples = min(int(len(test_dataset) * 0.1), len(test_dataset))
        
        train_dataset = Subset(train_dataset, range(train_samples))
        val_dataset = Subset(val_dataset, range(val_samples))
        test_dataset = Subset(test_dataset, range(test_samples))
        
        print(f"[Dev Run] Limited: train={train_samples}, val={val_samples}, test={test_samples}")
    
    print(f"[MixedDataLoader] Loaded: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # 创建 DataLoader
    num_workers = train_cfg.get('num_workers', 4)
    batch_size = train_cfg.get('batch_size', 8)
    
    # 4通道需要调整 batch_size
    if use_nir and batch_size > 4:
        adjusted = max(4, batch_size // 2)
        print(f"[MixedDataLoader] Adjusted batch_size: {batch_size} -> {adjusted} (4-channel)")
        batch_size = adjusted
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_unified_dataset_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_unified_dataset_collate
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_unified_dataset_collate
    )
    
    return train_loader, val_loader, test_loader
