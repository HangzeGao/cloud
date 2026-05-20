"""
云分割数据集加载器 - 精简版
直接使用上层 Data 目录的 UnifiedCloudDataset
保留数据增强和辅助功能
"""
import sys
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class ImageNormalizer:
    """
    业界标准的图像归一化方法
    
    支持多种遥感/卫星图像常用的归一化策略：
    - 'max': 最大值归一化
    - 'percentile': 基于分位数的归一化 (业界推荐)
    - 'zscore': Z-score 标准化
    - 'histeq': 直方图均衡化
    - 'clahe': 对比度受限的自适应直方图均衡化 (业界标准)
    - 'sensor': 传感器特定定标
    """
    
    def __init__(self, method: str = 'percentile', bit_depth: int = None,
                 lower_percentile: float = 1.0, upper_percentile: float = 99.0,
                 clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.method = method
        self.bit_depth = bit_depth
        self.lower_p = lower_percentile
        self.upper_p = upper_percentile
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.dataset_mean = None
        self.dataset_std = None
    
    def __call__(self, image: np.ndarray, per_image: bool = True) -> np.ndarray:
        img = image.astype(np.float32)
        
        if self.method == 'max':
            return self._normalize_max(img)
        elif self.method == 'percentile':
            return self._normalize_percentile(img, per_image)
        elif self.method == 'zscore':
            return self._normalize_zscore(img, per_image)
        elif self.method == 'histeq':
            return self._normalize_histeq(img)
        elif self.method == 'clahe':
            return self._normalize_clahe(img)
        elif self.method == 'sensor':
            return self._normalize_sensor(img)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def _normalize_max(self, img: np.ndarray) -> np.ndarray:
        if self.bit_depth is not None:
            max_val = (1 << self.bit_depth) - 1
        else:
            if img.dtype == np.uint8 or img.max() <= 255:
                max_val = 255.0
            elif img.dtype == np.uint16 or img.max() <= 65535:
                max_val = 65535.0
            else:
                max_val = img.max()
        return img / max_val
    
    def _normalize_percentile(self, img: np.ndarray, per_image: bool) -> np.ndarray:
        if per_image:
            lower = np.percentile(img, self.lower_p)
            upper = np.percentile(img, self.upper_p)
        else:
            lower = getattr(self, 'dataset_lower', np.percentile(img, self.lower_p))
            upper = getattr(self, 'dataset_upper', np.percentile(img, self.upper_p))
        img_normalized = (img - lower) / (upper - lower + 1e-8)
        return np.clip(img_normalized, 0, 1)
    
    def _normalize_zscore(self, img: np.ndarray, per_image: bool) -> np.ndarray:
        if per_image or self.dataset_mean is None:
            mean = img.mean()
            std = img.std()
        else:
            mean = self.dataset_mean
            std = self.dataset_std
        std = std + 1e-8
        return (img - mean) / std
    
    def _normalize_histeq(self, img: np.ndarray) -> np.ndarray:
        try:
            from skimage import exposure
        except ImportError:
            return self._normalize_max(img)
        result = np.zeros_like(img)
        for c in range(img.shape[-1]):
            channel = img[:, :, c]
            channel_min, channel_max = channel.min(), channel.max()
            channel_norm = (channel - channel_min) / (channel_max - channel_min + 1e-8)
            channel_eq = exposure.equalize_hist(channel_norm)
            result[:, :, c] = channel_eq
        return result.astype(np.float32)
    
    def _normalize_clahe(self, img: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError:
            return self._normalize_histeq(img)
        result = np.zeros_like(img)
        for c in range(img.shape[-1]):
            channel = img[:, :, c]
            channel_min, channel_max = channel.min(), channel.max()
            channel_uint8 = ((channel - channel_min) / (channel_max - channel_min + 1e-8) * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            channel_clahe = clahe.apply(channel_uint8)
            result[:, :, c] = channel_clahe.astype(np.float32) / 255.0
        return result
    
    def _normalize_sensor(self, img: np.ndarray) -> np.ndarray:
        img = self._normalize_max(img)
        return np.clip(img, 0, 1)
    
    def compute_dataset_statistics(self, images: list):
        all_pixels = []
        for img_path in images:
            img = np.array(Image.open(img_path))
            all_pixels.extend(img.flatten())
        all_pixels = np.array(all_pixels, dtype=np.float32)
        self.dataset_mean = all_pixels.mean()
        self.dataset_std = all_pixels.std()
        self.dataset_lower = np.percentile(all_pixels, self.lower_p)
        self.dataset_upper = np.percentile(all_pixels, self.upper_p)

    def set_bit_depth(self, bit_depth: int):
        """动态设置位宽，支持任意位宽（如8, 13, 16）"""
        self.bit_depth = bit_depth


class NIRGenerator:
    """
    伪 NIR 通道生成器
    
    从 RGB 图像生成伪近红外（NIR）通道的多种算法
    基于业界常用的物理模型和植被指数方法
    
    Methods:
        - 'physical': 基于物理模型的 NIR 估计 (NIR ≈ 0.7*R + 0.25*G + 0.05*B)
        - 'vegetation': 基于 NDVI 的植被增强 (更适合有云区域)
        - 'enhanced': 增强型，结合亮度和色彩信息
        - 'weighted': 自适应加权组合
    """
    
    def __init__(self, method: str = 'physical', gain: float = 1.1):
        self.method = method
        self.gain = gain
        
        print(f"[NIRGenerator] Initialized with method='{method}', gain={gain}")
    
    def __call__(self, rgb_image):
        """
        从 RGB 图像生成 NIR 通道
        
        Args:
            rgb_image: torch.Tensor [3, H, W] or [B, 3, H, W], 值范围 [0, 1]
            
        Returns:
            nir: torch.Tensor [1, H, W] or [B, 1, H, W], 值范围 [0, 1]
        """
        if isinstance(rgb_image, torch.Tensor):
            # 处理 torch Tensor
            if rgb_image.dim() == 3:
                r, g, b = rgb_image[0], rgb_image[1], rgb_image[2]
            elif rgb_image.dim() == 4:
                r, g, b = rgb_image[:, 0], rgb_image[:, 1], rgb_image[:, 2]
            else:
                raise ValueError(f"Expected 3 or 4 dim tensor, got {rgb_image.dim()}")
            
            if self.method == 'physical':
                nir = 0.7 * r + 0.25 * g + 0.05 * b
            elif self.method == 'vegetation':
                # 植被指数风格：增强绿色，抑制蓝色
                nir = 0.6 * r + 0.4 * g - 0.1 * b
                nir = torch.clamp(nir, 0, 1)
            elif self.method == 'enhanced':
                # 增强型：结合亮度
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                nir = 0.5 * r + 0.3 * g + 0.2 * luminance
            elif self.method == 'weighted':
                # 自适应：根据红色通道强度调整
                red_weight = torch.sigmoid((r - 0.5) * 4)  # 自适应权重
                nir = red_weight * (0.8 * r + 0.2 * g) + (1 - red_weight) * (0.5 * r + 0.3 * g + 0.2 * b)
            else:
                raise ValueError(f"Unknown NIR method: {self.method}")
            
            nir = nir * self.gain
            nir = torch.clamp(nir, 0, 1)
            
            # 添加通道维度
            if rgb_image.dim() == 3:
                return nir.unsqueeze(0)
            else:
                return nir.unsqueeze(1)
        
        elif isinstance(rgb_image, np.ndarray):
            # 处理 numpy array
            if rgb_image.ndim == 3:
                r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
            else:
                raise ValueError(f"Expected 3 dim numpy array, got {rgb_image.ndim}")
            
            if self.method == 'physical':
                nir = 0.7 * r + 0.25 * g + 0.05 * b
            elif self.method == 'vegetation':
                nir = 0.6 * r + 0.4 * g - 0.1 * b
                nir = np.clip(nir, 0, 1)
            elif self.method == 'enhanced':
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                nir = 0.5 * r + 0.3 * g + 0.2 * luminance
            elif self.method == 'weighted':
                red_weight = 1 / (1 + np.exp(-(r - 0.5) * 4))
                nir = red_weight * (0.8 * r + 0.2 * g) + (1 - red_weight) * (0.5 * r + 0.3 * g + 0.2 * b)
            else:
                raise ValueError(f"Unknown NIR method: {self.method}")
            
            nir = nir * self.gain
            nir = np.clip(nir, 0, 1)
            
            return nir[:, :, np.newaxis]  # [H, W, 1]
        
        else:
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(rgb_image)}")


class CloudAugmentation:
    """
    云分割数据增强
    
    支持随机裁剪、翻转、亮度/对比度调整、高斯噪声等
    特别优化支持 NIR 通道的数据增强
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.use_nir = config.get('use_nir', True)
        self.nir_method = config.get('nir_method', 'physical')
        self.nir_gain = config.get('nir_gain', 1.1)
        
        # 如果启用NIR，初始化生成器
        if self.use_nir:
            self.nir_generator = NIRGenerator(method=self.nir_method, gain=self.nir_gain)
        else:
            self.nir_generator = None
        
        if self.enabled:
            print(f"[CloudAugmentation] Enabled with NIR={self.use_nir}")
    
    def __call__(self, image, mask, mode='train'):
        """
        应用数据增强
        
        Args:
            image: PIL.Image (RGB) or torch.Tensor [C, H, W] or numpy.ndarray [C, H, W]
            mask: PIL.Image (L) or torch.Tensor [H, W] or numpy.ndarray [H, W]
            mode: 'train' or 'val'
            
        Returns:
            image: torch.Tensor [C, H, W] (如果 use_nir=True, C=4)
            mask: torch.Tensor [H, W]
        """
        # 统一转换为 torch.Tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()  # [C, H, W]
        elif isinstance(image, Image.Image):
            image = T.ToTensor()(image)  # [3, H, W]
            
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()  # [H, W]
        elif isinstance(mask, Image.Image):
            mask = torch.from_numpy(np.array(mask)).long()
        
        if not self.enabled or mode != 'train':
            # 验证模式或无增强：直接添加 NIR（如果需要）
            if self.use_nir and self.nir_generator is not None and image.shape[0] == 3:
                nir = self.nir_generator(image)  # [1, H, W]
                image = torch.cat([image, nir], dim=0)  # [4, H, W]
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
                # 4通道: 只调整RGB，保留NIR
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
                # 4通道: 只调整RGB，保留NIR
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
        
        # 7. 添加 NIR 通道 (如果输入是3通道且需要4通道)
        if self.use_nir and self.nir_generator is not None and image.shape[0] == 3:
            nir = self.nir_generator(image)  # [1, H, W]
            image = torch.cat([image, nir], dim=0)  # [4, H, W]
        
        # 如果输入已经是4通道但use_nir=False，只保留RGB
        if not self.use_nir and image.shape[0] == 4:
            image = image[:3]  # 只保留RGB
        
        return image, mask


def create_mixed_dataloaders(config: dict, dev_run: bool = False):
    """
    创建多数据集混合数据加载器
    
    使用上层 Data 目录的 UnifiedCloudDataset
    支持:
    - 多数据集混合训练
    - 自动统一通道和位宽
    - 从训练集分层采样 val/test
    
    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg = config['data']
    train_cfg = config['training']
    
    # 统一配置
    target_channels = data_cfg.get('target_channels', 4)
    target_bit_depth = data_cfg.get('target_bit_depth', 16)
    use_nir = target_channels == 4
    
    # 数据增强
    aug_config = train_cfg.get('augmentation', {'enabled': False})
    aug_config['use_nir'] = use_nir
    aug_config['nir_method'] = data_cfg.get('nir_method', 'physical')
    aug_config['nir_gain'] = data_cfg.get('nir_gain', 1.1)
    transform = CloudAugmentation(aug_config)
    
    # 获取数据集配置
    datasets = data_cfg.get('datasets', [])
    if not datasets:
        raise ValueError("No datasets configured for multi-dataset mode")
    
    # 提取数据集名称列表
    dataset_names = [ds['name'].lower().replace('_', '') for ds in datasets]
    print(f"\n[MixedDataLoader] Using UnifiedCloudDataset from upper Data directory")
    print(f"  Datasets: {dataset_names}")
    print(f"  Target: {target_channels}ch, normalize={data_cfg.get('normalization', {}).get('method', 'auto')}")
    
    # 从上层 Data 目录导入 UnifiedCloudDataset
    data_dir = Path(__file__).parent.parent.parent / "Data"
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    from cloud_dataset_loader import UnifiedCloudDataset
    
    # 获取 val/test 分割配置
    split_cfg = data_cfg.get('val_test_split', {})
    train_ratio = split_cfg.get('train_ratio', 0.75)
    val_ratio = split_cfg.get('val_ratio', 0.15)
    test_ratio = split_cfg.get('test_ratio', 0.10)
    seed = split_cfg.get('seed', 42)
    
    # 统一数据集目录
    unified_data_dir = data_cfg.get('unified_data_dir', '../Data/Unified_Cloud_Dataset')
    
    # 确定归一化方法
    norm_method = data_cfg.get('normalization', {}).get('method', 'auto')
    # 映射配置方法到 UnifiedCloudDataset 支持的方法
    norm_mapping = {
        'percentile': 'percentile',
        'max': 'minmax',
        'zscore': 'standard',
        'sensor': 'reflectance',
        'clahe': 'percentile',  # CLAHE 需要单独处理，fallback 到 percentile
        'histeq': 'percentile',
    }
    normalize = norm_mapping.get(norm_method, 'auto')
    
    # 确定波段
    bands = ["B02", "B03", "B04", "B08"] if use_nir else ["B02", "B03", "B04"]
    
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
        random_seed=seed,
    )
    
    # Dev run: 限制数据集大小
    if dev_run:
        train_samples = min(64, len(train_dataset))
        val_samples = min(16, len(val_dataset))
        test_samples = min(16, len(test_dataset))
        
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


def _unified_dataset_collate(batch):
    """
    自定义 collate 函数，将 UnifiedCloudDataset 的 tuple 输出转换为 dict
    
    输入: list of (image, mask, info) tuples
          image: numpy ndarray or torch.Tensor (C, H, W)
          mask: numpy ndarray or torch.Tensor (H, W)
    输出: dict with batched tensors
    """
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
