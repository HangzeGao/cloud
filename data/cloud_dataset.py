"""
云分割数据集加载器
支持RICE2、HRC_WHU等云分割数据集
支持 RGB + NIR 四通道输入
支持业界标准的图像归一化方法
"""
import os
import glob
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
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


def get_data_loaders(config: dict, dev_run: bool = False):
    """
    创建标准的数据加载器（单数据集模式）
    
    Args:
        config: 配置字典，包含 data 和 training 配置
        dev_run: 是否为开发测试模式（限制数据量）
        
    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg = config['data']
    train_cfg = config['training']
    
    # 数据路径
    train_path = data_cfg.get('train_data_path')
    val_path = data_cfg.get('val_data_path')
    test_path = data_cfg.get('test_data_path')
    
    # 数据集参数
    use_nir = data_cfg.get('use_nir', True)
    target_size = data_cfg.get('target_size', (512, 512))
    nir_method = data_cfg.get('nir_method', 'physical')
    nir_gain = data_cfg.get('nir_gain', 1.1)
    
    # 数据增强配置
    aug_config = train_cfg.get('augmentation', {'enabled': False})
    aug_config['use_nir'] = use_nir
    aug_config['nir_method'] = nir_method
    aug_config['nir_gain'] = nir_gain
    transform = CloudAugmentation(aug_config)
    
    # 归一化器
    normalization_cfg = data_cfg.get('normalization', {})
    normalizer = ImageNormalizer(
        method=normalization_cfg.get('method', 'percentile'),
        bit_depth=data_cfg.get('target_bit_depth', 8)
    )
    
    # 创建数据集
    train_dataset = None
    val_dataset = None
    test_dataset = None
    
    if train_path:
        train_image_dir = os.path.join(train_path, 'images')
        train_mask_dir = os.path.join(train_path, 'masks')
        train_dataset = CloudSegmentationDataset(
            image_dir=train_image_dir,
            mask_dir=train_mask_dir,
            mode='train',
            transform=transform,
            target_size=target_size,
            use_nir=use_nir,
            nir_method=nir_method,
            nir_gain=nir_gain
        )
    
    if val_path:
        val_image_dir = os.path.join(val_path, 'images')
        val_mask_dir = os.path.join(val_path, 'masks')
        val_dataset = CloudSegmentationDataset(
            image_dir=val_image_dir,
            mask_dir=val_mask_dir,
            mode='val',
            transform=None,  # 验证集不使用数据增强
            target_size=target_size,
            use_nir=use_nir,
            nir_method=nir_method,
            nir_gain=nir_gain
        )
    
    if test_path:
        test_image_dir = os.path.join(test_path, 'images')
        test_mask_dir = os.path.join(test_path, 'masks')
        test_dataset = CloudSegmentationDataset(
            image_dir=test_image_dir,
            mask_dir=test_mask_dir,
            mode='test',
            transform=None,
            target_size=target_size,
            use_nir=use_nir,
            nir_method=nir_method,
            nir_gain=nir_gain
        )
    
    # Dev run: 限制数据集大小
    if dev_run and train_dataset:
        from torch.utils.data import Subset
        train_samples = min(64, len(train_dataset))
        train_dataset = Subset(train_dataset, range(train_samples))
        print(f"[Dev Run] Limited train: {train_samples} samples")
    
    # 创建 DataLoader
    num_workers = train_cfg.get('num_workers', 4)
    batch_size = train_cfg.get('batch_size', 8)
    
    # 4通道需要调整 batch_size
    if use_nir and batch_size > 4:
        adjusted = max(4, batch_size // 2)
        print(f"[DataLoader] Adjusted batch_size: {batch_size} -> {adjusted} (4-channel)")
        batch_size = adjusted
    
    train_loader = None
    val_loader = None
    test_loader = None
    
    if train_dataset:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    print(f"[DataLoader] Created: train={len(train_dataset) if train_dataset else 0}, "
          f"val={len(val_dataset) if val_dataset else 0}, "
          f"test={len(test_dataset) if test_dataset else 0}")
    
    return train_loader, val_loader, test_loader


class CloudSegmentationDataset(Dataset):
    """
    云分割数据集 (支持 RGB + NIR 四通道)
    
    支持的数据集格式：
    - 图像: .jpg, .png, .tif (RGB)
    - 标注: 同名的mask文件
    - NIR: 基于业界最优算法从 RGB 生成伪 NIR 通道
    
    Args:
        image_dir: 图像目录
        mask_dir: 标注目录
        mode: 'train', 'val', 或 'test'
        transform: 数据增强变换
        target_size: 目标尺寸 (H, W)
        use_nir: 是否生成 NIR 通道 (默认 True)
        nir_method: NIR 生成算法 ('physical', 'vegetation', 'enhanced', 'weighted')
        nir_gain: NIR 增益系数
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        mode: str = 'train',
        transform=None,
        target_size: tuple = (512, 512),
        use_nir: bool = True,
        nir_method: str = 'physical',
        nir_gain: float = 1.1
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.transform = transform
        self.target_size = target_size
        self.use_nir = use_nir
        
        # 初始化 NIR 生成器
        if use_nir:
            self.nir_generator = NIRGenerator(method=nir_method, gain=nir_gain)
            print(f"[CloudDataset] NIR enabled (method={nir_method}, gain={nir_gain})")
        else:
            self.nir_generator = None
        
        # 获取图像列表
        self.image_paths = []
        for ext in ['*.jpg', '*.png', '*.tif', '*.tiff']:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
        self.image_paths.sort()
        
        # 验证mask存在性
        self.valid_indices = []
        for i, img_path in enumerate(self.image_paths):
            img_name = os.path.basename(img_path)
            name_wo_ext = os.path.splitext(img_name)[0]
            
            # 尝试不同的mask命名方式
            mask_candidates = [
                os.path.join(mask_dir, img_name),
                os.path.join(mask_dir, name_wo_ext + '.png'),
                os.path.join(mask_dir, name_wo_ext + '.jpg'),
                os.path.join(mask_dir, name_wo_ext + '.tif'),
            ]
            
            mask_exists = any(os.path.exists(m) for m in mask_candidates)
            if mask_exists:
                self.valid_indices.append(i)
        
        self.image_paths = [self.image_paths[i] for i in self.valid_indices]
        
        print(f"[CloudDataset] {mode}: {len(self.image_paths)} valid samples")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 加载mask
        img_name = os.path.basename(img_path)
        name_wo_ext = os.path.splitext(img_name)[0]
        
        mask_candidates = [
            os.path.join(self.mask_dir, img_name),
            os.path.join(self.mask_dir, name_wo_ext + '.png'),
            os.path.join(self.mask_dir, name_wo_ext + '.jpg'),
            os.path.join(self.mask_dir, name_wo_ext + '.tif'),
        ]
        
        mask = None
        for mask_path in mask_candidates:
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                break
        
        if mask is None:
            raise FileNotFoundError(f"Mask not found for {img_path}")
        
        # 确保mask是二值化的（云/背景）
        mask_np = np.array(mask)
        # 将非零值设为1（云），零值设为0（背景）
        mask_np = (mask_np > 127).astype(np.uint8)
        mask = Image.fromarray(mask_np)
        
        # 数据增强
        if self.transform is not None:
            image, mask = self.transform(image, mask, self.mode)
        else:
            # 默认变换
            image = T.ToTensor()(image)  # [3, H, W]
            mask = torch.from_numpy(np.array(mask)).long()
        
        # Note: NIR 通道由 CloudAugmentation 统一处理（在数据增强后添加）
        # 如果 transform 为 None（无数据增强），则在下方处理
        if self.transform is None and self.use_nir and self.nir_generator is not None:
            nir = self.nir_generator(image)  # [1, H, W]
            image = torch.cat([image, nir], dim=0)  # [4, H, W]

        return {
            'image': image,
            'mask': mask,
            'filename': img_name
        }


class InferenceDataset(Dataset):
    """
    推理专用数据集 - 支持任意尺寸图像 (RGB+NIR 四通道)
    """
    
    def __init__(self, image_dir: str, use_nir: bool = True, nir_method: str = 'physical', nir_gain: float = 1.1):
        self.image_paths = []
        for ext in ['*.jpg', '*.png', '*.tif', '*.tiff']:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        self.image_paths.sort()
        
        self.use_nir = use_nir
        if use_nir:
            self.nir_generator = NIRGenerator(method=nir_method, gain=nir_gain)
            print(f"[InferenceDataset] NIR enabled (method={nir_method}, gain={nir_gain})")
        else:
            self.nir_generator = None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        # 转换为Tensor
        image = T.ToTensor()(image)  # [3, H, W]
        
        # 生成 NIR 通道
        if self.use_nir and self.nir_generator is not None:
            nir = self.nir_generator(image)  # [1, H, W]
            image = torch.cat([image, nir], dim=0)  # [4, H, W]
        
        return {
            'image': image,
            'filename': os.path.basename(img_path),
            'original_size': original_size
        }


class UnifiedDataset(Dataset):
    """
    统一数据集 - 支持多数据集混合训练，自动统一通道和位宽
    
    功能:
    - 混合多个数据集（均作为训练集）
    - 自动统一通道数（目标：4通道 RGB+NIR）
    - 自动统一位宽（目标：16bit）
    - 支持 val/test 从训练集采样
    
    数据统一处理:
    1. 通道统一：
       - 3通道数据：通过 NIR 生成算法添加第4通道
       - 4通道数据：保留原生 NIR
    
    2. 位宽统一：
       - 8bit 数据：×255，转为 16bit 范围
       - 16bit 数据：直接使用
       - 所有数据最终归一化到 [0, 1] float32
    
    Example:
        datasets = [
            {'name': 'RICE2', 'path': '../Data/RICE2', 'type': 'standard', 'bit_depth': 8},
            {'name': 'cloud_cover', 'path': '../Data/cloud_cover', 'type': 'cloud_cover', 'bit_depth': 16}
        ]
        
        dataset = UnifiedDataset(
            datasets=datasets,
            target_channels=4,
            target_bit_depth=16,
            transform=transform
        )
    """
    
    def __init__(
        self,
        datasets: list,
        target_channels: int = 4,
        target_bit_depth: int = 16,
        transform=None,
        mode: str = 'train',
        normalizer=None
    ):
        """
        Args:
            datasets: 数据集配置列表，每个元素为 dict
                     {'name': str, 'path': str, 'type': str, 'bit_depth': int, 'use_all_bands': bool}
            target_channels: 目标通道数 (3 或 4)
            target_bit_depth: 目标位深 (8 或 16)
            transform: 数据增强
            mode: 'train', 'val', 或 'test'
            normalizer: 图像归一化器 (ImageNormalizer 实例)
        """
        self.datasets = datasets
        self.target_channels = target_channels
        self.target_bit_depth = target_bit_depth
        self.transform = transform
        self.mode = mode
        self.use_nir = target_channels == 4
        
        # 初始化图像归一化器
        if normalizer is None:
            # 默认使用百分位数归一化（业界推荐）
            self.normalizer = ImageNormalizer(method='percentile', bit_depth=target_bit_depth)
        else:
            self.normalizer = normalizer
        
        # 初始化 NIR 生成器（用于 3 通道数据集）
        if self.use_nir:
            self.nir_generator = NIRGenerator(method='physical', gain=1.1)
        else:
            self.nir_generator = None
        
        # 收集所有样本
        self.samples = []  # [(dataset_name, dataset_type, bit_depth, image_path, mask_path), ...]
        
        for ds_cfg in datasets:
            name = ds_cfg['name']
            path = ds_cfg['path']
            ds_type = ds_cfg.get('type', 'standard')
            bit_depth = ds_cfg.get('bit_depth', 8)
            use_all_bands = ds_cfg.get('use_all_bands', False)
            
            samples = self._collect_dataset_samples(name, path, ds_type, bit_depth, use_all_bands)
            self.samples.extend(samples)
            
            print(f"[UnifiedDataset] {name} ({ds_type}, {bit_depth}bit): {len(samples)} samples")
        
        print(f"[UnifiedDataset] Total: {len(self.samples)} samples, "
              f"{target_channels}ch, {target_bit_depth}bit, mode={mode}")
    
    def _collect_dataset_samples(self, name, path, ds_type, bit_depth, use_all_bands):
        """
        收集单个数据集的样本
        
        支持两种数据结构:
        1. standard 类型:
           - images/image.jpg, masks/mask.png
           - train/images/image.jpg, train/masks/mask.png
        
        2. cloud_cover 类型 (Sentinel-2 格式):
           - train_features/<sample_id>/B02.tif (Blue)
           - train_features/<sample_id>/B03.tif (Green)
           - train_features/<sample_id>/B04.tif (Red)
           - train_features/<sample_id>/B08.tif (NIR)
           - train_labels/<sample_id>.tif
        """
        samples = []
        
        # cloud_cover 类型特殊处理
        if ds_type == 'cloud_cover':
            return self._collect_cloud_cover_samples(name, path, bit_depth, use_all_bands)
        
        # standard 类型的标准目录结构
        possible_dirs = [
            (os.path.join(path, 'images'), os.path.join(path, 'masks')),
            (os.path.join(path, 'train', 'images'), os.path.join(path, 'train', 'masks')),
            (os.path.join(path, 'val', 'images'), os.path.join(path, 'val', 'masks')),
        ]
        
        image_dir = None
        mask_dir = None
        
        for img_dir, msk_dir in possible_dirs:
            if os.path.exists(img_dir) and os.path.exists(msk_dir):
                image_dir = img_dir
                mask_dir = msk_dir
                break
        
        if image_dir is None:
            print(f"Warning: Could not find valid dirs for {name} at {path}")
            return []
        
        # 获取所有图像
        image_paths = []
        for ext in ['*.jpg', '*.png', '*.tif', '*.tiff']:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
        image_paths.sort()
        
        # 验证并收集
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            name_wo_ext = os.path.splitext(img_name)[0]
            
            mask_candidates = [
                os.path.join(mask_dir, img_name),
                os.path.join(mask_dir, name_wo_ext + '.png'),
                os.path.join(mask_dir, name_wo_ext + '.jpg'),
                os.path.join(mask_dir, name_wo_ext + '.tif'),
            ]
            
            for mask_path in mask_candidates:
                if os.path.exists(mask_path):
                    samples.append((name, ds_type, bit_depth, use_all_bands, img_path, mask_path))
                    break
        
        return samples
    
    def _collect_cloud_cover_samples(self, name, path, bit_depth, use_all_bands):
        """
        收集 cloud_cover 类型的样本（Sentinel-2 格式）
        
        结构:
        - train_features/<sample_id>/B02.tif (Blue)
        - train_features/<sample_id>/B03.tif (Green)
        - train_features/<sample_id>/B04.tif (Red)
        - train_features/<sample_id>/B08.tif (NIR)
        - train_labels/<sample_id>.tif
        """
        samples = []
        
        # cloud_cover 类型的标准路径
        feature_dir = os.path.join(path, 'train_features')
        label_dir = os.path.join(path, 'train_labels')
        
        if not os.path.exists(feature_dir) or not os.path.exists(label_dir):
            print(f"Warning: cloud_cover dataset requires 'train_features' and 'train_labels' dirs at {path}")
            return []
        
        # 获取所有样本ID（子目录名）
        sample_ids = []
        for item in os.listdir(feature_dir):
            item_path = os.path.join(feature_dir, item)
            if os.path.isdir(item_path):
                # 检查是否有必要的波段文件
                has_all_bands = all(
                    os.path.exists(os.path.join(item_path, f'band_{band}.tif')) or
                    os.path.exists(os.path.join(item_path, f'B{band}.tif'))
                    for band in ['02', '03', '04', '08']
                )
                if has_all_bands:
                    sample_ids.append(item)
        
        sample_ids.sort()
        
        # 验证并收集
        for sample_id in sample_ids:
            feature_path = os.path.join(feature_dir, sample_id)
            
            # 检查标签文件
            label_candidates = [
                os.path.join(label_dir, f'{sample_id}.tif'),
                os.path.join(label_dir, f'{sample_id}.png'),
                os.path.join(label_dir, f'{sample_id}.jpg'),
            ]
            
            label_path = None
            for cand in label_candidates:
                if os.path.exists(cand):
                    label_path = cand
                    break
            
            if label_path:
                # 对于 cloud_cover 类型，img_path 是 feature 目录路径
                # 实际的波段文件会在 _load_cloud_cover_image 中读取
                samples.append((name, 'cloud_cover', bit_depth, use_all_bands, feature_path, label_path))
        
        return samples
    
    def _load_and_unify_image(self, ds_type, bit_depth, use_all_bands, img_path):
        """
        加载图像并统一通道和位宽，使用业界标准的归一化方法
        
        支持任意位宽（8bit, 13bit, 16bit等），统一归一化到 [0, 1] 范围
        通道补齐到统一的 4 通道（RGB+NIR）
        
        Args:
            ds_type: 数据集类型 ('cloud_cover' 或 'standard')
            bit_depth: 位宽 (8, 13, 16等)
            use_all_bands: 是否使用所有波段（cloud_cover数据集）
            img_path: 图像路径（standard类型是文件路径，cloud_cover类型是目录路径）
            
        Returns:
            image: [C, H, W] tensor in [0, 1], dtype float32
        """
        # 临时设置归一化器的位宽，用于计算最大值
        original_bit_depth = self.normalizer.bit_depth
        self.normalizer.set_bit_depth(bit_depth)
        
        try:
            if ds_type == 'cloud_cover':
                # cloud_cover 类型: img_path 是特征目录，包含 B02,B03,B04,B08
                img = self._load_cloud_cover_image(img_path, bit_depth, use_all_bands)
            else:
                # 标准 3 通道数据 (8bit)
                img = Image.open(img_path).convert('RGB')
                img = np.array(img, dtype=np.float32)
                
                # 8bit 数据 max_val = 255
                max_val = (1 << bit_depth) - 1
                img = img / max_val  # 先缩放到 [0, 1]
                
                # 使用业界标准的归一化方法
                img = self.normalizer(img, per_image=True)
                
                # 如果需要 4 通道，生成 NIR
                if self.target_channels == 4:
                    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                    # 使用 NIRGenerator 的物理模型生成 NIR
                    nir = 0.7 * r + 0.25 * g + 0.05 * b
                    img = np.concatenate([img, nir[:, :, np.newaxis]], axis=-1)
            
            # 转为 tensor [C, H, W]
            img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            
            return img_tensor
        
        finally:
            # 恢复原来的位宽设置
            self.normalizer.set_bit_depth(original_bit_depth)
    
    def _load_cloud_cover_image(self, feature_dir, bit_depth, use_all_bands):
        """
        加载 cloud_cover 类型的图像（Sentinel-2 格式）
        
        从 B02(Blue), B03(Green), B04(Red), B08(NIR) 合成 4 通道图像
        
        Args:
            feature_dir: 特征目录路径 (如 train_features/<sample_id>/)
            bit_depth: 位宽
            use_all_bands: 是否使用原生 NIR (B08)
            
        Returns:
            img: [H, W, C] numpy array in [0, 1]
        """
        # 波段文件映射（支持两种命名格式）
        band_files = {
            'B02': ['B02.tif', 'band_02.tif', 'B02.jp2', 'blue.tif'],
            'B03': ['B03.tif', 'band_03.tif', 'B03.jp2', 'green.tif'],
            'B04': ['B04.tif', 'band_04.tif', 'B04.jp2', 'red.tif'],
            'B08': ['B08.tif', 'band_08.tif', 'B08.jp2', 'nir.tif'],
        }
        
        # 加载各波段
        bands = {}
        for band_name, candidates in band_files.items():
            band_path = None
            for cand in candidates:
                path = os.path.join(feature_dir, cand)
                if os.path.exists(path):
                    band_path = path
                    break
            
            if band_path is None:
                raise FileNotFoundError(f"Band {band_name} not found in {feature_dir}")
            
            # 读取波段图像
            try:
                band_img = np.array(Image.open(band_path))
            except Exception as e:
                raise RuntimeError(f"Failed to load {band_path}: {e}")
            
            bands[band_name] = band_img
        
        # 确保所有波段尺寸一致
        target_shape = bands['B02'].shape
        for band_name, band_img in bands.items():
            if band_img.shape != target_shape:
                raise ValueError(f"Band {band_name} shape {band_img.shape} doesn't match {target_shape}")
        
        # 合成 RGB+NIR (按照 Sentinel-2 波段定义)
        # B04=Red, B03=Green, B02=Blue, B08=NIR
        max_val = (1 << bit_depth) - 1
        
        red = bands['B04'].astype(np.float32) / max_val
        green = bands['B03'].astype(np.float32) / max_val
        blue = bands['B02'].astype(np.float32) / max_val
        nir_native = bands['B08'].astype(np.float32) / max_val
        
        # 合成 RGB
        rgb = np.stack([red, green, blue], axis=-1)  # [H, W, 3]
        
        # 应用归一化
        rgb = self.normalizer(rgb, per_image=True)
        
        # 处理 NIR 通道
        if self.target_channels == 4:
            if use_all_bands:
                # 使用原生 NIR (B08)
                nir = self.normalizer(nir_native, per_image=True)
                nir = nir[:, :, np.newaxis]  # [H, W, 1]
            else:
                # 从归一化后的 RGB 生成伪 NIR
                r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
                nir = 0.7 * r + 0.25 * g + 0.05 * b
                nir = nir[:, :, np.newaxis]  # [H, W, 1]
            
            # 合并 RGB + NIR
            img = np.concatenate([rgb, nir], axis=-1)  # [H, W, 4]
        else:
            # 只使用 RGB
            img = rgb  # [H, W, 3]
        
        return img
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        dataset_name, ds_type, bit_depth, use_all_bands, img_path, mask_path = self.samples[idx]
        
        # 加载并统一图像
        image = self._load_and_unify_image(ds_type, bit_depth, use_all_bands, img_path)
        
        # 加载 mask
        mask_pil = Image.open(mask_path).convert('L')
        mask_np = np.array(mask_pil)
        # 二值化：适应不同格式的 mask（0-255 或 0-1）
        # 如果最大值 <= 1，说明已经是二值；否则使用 127 作为阈值
        if mask_np.max() <= 1:
            mask_np = mask_np.astype(np.uint8)
        else:
            mask_np = (mask_np > 127).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_np).long()
        
        # 默认使用 tensor
        mask = mask_tensor
        
        # 应用数据增强（如果需要）
        if self.transform is not None:
            # 转回 PIL 用于变换（只取RGB部分）
            img_rgb = image[:3].permute(1, 2, 0).numpy()
            img_rgb = (img_rgb * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_rgb)
            mask_pil = Image.fromarray(mask_np)
            
            # 变换（返回3通道 + NIR）
            img_transformed, mask_transformed = self.transform(img_pil, mask_pil, self.mode)
            
            # 如果 transform 生成了4通道，使用它；否则手动添加保存的NIR
            if self.use_nir and image.shape[0] == 4 and img_transformed.shape[0] == 3:
                # resize 保存的 NIR
                nir = TF.resize(image[3:4], img_transformed.shape[1:])
                image = torch.cat([img_transformed, nir], dim=0)
            else:
                image = img_transformed
            
            mask = mask_transformed
        
        return {
            'image': image,
            'mask': mask,
            'filename': os.path.basename(img_path),
            'dataset': dataset_name,
            'source_type': ds_type,
            'original_bit_depth': bit_depth
        }


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
    import sys
    from pathlib import Path
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
        from torch.utils.data import Subset
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
