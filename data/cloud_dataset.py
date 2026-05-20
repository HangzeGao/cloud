"""
云分割数据集加载器
支持RICE2、HRC_WHU等云分割数据集
支持 RGB + NIR 四通道输入
"""
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class NIRGenerator:
    """
    NIR（近红外）通道生成器
    
    基于业界最优算法，从 RGB 图像生成伪 NIR 通道。
    利用云在 NIR 波段的高反射率特性（通常比红波段高 10-20%）。
    
    支持算法:
    - 'physical': 基于云光谱物理特性的线性组合 (推荐)
    - 'vegetation': 基于植被指数启发的方法
    - 'enhanced': 增强型，结合对比度和边缘信息
    - 'weighted': 可配置加权组合
    
    Reference:
    - 云在 NIR (0.7-1.0μm) 具有高反射率，接近红波段
    - 使用经验系数: NIR = 0.7*R + 0.25*G + 0.05*B + offset
    """
    
    def __init__(self, method: str = 'physical', gain: float = 1.1, offset: float = 0.05):
        """
        Args:
            method: NIR生成算法 ('physical', 'vegetation', 'enhanced', 'weighted')
            gain: NIR 增益系数 (云在 NIR 反射率通常比 RGB 高 10-30%)
            offset: 基础偏移量
        """
        self.method = method
        self.gain = gain
        self.offset = offset
        
        # 预定义算法系数
        self.coefficients = {
            'physical': {'r': 0.70, 'g': 0.25, 'b': 0.05},  # 基于云光谱特性
            'vegetation': {'r': 0.60, 'g': 0.35, 'b': 0.05},  # 植被指数启发
            'enhanced': {'r': 0.65, 'g': 0.30, 'b': 0.05},   # 增强对比度
            'weighted': {'r': 0.65, 'g': 0.25, 'b': 0.10},   # 平衡加权
        }
    
    def __call__(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        从 RGB 图像生成 NIR 通道
        
        Args:
            rgb_image: [3, H, W] 的 RGB 张量，值域 [0, 1]
            
        Returns:
            nir: [1, H, W] 的 NIR 张量，值域 [0, 1]
        """
        r, g, b = rgb_image[0], rgb_image[1], rgb_image[2]
        
        if self.method == 'physical':
            # 基于云物理光谱特性
            # 云在 NIR (0.76-0.90μm) 具有高反射率，与红波段强相关
            nir = self.coefficients['physical']['r'] * r + \
                  self.coefficients['physical']['g'] * g + \
                  self.coefficients['physical']['b'] * b
            
        elif self.method == 'vegetation':
            # 植被指数启发: NIR ≈ (R + G) / 2 + α * (G - R)
            # 利用绿波段与红波段的差异来估算近红外反射率
            base = (r + g) / 2
            diff = torch.abs(g - r) * 0.3
            nir = base + diff
            
        elif self.method == 'enhanced':
            # 增强型: 结合局部对比度
            nir = self.coefficients['enhanced']['r'] * r + \
                  self.coefficients['enhanced']['g'] * g + \
                  self.coefficients['enhanced']['b'] * b
            # 增加局部方差增强云的边缘
            local_mean = torch.nn.functional.avg_pool2d(
                nir.unsqueeze(0).unsqueeze(0), 
                kernel_size=5, stride=1, padding=2
            ).squeeze()
            variance = (nir - local_mean).abs() * 0.2
            nir = nir + variance
            
        elif self.method == 'weighted':
            # 简单加权
            nir = self.coefficients['weighted']['r'] * r + \
                  self.coefficients['weighted']['g'] * g + \
                  self.coefficients['weighted']['b'] * b
        
        else:
            raise ValueError(f"Unknown NIR method: {self.method}")
        
        # 应用增益和偏移
        nir = nir * self.gain + self.offset
        
        # 裁剪到有效范围
        nir = torch.clamp(nir, 0.0, 1.0)
        
        return nir.unsqueeze(0)  # [1, H, W]


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


class CloudAugmentation:
    """
    云分割数据增强 (支持 RGB+NIR 四通道)
    包含空间变换、颜色变换、噪声等
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get('enabled', True)
        
        if not self.enabled:
            return
        
        self.crop_size = config.get('random_crop_size', [512, 512])
        self.hflip_prob = config.get('horizontal_flip', 0.5)
        self.vflip_prob = config.get('vertical_flip', 0.5)
        self.brightness = config.get('brightness', 0.2)
        self.contrast = config.get('contrast', 0.2)
        self.noise_std = config.get('gaussian_noise', 0.01)
        self.use_nir = config.get('use_nir', True)
        
        # NIR 生成器（用于 val/test 模式）
        if self.use_nir:
            nir_method = config.get('nir_method', 'physical')
            nir_gain = config.get('nir_gain', 1.1)
            self.nir_generator = NIRGenerator(method=nir_method, gain=nir_gain)
        else:
            self.nir_generator = None
    
    def __call__(self, image: Image.Image, mask: Image.Image, mode: str):
        """
        Args:
            image: PIL Image (RGB)
            mask: PIL Image (L)
            mode: 'train' 或 'val'
            
        Returns:
            image: Tensor [3, H, W] 或 [4, H, W] (RGB or RGB+NIR)
            mask: Tensor [H, W]
        """
        if not self.enabled or mode != 'train':
            # 验证模式：调整尺寸并进行必要的预处理
            image = TF.resize(image, self.crop_size)
            mask = TF.resize(mask, self.crop_size, interpolation=TF.InterpolationMode.NEAREST)
            image = T.ToTensor()(image)  # [3, H, W]
            mask = torch.from_numpy(np.array(mask)).long()
            
            # 生成 NIR 通道
            if self.use_nir and self.nir_generator is not None:
                nir = self.nir_generator(image)  # [1, H, W]
                image = torch.cat([image, nir], dim=0)  # [4, H, W]
            
            return image, mask
        
        # 训练模式：应用数据增强
        
        # 1. 随机裁剪
        i, j, h, w = T.RandomCrop.get_params(
            image, 
            output_size=(self.crop_size[0], self.crop_size[1])
        )
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        
        # 2. 随机水平翻转
        if torch.rand(1) < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # 3. 随机垂直翻转
        if torch.rand(1) < self.vflip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # 4. 颜色抖动（仅应用于图像 RGB 通道）
        if self.brightness > 0:
            brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.brightness
            image = TF.adjust_brightness(image, brightness_factor)
        
        if self.contrast > 0:
            contrast_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.contrast
            image = TF.adjust_contrast(image, contrast_factor)
        
        # 5. 转换为Tensor
        image = T.ToTensor()(image)  # [3, H, W]
        mask = torch.from_numpy(np.array(mask)).long()
        
        # 6. 添加高斯噪声（模拟传感器噪声，仅应用于 RGB）
        if self.noise_std > 0:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
            image = torch.clamp(image, 0, 1)
        
        # 7. 生成 NIR 通道（基于增强后的 RGB）
        if self.use_nir and self.nir_generator is not None:
            nir = self.nir_generator(image)  # [1, H, W]
            image = torch.cat([image, nir], dim=0)  # [4, H, W]
        
        return image, mask


def get_data_loaders(config: dict, dev_run: bool = False):
    """
    创建数据加载器 (支持 RGB+NIR 四通道和原生 4 通道高比特数据集)
    
    Args:
        config: 配置字典
        dev_run: 如果为True，只使用少量样本进行快速开发测试
        
    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg = config['data']
    train_cfg = config['training']
    
    # 检测数据集类型
    dataset_type = data_cfg.get('dataset_type', 'standard')
    use_native_4ch = data_cfg.get('native_4channel', False)
    
    # NIR 配置
    use_nir = data_cfg.get('use_nir', True)
    nir_method = data_cfg.get('nir_method', 'physical')
    nir_gain = data_cfg.get('nir_gain', 1.1)
    
    if use_nir:
        print(f"[DataLoader] NIR enabled: method={nir_method}, gain={nir_gain}")
    
    # 数据增强配置
    aug_config = train_cfg.get('augmentation', {'enabled': False})
    aug_config['use_nir'] = use_nir
    aug_config['nir_method'] = nir_method
    aug_config['nir_gain'] = nir_gain
    transform = CloudAugmentation(aug_config)
    
    # 根据数据集类型创建数据集
    if dataset_type == 'cloud_cover' or use_native_4ch:
        # 原生 4 通道高比特数据集
        print(f"[DataLoader] Using native 4-channel dataset: {dataset_type}")
        
        train_dataset = CloudCoverDataset(
            data_root=data_cfg['train_data_path'],
            split='train',
            transform=transform,
            bit_depth=data_cfg.get('bit_depth', None),
            use_all_bands=data_cfg.get('use_all_bands', True)
        )
        
        val_dataset = CloudCoverDataset(
            data_root=data_cfg['val_data_path'],
            split='val',
            transform=transform,
            bit_depth=data_cfg.get('bit_depth', None),
            use_all_bands=data_cfg.get('use_all_bands', True)
        )
        
        test_dataset = CloudCoverDataset(
            data_root=data_cfg['test_data_path'],
            split='test',
            transform=transform,
            bit_depth=data_cfg.get('bit_depth', None),
            use_all_bands=data_cfg.get('use_all_bands', True)
        )
    else:
        # 标准 3 通道数据集（RGB+NIR 生成）
        train_dataset = CloudSegmentationDataset(
            image_dir=os.path.join(data_cfg['train_data_path'], 'images'),
            mask_dir=os.path.join(data_cfg['train_data_path'], 'masks'),
            mode='train',
            transform=transform,
            use_nir=use_nir,
            nir_method=nir_method,
            nir_gain=nir_gain
        )
        
        val_dataset = CloudSegmentationDataset(
            image_dir=os.path.join(data_cfg['val_data_path'], 'images'),
            mask_dir=os.path.join(data_cfg['val_data_path'], 'masks'),
            mode='val',
            transform=transform,
            use_nir=use_nir,
            nir_method=nir_method,
            nir_gain=nir_gain
        )
        
        test_dataset = CloudSegmentationDataset(
            image_dir=os.path.join(data_cfg['test_data_path'], 'images'),
            mask_dir=os.path.join(data_cfg['test_data_path'], 'masks'),
            mode='test',
            transform=transform,
            use_nir=use_nir,
            nir_method=nir_method,
            nir_gain=nir_gain
        )
    
    # Dev run: limit dataset size for quick testing
    if dev_run:
        from torch.utils.data import Subset
        train_samples = min(64, len(train_dataset))
        val_samples = min(16, len(val_dataset))
        test_samples = min(16, len(test_dataset))
        train_dataset = Subset(train_dataset, range(train_samples))
        val_dataset = Subset(val_dataset, range(val_samples))
        test_dataset = Subset(test_dataset, range(test_samples))
        print(f"[Dev Run] Limited datasets: train={train_samples}, val={val_samples}, test={test_samples}")

    # DataLoader
    num_workers = train_cfg.get('num_workers', 4)
    batch_size = train_cfg.get('batch_size', 8)
    
    # 根据通道数调整 batch_size（4通道比3通道占用更多显存）
    if use_nir:
        # 如果 batch_size > 4，建议减小以适应更大的通道数
        if batch_size > 4:
            adjusted_batch_size = max(4, batch_size // 2)
            print(f"[DataLoader] Adjusted batch_size: {batch_size} -> {adjusted_batch_size} (4-channel)")
            batch_size = adjusted_batch_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 测试时使用batch_size=1以支持任意尺寸
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


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


class CloudCoverDataset(Dataset):
    """
    原生 4 通道 (RGB+NIR) 高比特位宽云分割数据集
    
    专为 cloud_cover_detection 数据集设计:
    - 支持 4 通道原生图像 (R, G, B, NIR)
    - 支持 16-bit/32-bit 高比特位宽
    - 自动归一化到 [0, 1] 范围
    - 支持 TIFF/PNG 等格式
    
    数据集结构:
    cloud_cover_detection/
    ├── train/
    │   ├── images/     # 4通道高比特图像
    │   └── masks/      # 单通道标注
    ├── val/
    └── test/
    
    Args:
        data_root: 数据集根目录 (如 '../Data/cloud_cover_detection')
        split: 'train', 'val', 或 'test'
        transform: 数据增强变换
        target_size: 目标尺寸
        bit_depth: 输入图像位深 (16, 32, None=自动检测)
        use_all_bands: 是否使用所有4通道 (False则只用RGB)
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform=None,
        target_size: tuple = (512, 512),
        bit_depth: int = None,
        use_all_bands: bool = True
    ):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.bit_depth = bit_depth
        self.use_all_bands = use_all_bands
        
        # 图像和标注目录
        self.image_dir = os.path.join(data_root, split, 'images')
        self.mask_dir = os.path.join(data_root, split, 'masks')
        
        # 获取图像列表
        self.image_paths = []
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg']:
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext)))
        self.image_paths.sort()
        
        # 验证 mask 存在性
        self.valid_indices = []
        for i, img_path in enumerate(self.image_paths):
            img_name = os.path.basename(img_path)
            name_wo_ext = os.path.splitext(img_name)[0]
            
            mask_candidates = [
                os.path.join(self.mask_dir, img_name),
                os.path.join(self.mask_dir, name_wo_ext + '.png'),
                os.path.join(self.mask_dir, name_wo_ext + '.tif'),
            ]
            
            if any(os.path.exists(m) for m in mask_candidates):
                self.valid_indices.append(i)
        
        self.image_paths = [self.image_paths[i] for i in self.valid_indices]
        
        print(f"[CloudCoverDataset] {split}: {len(self.image_paths)} valid samples")
        if use_all_bands:
            print(f"[CloudCoverDataset] Using 4 channels (R+G+B+NIR)")
        else:
            print(f"[CloudCoverDataset] Using 3 channels (R+G+B), skipping native NIR")
    
    def __len__(self):
        return len(self.image_paths)
    
    def _load_multichannel_image(self, path: str) -> np.ndarray:
        """
        加载多通道高比特位宽图像
        
        Returns:
            image: [H, W, C] numpy array, float32 in [0, 1]
        """
        # 使用 imageio 或 tifffile 加载高比特图像
        try:
            import tifffile
            img = tifffile.imread(path)
        except ImportError:
            # 降级使用 PIL
            img = np.array(Image.open(path))
        
        # 处理不同维度格式
        if img.ndim == 2:
            # 单通道，复制为4通道
            img = np.stack([img] * 4, axis=-1)
        elif img.ndim == 3:
            if img.shape[0] <= 4 and img.shape[0] < img.shape[-1]:
                # [C, H, W] 格式，转为 [H, W, C]
                img = np.transpose(img, (1, 2, 0))
        
        # 确保至少3通道
        if img.shape[-1] < 3:
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
        
        # 检测位深并归一化
        if self.bit_depth is not None:
            max_val = (1 << self.bit_depth) - 1
        else:
            # 自动检测
            if img.dtype == np.uint8:
                max_val = 255
            elif img.dtype == np.uint16:
                max_val = 65535
            elif img.dtype == np.float32 or img.dtype == np.float64:
                max_val = 1.0
                if img.max() > 1:
                    max_val = img.max()
            else:
                max_val = img.max()
        
        # 归一化到 [0, 1]
        img = img.astype(np.float32) / max_val
        
        # 确保4通道 (RGB+NIR)
        if self.use_all_bands and img.shape[-1] >= 4:
            img = img[:, :, :4]  # 取前4通道
        elif self.use_all_bands and img.shape[-1] == 3:
            # 只有3通道，需要生成NIR
            print(f"Warning: {path} has only 3 channels, generating pseudo-NIR")
            nir = 0.7 * img[:, :, 0] + 0.25 * img[:, :, 1] + 0.05 * img[:, :, 2]
            nir = nir[:, :, np.newaxis]
            img = np.concatenate([img, nir], axis=-1)
        else:
            # 只使用RGB
            img = img[:, :, :3]
        
        return img
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        name_wo_ext = os.path.splitext(img_name)[0]
        
        # 加载多通道图像 [H, W, C]
        image_np = self._load_multichannel_image(img_path)
        
        # 加载 mask
        mask_candidates = [
            os.path.join(self.mask_dir, img_name),
            os.path.join(self.mask_dir, name_wo_ext + '.png'),
            os.path.join(self.mask_dir, name_wo_ext + '.tif'),
        ]
        
        mask = None
        for mask_path in mask_candidates:
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                break
        
        if mask is None:
            raise FileNotFoundError(f"Mask not found for {img_path}")
        
        # 二值化 mask
        mask_np = np.array(mask)
        mask_np = (mask_np > 127).astype(np.uint8)
        
        # 转为 PIL Image 以使用 transforms
        image_pil = Image.fromarray((image_np[:, :, :3] * 255).astype(np.uint8))
        mask_pil = Image.fromarray(mask_np)
        
        # 数据增强
        if self.transform is not None:
            # 注意：transform 返回的是 [C, H, W] tensor
            if self.use_all_bands and image_np.shape[-1] == 4:
                # 4通道情况：先转换RGB，然后合并原生NIR
                image_rgb_pil = Image.fromarray((image_np[:, :, :3] * 255).astype(np.uint8))
                image_tensor, mask_tensor = self.transform(image_rgb_pil, mask_pil, self.split)
                
                # 如果 transform 只返回了3通道，我们需要添加原生NIR
                if image_tensor.shape[0] == 3:
                    # 加载原生NIR通道
                    nir_native = torch.from_numpy(image_np[:, :, 3]).float()
                    nir_native = TF.resize(nir_native.unsqueeze(0), self.target_size)
                    image_tensor = torch.cat([image_tensor, nir_native], dim=0)
            else:
                # 3通道情况
                image_tensor, mask_tensor = self.transform(image_pil, mask_pil, self.split)
        else:
            # 默认变换
            image_pil = image_pil.resize(self.target_size)
            mask_pil = mask_pil.resize(self.target_size, Image.NEAREST)
            
            image_tensor = T.ToTensor()(image_pil)  # [3, H, W]
            
            # 添加原生NIR（如果是4通道模式）
            if self.use_all_bands and image_np.shape[-1] == 4:
                nir_resized = TF.resize(
                    Image.fromarray((image_np[:, :, 3] * 255).astype(np.uint8)),
                    self.target_size
                )
                nir_tensor = T.ToTensor()(nir_resized)  # [1, H, W]
                image_tensor = torch.cat([image_tensor, nir_tensor], dim=0)
            
            mask_tensor = torch.from_numpy(np.array(mask_pil)).long()
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'filename': img_name,
            'native_nir': self.use_all_bands and image_np.shape[-1] == 4
        }


class MultiDatasetSampler:
    """
    多数据集采样器 - 用于快速验证模型在不同数据集的效果
    
    从多个数据集中分别采样固定数量的样本，组合成一个验证集
    
    Example:
        sampler = MultiDatasetSampler({
            'RICE2': '../Data/RICE2',
            'HRC_WHU': '../Data/HRC_WHU', 
            'cloud_cover': '../Data/cloud_cover_detection'
        }, samples_per_dataset=20)
        
        val_loader = sampler.get_validation_loader(batch_size=4)
    """
    
    def __init__(
        self,
        dataset_paths: dict,
        samples_per_dataset: int = 20,
        target_size: tuple = (512, 512),
        use_nir: bool = True,
        transform=None
    ):
        """
        Args:
            dataset_paths: 字典，{数据集名称: 路径}
            samples_per_dataset: 每个数据集采样的样本数
            target_size: 目标尺寸
            use_nir: 是否使用NIR
            transform: 数据增强
        """
        self.dataset_paths = dataset_paths
        self.samples_per_dataset = samples_per_dataset
        self.target_size = target_size
        self.use_nir = use_nir
        self.transform = transform
        
        self.samples = []  # [(dataset_name, image_path, mask_path), ...]
        
        for dataset_name, path in dataset_paths.items():
            sampled = self._sample_dataset(dataset_name, path)
            self.samples.extend(sampled)
            print(f"[MultiDatasetSampler] {dataset_name}: sampled {len(sampled)} images")
        
        print(f"[MultiDatasetSampler] Total: {len(self.samples)} samples")
    
    def _sample_dataset(self, name: str, path: str):
        """从单个数据集采样"""
        samples = []
        
        # 尝试不同的目录结构
        possible_dirs = [
            (os.path.join(path, 'val', 'images'), os.path.join(path, 'val', 'masks')),
            (os.path.join(path, 'test', 'images'), os.path.join(path, 'test', 'masks')),
            (os.path.join(path, 'images'), os.path.join(path, 'masks')),
        ]
        
        image_dir = None
        mask_dir = None
        
        for img_dir, msk_dir in possible_dirs:
            if os.path.exists(img_dir) and os.path.exists(msk_dir):
                image_dir = img_dir
                mask_dir = msk_dir
                break
        
        if image_dir is None:
            print(f"Warning: Could not find valid image/mask dirs for {name}")
            return []
        
        # 获取所有图像
        image_paths = []
        for ext in ['*.jpg', '*.png', '*.tif', '*.tiff']:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        
        image_paths.sort()
        
        # 验证并收集有效样本
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
                    samples.append((name, img_path, mask_path))
                    break
        
        # 随机采样
        if len(samples) > self.samples_per_dataset:
            import random
            samples = random.sample(samples, self.samples_per_dataset)
        
        return samples
    
    def get_validation_loader(self, batch_size: int = 4, num_workers: int = 2):
        """
        获取验证数据加载器
        
        Returns:
            DataLoader that yields batches with 'dataset' key indicating source
        """
        dataset = MultiDatasetValidationSet(
            self.samples,
            target_size=self.target_size,
            use_nir=self.use_nir,
            transform=self.transform
        )
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return loader


class MultiDatasetValidationSet(Dataset):
    """内部类：多数据集验证集"""
    
    def __init__(self, samples, target_size, use_nir, transform):
        self.samples = samples
        self.target_size = target_size
        self.use_nir = use_nir
        self.transform = transform
        
        if use_nir:
            self.nir_generator = NIRGenerator()
        else:
            self.nir_generator = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        dataset_name, img_path, mask_path = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 加载 mask
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
        mask_np = (mask_np > 127).astype(np.uint8)
        mask = Image.fromarray(mask_np)
        
        # 数据增强
        if self.transform is not None:
            image, mask = self.transform(image, mask, 'val')
        else:
            image = image.resize(self.target_size)
            mask = mask.resize(self.target_size, Image.NEAREST)
            image = T.ToTensor()(image)
            mask = torch.from_numpy(np.array(mask)).long()
            
            if self.use_nir and self.nir_generator is not None:
                nir = self.nir_generator(image)
                image = torch.cat([image, nir], dim=0)
        
        return {
            'image': image,
            'mask': mask,
            'filename': os.path.basename(img_path),
            'dataset': dataset_name  # 标识数据来源
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
        mode: str = 'train'
    ):
        """
        Args:
            datasets: 数据集配置列表，每个元素为 dict
                     {'name': str, 'path': str, 'type': str, 'bit_depth': int, 'use_all_bands': bool}
            target_channels: 目标通道数 (3 或 4)
            target_bit_depth: 目标位深 (8 或 16)
            transform: 数据增强
            mode: 'train', 'val', 或 'test'
        """
        self.datasets = datasets
        self.target_channels = target_channels
        self.target_bit_depth = target_bit_depth
        self.transform = transform
        self.mode = mode
        self.use_nir = target_channels == 4
        
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
        """收集单个数据集的样本"""
        samples = []
        
        # 尝试不同的目录结构
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
    
    def _load_and_unify_image(self, ds_type, bit_depth, use_all_bands, img_path):
        """
        加载图像并统一通道和位宽
        
        Returns:
            image: [C, H, W] tensor in [0, 1], dtype float32
        """
        if ds_type == 'cloud_cover':
            # 原生多通道高比特数据
            try:
                import tifffile
                img = tifffile.imread(img_path)
            except ImportError:
                img = np.array(Image.open(img_path))
            
            # 处理维度
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[0] <= 4:
                img = np.transpose(img, (1, 2, 0))
            
            # 归一化
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            elif img.dtype in [np.float32, np.float64]:
                img = img.astype(np.float32)
                if img.max() > 1:
                    img = img / img.max()
            
            # 确保至少3通道
            if img.shape[-1] < 3:
                img = np.repeat(img, 3, axis=-1)
            
            # 处理第4通道
            if self.target_channels == 4:
                if img.shape[-1] >= 4 and use_all_bands:
                    # 有原生 NIR
                    img = img[:, :, :4]
                else:
                    # 需要生成 NIR
                    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                    nir = 0.7 * r + 0.25 * g + 0.05 * b
                    img = np.concatenate([img[:, :, :3], nir[:, :, np.newaxis]], axis=-1)
            else:
                img = img[:, :, :3]
            
        else:
            # 标准 3 通道 8bit 数据
            img = Image.open(img_path).convert('RGB')
            img = np.array(img).astype(np.float32) / 255.0
            
            # 如果需要 4 通道，生成 NIR
            if self.target_channels == 4:
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                nir = 0.7 * r + 0.25 * g + 0.05 * b
                img = np.concatenate([img, nir[:, :, np.newaxis]], axis=-1)
        
        # 转为 tensor [C, H, W]
        img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        
        return img_tensor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        dataset_name, ds_type, bit_depth, use_all_bands, img_path, mask_path = self.samples[idx]
        
        # 加载并统一图像
        image = self._load_and_unify_image(ds_type, bit_depth, use_all_bands, img_path)
        
        # 加载 mask
        mask = Image.open(mask_path).convert('L')
        mask_np = np.array(mask)
        mask_np = (mask_np > 127).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_np).long()
        
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
    
    支持:
    - 多数据集混合训练
    - 自动统一通道和位宽
    - 从训练集分层采样 val/test
    
    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg = config['data']
    train_cfg = config['training']
    
    # 检查是否启用多数据集模式
    if not data_cfg.get('multi_dataset', False):
        # 回退到标准 get_data_loaders
        return get_data_loaders(config, dev_run)
    
    # 统一配置
    target_channels = data_cfg.get('target_channels', 4)
    target_bit_depth = data_cfg.get('target_bit_depth', 16)
    use_nir = target_channels == 4
    
    print(f"\n[MixedDataLoader] Creating unified dataset: {target_channels}ch, {target_bit_depth}bit")
    
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
    
    # 创建统一训练集（包含所有数据）
    full_dataset = UnifiedDataset(
        datasets=datasets,
        target_channels=target_channels,
        target_bit_depth=target_bit_depth,
        transform=transform,
        mode='train'
    )
    
    # 获取 val/test 分割配置
    split_cfg = data_cfg.get('val_test_split', {})
    
    if split_cfg.get('enabled', True):
        # 从训练集分层采样 val/test
        train_ratio = split_cfg.get('train_ratio', 0.75)
        val_ratio = split_cfg.get('val_ratio', 0.15)
        test_ratio = split_cfg.get('test_ratio', 0.10)
        seed = split_cfg.get('seed', 42)
        stratify = split_cfg.get('stratify', True)
        
        total_samples = len(full_dataset)
        indices = list(range(total_samples))
        
        # 按数据集分层
        if stratify:
            # 收集每个数据集的索引
            dataset_indices = defaultdict(list)
            for idx, sample in enumerate(full_dataset.samples):
                dataset_name = sample[0]  # dataset_name 是第0个元素
                dataset_indices[dataset_name].append(idx)
            
            # 从每个数据集按比例采样
            train_indices = []
            val_indices = []
            test_indices = []
            
            import random
            random.seed(seed)
            
            for ds_name, ds_indices in dataset_indices.items():
                n = len(ds_indices)
                random.shuffle(ds_indices)
                
                n_test = int(n * test_ratio)
                n_val = int(n * val_ratio)
                n_train = n - n_val - n_test
                
                test_indices.extend(ds_indices[:n_test])
                val_indices.extend(ds_indices[n_test:n_test+n_val])
                train_indices.extend(ds_indices[n_test+n_val:])
                
                print(f"[MixedDataLoader] {ds_name}: train={n_train}, val={n_val}, test={n_test}")
        else:
            # 随机采样
            import random
            random.seed(seed)
            random.shuffle(indices)
            
            n_test = int(total_samples * test_ratio)
            n_val = int(total_samples * val_ratio)
            n_train = total_samples - n_test - n_val
            
            test_indices = indices[:n_test]
            val_indices = indices[n_test:n_test+n_val]
            train_indices = indices[n_test+n_val:]
        
        # 创建子集
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)
        
        print(f"[MixedDataLoader] Split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    else:
        # 使用全部数据训练，需要外部提供 val/test
        train_dataset = full_dataset
        val_dataset = None
        test_dataset = None
    
    # Dev run: 限制数据集大小
    if dev_run:
        from torch.utils.data import Subset
        train_samples = min(64, len(train_dataset))
        val_samples = min(16, len(val_dataset)) if val_dataset else 0
        test_samples = min(16, len(test_dataset)) if test_dataset else 0
        
        train_dataset = Subset(train_dataset, range(train_samples))
        if val_dataset:
            val_dataset = Subset(val_dataset, range(val_samples))
        if test_dataset:
            test_dataset = Subset(test_dataset, range(test_samples))
        
        print(f"[Dev Run] Limited: train={train_samples}, val={val_samples}, test={test_samples}")
    
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
        drop_last=True
    )
    
    val_loader = None
    test_loader = None
    
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
    
    return train_loader, val_loader, test_loader
