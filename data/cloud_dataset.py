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
    创建数据加载器 (支持 RGB+NIR 四通道)
    
    Args:
        config: 配置字典
        dev_run: 如果为True，只使用少量样本进行快速开发测试
        
    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg = config['data']
    train_cfg = config['training']
    
    # NIR 配置
    use_nir = data_cfg.get('use_nir', True)
    nir_method = data_cfg.get('nir_method', 'physical')
    nir_gain = data_cfg.get('nir_gain', 1.1)
    
    if use_nir:
        print(f"[DataLoader] NIR enabled: method={nir_method}, gain={nir_gain}")
    
    # 数据增强配置（传递 NIR 参数）
    aug_config = train_cfg.get('augmentation', {'enabled': False})
    aug_config['use_nir'] = use_nir
    aug_config['nir_method'] = nir_method
    aug_config['nir_gain'] = nir_gain
    transform = CloudAugmentation(aug_config)
    
    # 训练集
    train_dataset = CloudSegmentationDataset(
        image_dir=os.path.join(data_cfg['train_data_path'], 'images'),
        mask_dir=os.path.join(data_cfg['train_data_path'], 'masks'),
        mode='train',
        transform=transform,
        use_nir=use_nir,
        nir_method=nir_method,
        nir_gain=nir_gain
    )
    
    # 验证集
    val_dataset = CloudSegmentationDataset(
        image_dir=os.path.join(data_cfg['val_data_path'], 'images'),
        mask_dir=os.path.join(data_cfg['val_data_path'], 'masks'),
        mode='val',
        transform=transform,
        use_nir=use_nir,
        nir_method=nir_method,
        nir_gain=nir_gain
    )
    
    # 测试集
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
