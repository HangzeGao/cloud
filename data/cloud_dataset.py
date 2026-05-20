"""
云分割数据集加载器 - 精简版
直接使用上层 Data 目录的 UnifiedCloudDataset
保留数据增强功能
"""
import sys
from pathlib import Path

import torch
from torch.fx.experimental import normalize
from torch.utils.data import DataLoader, Subset
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF


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
            image: PIL.Image (RGB) or torch.Tensor [C, H, W] or numpy.ndarray [C, H, W]
            mask: PIL.Image (L) or torch.Tensor [H, W] or numpy.ndarray [H, W]
            mode: 'train' or 'val'
            
        Returns:
            image: torch.Tensor [C, H, W] (通道数由输入决定或根据 use_nir 调整)
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
        
        # 根据 use_nir 调整通道数（上层数据集已提供正确的通道）
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
    print(f"\n[MixedDataLoader] Using UnifiedCloudDataset from upper Data directory")
    print(f"  Datasets: {dataset_names}")
    print(f"  Target: {target_channels}ch (use_nir={use_nir})")
    
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
