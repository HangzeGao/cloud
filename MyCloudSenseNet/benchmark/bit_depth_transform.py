"""
位深度模拟变换 - Albumentations风格
用于在训练时模拟不同位深度的图像效果，增强模型泛化能力

使用示例:
    import albumentations as A
    from MyCloudSenseNet.benchmark.bit_depth_transform import BitDepthSimulation
    
    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.0625, rotate_limit=15, p=0.5),
        BitDepthSimulation(bit_depth_range=(8, 11), p=0.5),  # 50%概率模拟8-11位深度
        A.HorizontalFlip(p=0.5),
    ])
    
    # 应用变换
    transformed = transform(image=image)
    simulated_image = transformed['image']  # 模拟后的图像
    bit_depth = transformed.get('simulated_bit_depth', None)  # 实际使用的位深度
"""

import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.core.transforms_interface import DualTransform


class BitDepthSimulation(ImageOnlyTransform):
    """
    模拟特定位深度图像的变换
    
    通过量化到低位深度并添加量化噪声，模拟低比特图像的效果。
    输入图像应在 [0, 1] 或 [0, 255] 范围内。
    
    Args:
        bit_depth_range: 元组 (min, max)，指定随机选择的位深度范围
            默认 (8, 11) 表示随机选择 8, 9, 10 位深度
            设为固定值如 (10, 10) 则始终使用10位
        input_max_value: 输入图像的最大值（归一化方式）
            - 1.0: 图像已归一化到 [0, 1]（默认）
            - 255.0: 图像在 [0, 255] 范围
            - 1023.0: 图像在 [0, 1023] 范围（10-bit）
        add_noise: 是否添加量化噪声，默认True
        noise_factor: 噪声强度因子，默认0.5
        always_apply: 是否总是应用此变换
        p: 应用此变换的概率
        
    Targets:
        image
        
    Image types:
        float32, float64, uint8, uint16
        
    Example:
        >>> import albumentations as A
        >>> from bit_depth_transform import BitDepthSimulation
        >>> 
        >>> # 基础用法：50%概率模拟8-11位深度
        >>> transform = A.Compose([
        ...     BitDepthSimulation(bit_depth_range=(8, 11), p=0.5),
        ... ])
        >>> 
        >>> # 固定模拟10-bit
        >>> transform = BitDepthSimulation(bit_depth_range=(10, 10), p=1.0)
        >>> 
        >>> # 处理已归一化到[0,1]的图像（默认）
        >>> result = transform(image=normalized_image)
        >>> simulated = result['image']
        >>> used_bit_depth = result['simulated_bit_depth']  # 10
        >>> 
        >>> # 处理原始16-bit图像
        >>> transform = BitDepthSimulation(
        ...     bit_depth_range=(8, 12),
        ...     input_max_value=65535,  # 16-bit原始值
        ...     p=0.5
        ... )
    """
    
    def __init__(
        self,
        bit_depth_range: tuple = (8, 11),
        input_max_value: float = 1.0,
        add_noise: bool = True,
        noise_factor: float = 0.5,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.bit_depth_range = bit_depth_range
        self.input_max_value = input_max_value
        self.add_noise = add_noise
        self.noise_factor = noise_factor
        
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        应用位深度模拟变换
        
        Args:
            img: 输入图像，numpy数组
            **params: 包含 'target_bit_depth' 参数
            
        Returns:
            模拟后的图像
        """
        target_bit_depth = params['target_bit_depth']
        return self._simulate_bit_depth_numpy(
            img, 
            target_bit_depth,
            self.input_max_value,
            self.add_noise,
            self.noise_factor
        )
    
    def get_params(self) -> dict:
        """
        随机选择目标位深度
        """
        min_bd, max_bd = self.bit_depth_range
        # 包含边界，随机选择整数位深度
        target_bd = np.random.randint(int(min_bd), int(max_bd) + 1)
        return {'target_bit_depth': target_bd}
    
    def get_transform_init_args_names(self) -> tuple:
        """
        返回初始化参数名称，用于序列化
        """
        return ('bit_depth_range', 'input_max_value', 'add_noise', 'noise_factor')
    
    @property
    def targets_as_params(self) -> list:
        """
        指定哪些目标需要作为参数传递给apply
        """
        return ['image']
    
    def _simulate_bit_depth_numpy(
        self,
        x: np.ndarray,
        target_bit_depth: int,
        input_max_value: float,
        add_noise: bool,
        noise_factor: float,
    ) -> np.ndarray:
        """
        内部实现：模拟特定位深度
        
        Args:
            x: 输入数组，任意形状，值范围 [0, input_max_value]
            target_bit_depth: 目标位深度（8-16）
            input_max_value: 输入的最大值
            add_noise: 是否添加噪声
            noise_factor: 噪声因子
            
        Returns:
            模拟后的数组，与输入同shape
        """
        # 保存原始dtype
        original_dtype = x.dtype
        
        # 转换为float32处理
        x = x.astype(np.float32)
        
        # 如果目标位深度高于或等于输入的实际位深度，直接返回
        # 假设输入通常是12-bit或更高
        if target_bit_depth >= 12:
            return x.astype(original_dtype)
        
        # 归一化到 [0, 1]
        if input_max_value != 1.0:
            x_normalized = x / input_max_value
        else:
            x_normalized = x.copy()
        
        # 计算目标位深度的最大值
        max_val = 2 ** target_bit_depth - 1  # 8-bit: 255, 10-bit: 1023
        
        # 量化：映射到目标位深度范围
        # 先映射到 [0, max_val]，然后取整，再归一化回 [0, 1]
        x_quantized = np.floor(x_normalized * max_val + 0.5) / max_val
        
        # 添加量化噪声（模拟真实低比特相机的噪声特性）
        if add_noise:
            # 噪声范围为一个量化步长
            noise_scale = 1.0 / max_val
            noise = np.random.rand(*x.shape).astype(np.float32) * noise_scale * noise_factor
            x_simulated = x_quantized + noise
            # 裁剪到有效范围
            x_simulated = np.clip(x_simulated, 0.0, 1.0)
        else:
            x_simulated = x_quantized
        
        # 如果需要，转回原始尺度
        if input_max_value != 1.0:
            x_simulated = x_simulated * input_max_value
        
        # 转回原始dtype或保持float32
        if original_dtype in [np.uint8, np.uint16, np.int8, np.int16]:
            # 对于整数类型，需要裁剪并转换
            if original_dtype == np.uint8:
                x_simulated = np.clip(x_simulated, 0, 255).astype(np.uint8)
            elif original_dtype == np.uint16:
                x_simulated = np.clip(x_simulated, 0, 65535).astype(np.uint16)
            else:
                x_simulated = x_simulated.astype(original_dtype)
        else:
            x_simulated = x_simulated.astype(original_dtype)
        
        return x_simulated


class BitDepthSimulationWithLabel(DualTransform):
    """
    位深度模拟变换（同时处理图像和标签）
    
    用于语义分割任务，确保标签随图像一起变换。
    注意：位深度模拟只应用于图像，不应用于标签。
    
    Args:
        bit_depth_range: 元组 (min, max)，指定随机选择的位深度范围
        input_max_value: 输入图像的最大值
        add_noise: 是否添加量化噪声
        noise_factor: 噪声强度因子
        always_apply: 是否总是应用此变换
        p: 应用此变换的概率
        
    Targets:
        image, mask
        
    Example:
        >>> transform = A.Compose([
        ...     BitDepthSimulationWithLabel(bit_depth_range=(8, 11), p=0.5),
        ...     A.RandomCrop(512, 512),
        ... ])
        >>> result = transform(image=img, mask=label)
        >>> simulated_img = result['image']
        >>> mask = result['mask']  # 标签不变
    """
    
    def __init__(
        self,
        bit_depth_range: tuple = (8, 11),
        input_max_value: float = 1.0,
        add_noise: bool = True,
        noise_factor: float = 0.5,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.bit_depth_range = bit_depth_range
        self.input_max_value = input_max_value
        self.add_noise = add_noise
        self.noise_factor = noise_factor
        
        # 内部使用ImageOnlyTransform处理图像
        self._img_transform = BitDepthSimulation(
            bit_depth_range=bit_depth_range,
            input_max_value=input_max_value,
            add_noise=add_noise,
            noise_factor=noise_factor,
            always_apply=True,  # 内部总是应用，外部控制概率
            p=1.0,
        )
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """应用于图像"""
        return self._img_transform.apply(img, **params)
    
    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        """应用于标签（保持不变）"""
        # 位深度模拟不改变标签
        return mask
    
    def get_params(self) -> dict:
        """获取随机参数"""
        return self._img_transform.get_params()
    
    def get_transform_init_args_names(self) -> tuple:
        """返回初始化参数名称"""
        return self._img_transform.get_transform_init_args_names()
    
    @property
    def targets_as_params(self) -> list:
        """指定需要作为参数的目标"""
        return ['image']


class RandomBitDepthBatch:
    """
    批次级随机位深度模拟
    
    用于PyTorch DataLoader的collate_fn中，对整个batch应用随机位深度。
    不同于单样本变换，这可以在GPU上进行（如果输入是tensor）。
    
    Args:
        bit_depth_range: 位深度范围
        probability: 应用变换的概率
        
    Example:
        >>> # 在DataLoader中使用
        >>> from torch.utils.data import DataLoader
        >>> 
        >>> bit_depth_sim = RandomBitDepthBatch(bit_depth_range=(8, 11), probability=0.5)
        >>> 
        >>> def custom_collate_fn(batch):
        ...     # 标准collate
        ...     images = torch.stack([item['chip'] for item in batch])
        ...     labels = torch.stack([item['label'] for item in batch])
        ...     
        ...     # 应用位深度模拟
        ...     images = bit_depth_sim(images)
        ...     
        ...     return {'chip': images, 'label': labels}
        >>> 
        >>> loader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn)
    """
    
    def __init__(
        self,
        bit_depth_range: tuple = (8, 11),
        probability: float = 0.5,
    ):
        self.bit_depth_range = bit_depth_range
        self.probability = probability
    
    def __call__(self, images: 'torch.Tensor') -> 'torch.Tensor':
        """
        应用于batch图像
        
        Args:
            images: 图像tensor [B, C, H, W]，值范围 [0, 1]
            
        Returns:
            模拟后的图像tensor
        """
        import torch
        
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(images)}")
        
        if torch.rand(1).item() > self.probability:
            return images
        
        # 随机选择目标位深度
        min_bd, max_bd = self.bit_depth_range
        target_bd = torch.randint(int(min_bd), int(max_bd) + 1, (1,)).item()
        
        if target_bd >= 12:
            return images
        
        # 量化
        max_val = 2 ** target_bd - 1
        x_quantized = torch.floor(images * max_val + 0.5) / max_val
        
        # 添加噪声
        noise = torch.rand_like(images) / max_val * 0.5
        x_simulated = x_quantized + noise
        
        return torch.clamp(x_simulated, 0, 1)


# 辅助函数：快速创建位深度模拟变换
def create_bit_depth_transform(
    mode: str = "random",
    fixed_bit_depth: int = None,
    probability: float = 0.5,
    input_range: str = "normalized",  # "normalized", "uint8", "uint16", "10bit", "12bit"
):
    """
    快速创建位深度模拟变换
    
    Args:
        mode: "random" - 随机选择位深度, "fixed" - 固定位深度
        fixed_bit_depth: mode="fixed"时使用的固定位深度
        probability: 应用变换的概率
        input_range: 输入图像的值范围
            - "normalized": [0, 1]（默认）
            - "uint8": [0, 255]
            - "uint16": [0, 65535]
            - "10bit": [0, 1023]
            - "12bit": [0, 4095]
            
    Returns:
        配置好的BitDepthSimulation实例
        
    Example:
        >>> # 随机8-11位，输入已归一化
        >>> transform = create_bit_depth_transform(mode="random", probability=0.5)
        >>> 
        >>> # 固定10位，输入是10-bit原始值
        >>> transform = create_bit_depth_transform(
        ...     mode="fixed",
        ...     fixed_bit_depth=10,
        ...     input_range="10bit"
        ... )
        >>> 
        >>> # 在Compose中使用
        >>> import albumentations as A
        >>> composed = A.Compose([
        ...     A.HorizontalFlip(p=0.5),
        ...     create_bit_depth_transform(mode="random", probability=0.3),
        ... ])
    """
    # 映射input_range到max_value
    range_to_max = {
        "normalized": 1.0,
        "uint8": 255.0,
        "uint16": 65535.0,
        "10bit": 1023.0,
        "12bit": 4095.0,
    }
    
    if input_range not in range_to_max:
        raise ValueError(f"Unknown input_range: {input_range}. Choose from {list(range_to_max.keys())}")
    
    max_value = range_to_max[input_range]
    
    if mode == "fixed":
        if fixed_bit_depth is None:
            raise ValueError("fixed_bit_depth must be specified when mode='fixed'")
        bit_depth_range = (fixed_bit_depth, fixed_bit_depth)
    elif mode == "random":
        bit_depth_range = (8, 11)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'random' or 'fixed'")
    
    return BitDepthSimulation(
        bit_depth_range=bit_depth_range,
        input_max_value=max_value,
        p=probability,
    )


# 预定义的常用配置
BIT_DEPTH_8_TO_11_RANDOM = lambda p=0.5: BitDepthSimulation(bit_depth_range=(8, 11), p=p)
BIT_DEPTH_FIXED_8 = lambda p=1.0: BitDepthSimulation(bit_depth_range=(8, 8), p=p)
BIT_DEPTH_FIXED_10 = lambda p=1.0: BitDepthSimulation(bit_depth_range=(10, 10), p=p)
BIT_DEPTH_FIXED_12 = lambda p=1.0: BitDepthSimulation(bit_depth_range=(12, 12), p=p)
