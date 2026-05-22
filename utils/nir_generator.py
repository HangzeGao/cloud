"""
伪NIR通道生成器
===============

从RGB图像生成伪近红外（NIR）通道的多种算法。
支持物理模型、植被指数、引导滤波等多种方法。

主要功能:
- 物理模型生成: 基于光谱响应的线性组合
- 植被指数反推: 基于NDVI反演
- 引导滤波生成: 边缘保持的平滑生成
- 上下文感知生成: 根据场景类型自适应调整
- 集成生成: 多种方法融合

参考:
- Gitelson et al. (1996) 植被指数估计
- He et al. (2010) 引导滤波算法

使用示例:
    >>> from utils.nir_generator import generate_pseudo_nir, NIRMethod
    >>> nir = generate_pseudo_nir(rgb_image, method='ensemble')
"""
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import numpy as np
from typing import Union, Optional, Dict, List
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# 枚举和方法定义
# =============================================================================

class NIRMethod(Enum):
    """伪NIR生成方法枚举"""
    PHYSICAL = "physical"
    VEGETATION = "vegetation"
    GUIDED = "guided"
    CONTEXT = "context"
    ENSEMBLE = "ensemble"


# =============================================================================
# 基类
# =============================================================================

class BaseNIRGenerator(ABC):
    """
    伪NIR生成器基类
    
    定义生成器的统一接口，支持 NumPy 数组和 PyTorch Tensor。
    """
    
    @abstractmethod
    def generate(
        self,
        rgb: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        从RGB生成伪NIR
        
        Args:
            rgb: RGB图像
                - numpy: [H, W, 3] 或 [B, H, W, 3]
                - torch: [3, H, W] 或 [B, 3, H, W]
            
        Returns:
            nir: 伪NIR通道，格式与输入一致
                - numpy: [H, W] 或 [B, H, W]
                - torch: [H, W] 或 [B, H, W]
        """
        pass
    
    def __call__(
        self,
        rgb: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """便捷调用接口"""
        return self.generate(rgb)
    
    @staticmethod
    def _extract_channels(
        rgb: Union[np.ndarray, torch.Tensor]
    ) -> tuple:
        """
        提取RGB通道
        
        Returns:
            Tuple[R, G, B]: 三个通道
        """
        if isinstance(rgb, torch.Tensor):
            if rgb.dim() == 3:  # [3, H, W]
                return rgb[0], rgb[1], rgb[2]
            elif rgb.dim() == 4:  # [B, 3, H, W]
                return rgb[:, 0], rgb[:, 1], rgb[:, 2]
            else:
                raise ValueError(f"Expected 3 or 4 dim tensor, got {rgb.dim()}")
        else:
            # numpy [H, W, 3]
            return rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]


# =============================================================================
# 物理模型生成器
# =============================================================================

class PhysicalNIRGenerator(BaseNIRGenerator):
    """
    物理模型生成器
    
    基于光谱响应的线性组合，参考典型卫星传感器光谱响应函数。
    
    默认权重参考:
    - R: 0.65 (红光在NIR波段的响应)
    - G: 0.25 (绿光在NIR波段的响应)
    - B: 0.10 (蓝光在NIR波段的响应)
    
    参考: Gitelson et al. (1996)
    """
    
    DEFAULT_WEIGHTS = [0.65, 0.25, 0.10]
    
    def __init__(self, weights: Optional[List[float]] = None):
        """
        Args:
            weights: [w_r, w_g, w_b] 权重列表
        """
        self.weights = np.array(weights or self.DEFAULT_WEIGHTS, dtype=np.float32)
    
    def generate(
        self,
        rgb: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """生成伪NIR"""
        r, g, b = self._extract_channels(rgb)
        
        nir = (
            self.weights[0] * r +
            self.weights[1] * g +
            self.weights[2] * b
        )
        
        if isinstance(rgb, torch.Tensor):
            return torch.clamp(nir, 0, 1)
        else:
            return np.clip(nir, 0, 255).astype(rgb.dtype)


# =============================================================================
# 植被指数生成器
# =============================================================================

class VegetationIndexNIRGenerator(BaseNIRGenerator):
    """
    植被指数反推生成器
    
    利用NDVI与植被覆盖度的关系反推NIR值。
    适合有云区域，因为云的NDVI接近0。
    
    公式:
        NIR = NDVI * (R + R) / (1 - NDVI)
        
    其中NDVI通过绿红比值估算:
        VI = (G - R) / (G + R)
        NDVI_est = ndvi_base + ndvi_scale * sigmoid(VI * 4)
    """
    
    def __init__(
        self,
        ndvi_base: float = 0.1,
        ndvi_scale: float = 0.5
    ):
        """
        Args:
            ndvi_base: 基础NDVI值（云的NDVI约0.1）
            ndvi_scale: NDVI缩放因子
        """
        self.ndvi_base = ndvi_base
        self.ndvi_scale = ndvi_scale
    
    def generate(
        self,
        rgb: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """生成伪NIR"""
        r, g, _ = self._extract_channels(rgb)
        
        eps = 1e-8
        
        if isinstance(rgb, torch.Tensor):
            # 使用绿红比值作为植被指数近似
            vi = (g - r) / (g + r + eps)
            ndvi_est = self.ndvi_base + self.ndvi_scale * torch.sigmoid(vi * 4)
            
            # 反推NIR
            nir = ndvi_est * (r + r) / (1 - ndvi_est + eps)
            return torch.clamp(nir, 0, 1)
        else:
            vi = (g - r) / (g + r + eps)
            ndvi_est = self.ndvi_base + self.ndvi_scale * (1 / (1 + np.exp(-vi * 4)))
            nir = ndvi_est * (r + r) / (1 - ndvi_est + eps)
            return np.clip(nir, 0, 255).astype(rgb.dtype)


# =============================================================================
# 引导滤波生成器
# =============================================================================

class GuidedFilterNIRGenerator(BaseNIRGenerator):
    """
    引导滤波生成器
    
    使用RGB作为引导图，生成边缘对齐的NIR。
    基于He et al. (2010) 引导滤波算法。
    
    特点:
    - 保持边缘细节
    - 在平滑区域进行滤波
    - 适合处理具有清晰边界的图像
    """
    
    def __init__(
        self,
        radius: int = 8,
        eps: float = 0.01
    ):
        """
        Args:
            radius: 滤波半径
            eps: 正则化参数
        """
        self.radius = radius
        self.eps = eps
        self.physical_gen = PhysicalNIRGenerator()
    
    def _box_filter(self, x, r):
        """盒式滤波（均值滤波）"""
        if isinstance(x, torch.Tensor):
            return F.avg_pool2d(
                x,
                kernel_size=2*r+1,
                stride=1,
                padding=r,
                count_include_pad=False
            )
        else:
            from scipy.ndimage import uniform_filter
            return uniform_filter(x, size=2*r+1)
    
    def generate(
        self,
        rgb: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """生成伪NIR"""
        # 先生成初始NIR估计
        p = self.physical_gen(rgb)
        
        if isinstance(rgb, torch.Tensor):
            # 计算亮度作为引导图
            if rgb.dim() == 3:
                I = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            else:
                I = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
            
            # 高斯平滑
            if p.dim() == 2:
                p_smooth = TVF.gaussian_blur(
                    p.unsqueeze(0),
                    kernel_size=self.radius*2+1,
                    sigma=self.radius/2
                ).squeeze(0)
            else:
                p_smooth = TVF.gaussian_blur(
                    p.unsqueeze(1),
                    kernel_size=self.radius*2+1,
                    sigma=self.radius/2
                ).squeeze(1)
            
            # 使用梯度检测边缘
            if p.dim() == 2:
                grad_x = torch.abs(p[:, 1:] - p[:, :-1])
                grad_y = torch.abs(p[1:, :] - p[:-1, :])
                edges = torch.zeros_like(p)
                edges[:, :-1] += grad_x
                edges[:-1, :] += grad_y
            else:
                grad_x = torch.abs(p[:, :, 1:] - p[:, :, :-1])
                grad_y = torch.abs(p[:, 1:, :] - p[:, :-1, :])
                edges = torch.zeros_like(p)
                edges[:, :, :-1] += grad_x
                edges[:, :-1, :] += grad_y
            
            # 边缘区域减少平滑
            edge_weight = torch.sigmoid(-edges * 5)
            
            # 边缘加权融合
            nir = edge_weight * p + (1 - edge_weight) * p_smooth
            return torch.clamp(nir, 0, 1)
        else:
            # numpy版本简化处理
            from scipy.ndimage import gaussian_filter
            p_smooth = gaussian_filter(p, sigma=self.radius/2)
            nir = 0.7 * p + 0.3 * p_smooth
            return np.clip(nir, 0, 255).astype(rgb.dtype)


# =============================================================================
# 上下文感知生成器
# =============================================================================

class ContextAwareNIRGenerator(BaseNIRGenerator):
    """
    上下文感知生成器
    
    分析图像内容（云、植被、水体），分别使用不同策略生成NIR。
    
    策略:
    - 云区域: 偏红色，使用 cloud_weights
    - 植被区域: 偏绿色，使用 veg_weights
    - 水体区域: 偏蓝色，使用 water_weights
    - 其他区域: 使用物理模型
    """
    
    # 各类型的预设权重
    CLOUD_WEIGHTS = np.array([0.7, 0.2, 0.1])   # 云：偏红
    VEG_WEIGHTS = np.array([0.3, 0.6, 0.1])     # 植被：偏绿
    WATER_WEIGHTS = np.array([0.2, 0.3, 0.5]) # 水体：偏蓝
    
    # 分类阈值（归一化后0-1范围）
    CLOUD_BRIGHTNESS_THRESH = 0.7
    CLOUD_COLOR_THRESH = 0.1
    VEG_BRIGHTNESS_THRESH = 0.4
    
    def __init__(self):
        self.physical_gen = PhysicalNIRGenerator()
    
    def _classify_regions(
        self,
        rgb: Union[np.ndarray, torch.Tensor]
    ) -> tuple:
        """
        简单分类：云、植被、水体、其他
        
        Returns:
            Tuple[is_cloud, is_veg, is_water, is_other]: 各类别的掩码
        """
        r, g, b = self._extract_channels(rgb)
        
        if isinstance(rgb, torch.Tensor):
            brightness = (r + g + b) / 3
            
            # 云：高亮度 + 白（R≈G≈B）
            is_cloud = (
                (brightness > self.CLOUD_BRIGHTNESS_THRESH) &
                (torch.abs(r - g) < self.CLOUD_COLOR_THRESH) &
                (torch.abs(g - b) < self.CLOUD_COLOR_THRESH)
            )
            
            # 植被：绿 > 红 > 蓝
            is_veg = (g > r) & (g > b) & (r > b)
            
            # 水体：蓝 > 绿 > 红，低亮度
            is_water = (b > g) & (g > r) & (brightness < self.VEG_BRIGHTNESS_THRESH)
            
            # 默认：其他
            is_other = ~(is_cloud | is_veg | is_water)
            
            return (
                is_cloud.float(),
                is_veg.float(),
                is_water.float(),
                is_other.float()
            )
        else:
            brightness = (r + g + b) / 3
            
            # numpy版本使用0-255范围
            is_cloud = (
                (brightness > 180) &
                (np.abs(r - g) < 30) &
                (np.abs(g - b) < 30)
            )
            is_veg = (g > r) & (g > b) & (r > b)
            is_water = (b > g) & (g > r) & (brightness < 100)
            is_other = ~(is_cloud | is_veg | is_water)
            
            return is_cloud, is_veg, is_water, is_other
    
    def generate(
        self,
        rgb: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """生成伪NIR"""
        r, g, b = self._extract_channels(rgb)
        
        # 获取区域掩码
        mask_cloud, mask_veg, mask_water, mask_other = self._classify_regions(rgb)
        
        if isinstance(rgb, torch.Tensor):
            # 各区域使用不同权重生成NIR
            nir_cloud = self.CLOUD_WEIGHTS[0] * r + self.CLOUD_WEIGHTS[1] * g + self.CLOUD_WEIGHTS[2] * b
            nir_veg = self.VEG_WEIGHTS[0] * r + self.VEG_WEIGHTS[1] * g + self.VEG_WEIGHTS[2] * b
            nir_water = self.WATER_WEIGHTS[0] * r + self.WATER_WEIGHTS[1] * g + self.WATER_WEIGHTS[2] * b
            
            # 其他区域使用物理模型
            nir_other = self.physical_gen(rgb)
            
            # 融合
            nir = mask_cloud * nir_cloud + mask_veg * nir_veg + mask_water * nir_water + mask_other * nir_other
            
            return torch.clamp(nir, 0, 1)
        else:
            nir_cloud = self.CLOUD_WEIGHTS[0] * r + self.CLOUD_WEIGHTS[1] * g + self.CLOUD_WEIGHTS[2] * b
            nir_veg = self.VEG_WEIGHTS[0] * r + self.VEG_WEIGHTS[1] * g + self.VEG_WEIGHTS[2] * b
            nir_water = self.WATER_WEIGHTS[0] * r + self.WATER_WEIGHTS[1] * g + self.WATER_WEIGHTS[2] * b
            nir_other = self.physical_gen(rgb)
            
            nir = mask_cloud * nir_cloud + mask_veg * nir_veg + mask_water * nir_water + mask_other * nir_other
            
            return np.clip(nir, 0, 255).astype(rgb.dtype)


# =============================================================================
# 集成生成器
# =============================================================================

class EnsembleNIRGenerator(BaseNIRGenerator):
    """
    集成生成器
    
    组合多个生成器的结果，通过加权融合得到最终的NIR估计。
    通常能获得比单一方法更稳定的结果。
    """
    
    DEFAULT_METHODS = ['physical', 'vegetation', 'guided', 'context']
    DEFAULT_WEIGHTS = [1.0, 1.0, 1.0, 1.0]
    
    def __init__(
        self,
        methods: Optional[List[str]] = None,
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            methods: 方法列表 ['physical', 'vegetation', 'guided', 'context']
            weights: 各方法权重，None表示等权重
        """
        methods = methods or self.DEFAULT_METHODS
        weights = weights or self.DEFAULT_WEIGHTS
        
        self.generators: Dict[str, BaseNIRGenerator] = {}
        
        if 'physical' in methods:
            self.generators['physical'] = PhysicalNIRGenerator()
        if 'vegetation' in methods:
            self.generators['vegetation'] = VegetationIndexNIRGenerator()
        if 'guided' in methods:
            self.generators['guided'] = GuidedFilterNIRGenerator()
        if 'context' in methods:
            self.generators['context'] = ContextAwareNIRGenerator()
        
        # 归一化权重
        self.weights = np.array(weights[:len(self.generators)]) / sum(weights[:len(self.generators)])
    
    def generate(
        self,
        rgb: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """生成伪NIR（集成方法）"""
        results = []
        for gen in self.generators.values():
            results.append(gen(rgb))
        
        if isinstance(rgb, torch.Tensor):
            # 加权融合
            nir = sum(w * r for w, r in zip(self.weights, results))
            return torch.clamp(nir, 0, 1)
        else:
            nir = sum(w * r for w, r in zip(self.weights, results))
            return np.clip(nir, 0, 255).astype(rgb.dtype)


# =============================================================================
# 便捷函数
# =============================================================================

def generate_pseudo_nir(
    rgb: Union[np.ndarray, torch.Tensor],
    method: str = 'ensemble',
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    便捷函数：从RGB生成伪NIR
    
    Args:
        rgb: RGB图像
            - numpy: [H, W, 3] 或 [B, H, W, 3]，值范围 [0, 255]
            - torch: [3, H, W] 或 [B, 3, H, W]，值范围 [0, 1]
        method: 生成方法
            - 'physical': 物理模型
            - 'vegetation': 植被指数
            - 'guided': 引导滤波
            - 'context': 上下文感知
            - 'ensemble': 集成方法（默认）
        **kwargs: 各方法特定参数
        
    Returns:
        nir: 伪NIR通道
            - numpy: [H, W] 或 [B, H, W]，值范围 [0, 255]
            - torch: [H, W] 或 [B, H, W]，值范围 [0, 1]
            
    Example:
        >>> import numpy as np
        >>> rgb = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        >>> nir = generate_pseudo_nir(rgb, method='physical')
        >>> nir.shape  # (512, 512)
    """
    generators = {
        'physical': PhysicalNIRGenerator,
        'vegetation': VegetationIndexNIRGenerator,
        'guided': GuidedFilterNIRGenerator,
        'context': ContextAwareNIRGenerator,
        'ensemble': EnsembleNIRGenerator,
    }
    
    if method not in generators:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Available: {list(generators.keys())}"
        )
    
    gen = generators[method](**kwargs)
    return gen(rgb)


def get_available_methods() -> List[str]:
    """获取可用的NIR生成方法列表"""
    return ['physical', 'vegetation', 'guided', 'context', 'ensemble']


def get_method_info(method: str) -> Dict[str, str]:
    """
    获取方法的详细信息
    
    Args:
        method: 方法名称
        
    Returns:
        Dict包含name, description, use_case等信息
    """
    info = {
        'physical': {
            'name': '物理模型',
            'description': '基于光谱响应的线性组合',
            'use_case': '通用场景，计算简单',
            'speed': '快'
        },
        'vegetation': {
            'name': '植被指数',
            'description': '基于NDVI反演',
            'use_case': '植被丰富的场景',
            'speed': '快'
        },
        'guided': {
            'name': '引导滤波',
            'description': '边缘保持的平滑生成',
            'use_case': '需要保持边缘细节的场景',
            'speed': '中等'
        },
        'context': {
            'name': '上下文感知',
            'description': '根据场景类型自适应调整',
            'use_case': '复杂场景（云、植被、水体并存）',
            'speed': '中等'
        },
        'ensemble': {
            'name': '集成方法',
            'description': '多种方法融合',
            'use_case': '追求最佳质量的场景',
            'speed': '慢'
        }
    }
    return info.get(method, {'name': 'Unknown', 'description': ''})


class NIRGeneratorFactory:
    """
    NIR生成器工厂类
    
    提供统一的生成器创建接口。
    
    Example:
        >>> factory = NIRGeneratorFactory()
        >>> gen = factory.create('guided', radius=16)
        >>> nir = gen(rgb_image)
    """
    
    _generators = {
        'physical': PhysicalNIRGenerator,
        'vegetation': VegetationIndexNIRGenerator,
        'guided': GuidedFilterNIRGenerator,
        'context': ContextAwareNIRGenerator,
        'ensemble': EnsembleNIRGenerator,
    }
    
    @classmethod
    def create(
        cls,
        method: str,
        **kwargs
    ) -> BaseNIRGenerator:
        """
        创建生成器实例
        
        Args:
            method: 方法名称
            **kwargs: 构造参数
            
        Returns:
            BaseNIRGenerator: 生成器实例
        """
        if method not in cls._generators:
            raise ValueError(f"Unknown method: {method}")
        return cls._generators[method](**kwargs)
    
    @classmethod
    def list_methods(cls) -> List[str]:
        """列出可用的方法"""
        return list(cls._generators.keys())
