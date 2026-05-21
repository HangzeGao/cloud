"""
伪NIR通道生成器 - 多种先进算法

用于从RGB图像生成伪近红外（NIR）通道的多种算法，
基于业界常用的物理模型、植被指数和深度学习方法。
"""
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import numpy as np
from typing import Union, Optional


class BaseNIRGenerator:
    """伪NIR生成器基类"""
    
    def __call__(self, rgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        从RGB生成伪NIR
        
        Args:
            rgb: RGB图像，numpy [H, W, 3] 或 torch [3, H, W] 或 [B, 3, H, W]
            
        Returns:
            nir: 伪NIR通道，格式与输入一致
        """
        raise NotImplementedError


class PhysicalNIRGenerator(BaseNIRGenerator):
    """
    物理模型生成器 - 基于光谱响应的线性组合
    
    参考：
    - Gitelson et al. (1996) 植被指数估计
    - 典型卫星传感器光谱响应函数
    """
    
    def __init__(self, weights: Optional[list] = None):
        """
        Args:
            weights: [w_r, w_g, w_b] 权重，默认 [0.65, 0.25, 0.10]
        """
        self.weights = np.array(weights or [0.65, 0.25, 0.10], dtype=np.float32)
        
    def __call__(self, rgb):
        if isinstance(rgb, torch.Tensor):
            # 处理不同维度
            if rgb.dim() == 3:  # [3, H, W]
                r, g, b = rgb[0], rgb[1], rgb[2]
            elif rgb.dim() == 4:  # [B, 3, H, W]
                r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            else:
                raise ValueError(f"Expected 3 or 4 dim tensor, got {rgb.dim()}")
            
            nir = self.weights[0] * r + self.weights[1] * g + self.weights[2] * b
            return torch.clamp(nir, 0, 1)
        else:
            # numpy [H, W, 3]
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            nir = self.weights[0] * r + self.weights[1] * g + self.weights[2] * b
            return np.clip(nir, 0, 255).astype(rgb.dtype)


class VegetationIndexNIRGenerator(BaseNIRGenerator):
    """
    植被指数反推生成器 - 基于NDVI反演
    
    原理：利用NDVI与植被覆盖度的关系，反推NIR值
    NIR = NDVI * (R + R) / (1 - NDVI)
    
    适合有云区域，因为云的NDVI接近0
    """
    
    def __init__(self, ndvi_base: float = 0.1, ndvi_scale: float = 0.5):
        """
        Args:
            ndvi_base: 基础NDVI值（云的NDVI约0.1）
            ndvi_scale: NDVI缩放因子
        """
        self.ndvi_base = ndvi_base
        self.ndvi_scale = ndvi_scale
    
    def __call__(self, rgb):
        if isinstance(rgb, torch.Tensor):
            if rgb.dim() == 3:
                r, g, _ = rgb[0], rgb[1], rgb[2]
            elif rgb.dim() == 4:
                r, g, _ = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            
            # 估计NDVI（简化版，实际NDVI需要NIR）
            # 使用绿红比值作为植被指数近似
            vi = (g - r) / (g + r + 1e-8)
            ndvi_est = self.ndvi_base + self.ndvi_scale * torch.sigmoid(vi * 4)
            
            # 反推NIR
            nir = ndvi_est * (r + r) / (1 - ndvi_est + 1e-8)
            return torch.clamp(nir, 0, 1)
        else:
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            vi = (g - r) / (g + r + 1e-8)
            ndvi_est = self.ndvi_base + self.ndvi_scale * (1 / (1 + np.exp(-vi * 4)))
            nir = ndvi_est * (r + r) / (1 - ndvi_est + 1e-8)
            return np.clip(nir, 0, 255).astype(rgb.dtype)


class GuidedFilterNIRGenerator(BaseNIRGenerator):
    """
    引导滤波生成器 - 保持边缘的平滑生成
    
    使用RGB作为引导图，生成边缘对齐的NIR
    基于He et al. (2010) 引导滤波算法
    """
    
    def __init__(self, radius: int = 8, eps: float = 0.01):
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
            # 使用平均池化实现盒式滤波
            return F.avg_pool2d(x, kernel_size=2*r+1, stride=1, padding=r, count_include_pad=False)
        else:
            # numpy实现
            from scipy.ndimage import uniform_filter
            return uniform_filter(x, size=2*r+1)
    
    def _guided_filter(self, I, p):
        """
        引导滤波核心
        I: 引导图（亮度）
        p: 输入图（初始NIR估计）
        """
        if isinstance(I, torch.Tensor):
            r = self.radius
            eps = self.eps
            
            mean_I = self._box_filter(I, r)
            mean_p = self._box_filter(p, r)
            mean_Ip = self._box_filter(I * p, r)
            cov_Ip = mean_Ip - mean_I * mean_p
            
            mean_II = self._box_filter(I * I, r)
            var_I = mean_II - mean_I * mean_I
            
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            
            mean_a = self._box_filter(a, r)
            mean_b = self._box_filter(b, r)
            
            q = mean_a * I + mean_b
            return q
        else:
            # 简化版：直接使用高斯滤波
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(p, sigma=self.radius/2)
    
    def __call__(self, rgb):
        # 先生成初始NIR估计
        if isinstance(rgb, torch.Tensor):
            # 计算亮度作为引导图
            if rgb.dim() == 3:  # [3, H, W]
                I = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                p = self.physical_gen(rgb)  # [H, W]
                
                # 高斯平滑
                p_smooth = TVF.gaussian_blur(p.unsqueeze(0), 
                                             kernel_size=self.radius*2+1, 
                                             sigma=self.radius/2).squeeze(0)
                
                # 使用梯度检测边缘
                grad_x = torch.abs(p[:, 1:] - p[:, :-1])
                grad_y = torch.abs(p[1:, :] - p[:-1, :])
                edges = torch.zeros_like(p)
                edges[:, :-1] += grad_x
                edges[:-1, :] += grad_y
                
                # 边缘区域减少平滑
                edge_weight = torch.sigmoid(-edges * 5)  # 边缘处权重低
                
                # 边缘加权融合
                nir = edge_weight * p + (1 - edge_weight) * p_smooth
                
            else:  # [B, 3, H, W]
                I = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]  # [B, H, W]
                p = self.physical_gen(rgb)  # [B, H, W]
                
                # 高斯平滑
                p_smooth = TVF.gaussian_blur(p.unsqueeze(1), 
                                            kernel_size=self.radius*2+1, 
                                            sigma=self.radius/2).squeeze(1)
                
                # 使用梯度检测边缘
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
            # numpy版本
            p = self.physical_gen(rgb)
            from scipy.ndimage import gaussian_filter
            p_smooth = gaussian_filter(p, sigma=self.radius/2)
            return (0.7 * p + 0.3 * p_smooth).astype(rgb.dtype)


class ContextAwareNIRGenerator(BaseNIRGenerator):
    """
    上下文感知生成器 - 根据场景类型自适应调整
    
    分析图像内容（云、植被、水体），分别使用不同策略
    """
    
    def __init__(self):
        self.cloud_weights = np.array([0.7, 0.2, 0.1])   # 云：偏红
        self.veg_weights = np.array([0.3, 0.6, 0.1])       # 植被：偏绿
        self.water_weights = np.array([0.2, 0.3, 0.5])     # 水体：偏蓝
        self.physical_gen = PhysicalNIRGenerator()
    
    def _classify_regions(self, rgb):
        """
        简单分类：云、植被、水体
        返回分类掩码
        """
        if isinstance(rgb, torch.Tensor):
            if rgb.dim() == 3:
                r, g, b = rgb[0], rgb[1], rgb[2]
            else:
                r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            
            # 计算特征
            brightness = (r + g + b) / 3
            
            # 云：高亮度 + 白（R≈G≈B）
            is_cloud = (brightness > 0.7) & (torch.abs(r - g) < 0.1) & (torch.abs(g - b) < 0.1)
            
            # 植被：绿 > 红 > 蓝
            is_veg = (g > r) & (g > b) & (r > b)
            
            # 水体：蓝 > 绿 > 红，低亮度
            is_water = (b > g) & (g > r) & (brightness < 0.4)
            
            # 默认：其他
            is_other = ~(is_cloud | is_veg | is_water)
            
            return is_cloud.float(), is_veg.float(), is_water.float(), is_other.float()
        else:
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            brightness = (r + g + b) / 3
            
            is_cloud = (brightness > 180) & (np.abs(r - g) < 30) & (np.abs(g - b) < 30)
            is_veg = (g > r) & (g > b) & (r > b)
            is_water = (b > g) & (g > r) & (brightness < 100)
            is_other = ~(is_cloud | is_veg | is_water)
            
            return is_cloud, is_veg, is_water, is_other
    
    def __call__(self, rgb):
        if isinstance(rgb, torch.Tensor):
            if rgb.dim() == 3:
                r, g, b = rgb[0], rgb[1], rgb[2]
            else:
                r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            
            # 获取区域掩码
            mask_cloud, mask_veg, mask_water, mask_other = self._classify_regions(rgb)
            
            # 各区域使用不同权重生成NIR
            nir_cloud = self.cloud_weights[0] * r + self.cloud_weights[1] * g + self.cloud_weights[2] * b
            nir_veg = self.veg_weights[0] * r + self.veg_weights[1] * g + self.veg_weights[2] * b
            nir_water = self.water_weights[0] * r + self.water_weights[1] * g + self.water_weights[2] * b
            
            # 其他区域使用物理模型
            nir_other = self.physical_gen(rgb)
            
            # 融合
            nir = mask_cloud * nir_cloud + mask_veg * nir_veg + mask_water * nir_water + mask_other * nir_other
            
            return torch.clamp(nir, 0, 1)
        else:
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            mask_cloud, mask_veg, mask_water, mask_other = self._classify_regions(rgb)
            
            nir_cloud = (self.cloud_weights[0] * r + self.cloud_weights[1] * g + self.cloud_weights[2] * b).astype(np.float32)
            nir_veg = (self.veg_weights[0] * r + self.veg_weights[1] * g + self.veg_weights[2] * b).astype(np.float32)
            nir_water = (self.water_weights[0] * r + self.water_weights[1] * g + self.water_weights[2] * b).astype(np.float32)
            nir_other = self.physical_gen(rgb).astype(np.float32)
            
            nir = mask_cloud * nir_cloud + mask_veg * nir_veg + mask_water * nir_water + mask_other * nir_other
            
            return np.clip(nir, 0, 255).astype(rgb.dtype)


class EnsembleNIRGenerator(BaseNIRGenerator):
    """
    集成生成器 - 多种方法融合
    
    组合多个生成器的结果，通过简单平均或加权
    """
    
    def __init__(self, methods: Optional[list] = None, weights: Optional[list] = None):
        """
        Args:
            methods: 方法列表 ['physical', 'vegetation', 'guided', 'context']
            weights: 各方法权重
        """
        methods = methods or ['physical', 'vegetation', 'guided', 'context']
        
        self.generators = {}
        if 'physical' in methods:
            self.generators['physical'] = PhysicalNIRGenerator()
        if 'vegetation' in methods:
            self.generators['vegetation'] = VegetationIndexNIRGenerator()
        if 'guided' in methods:
            self.generators['guided'] = GuidedFilterNIRGenerator()
        if 'context' in methods:
            self.generators['context'] = ContextAwareNIRGenerator()
        
        self.weights = weights or [1.0] * len(self.generators)
        self.weights = np.array(self.weights) / sum(self.weights)
    
    def __call__(self, rgb):
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


# 便捷函数
def generate_pseudo_nir(
    rgb,
    method: str = 'ensemble',
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    便捷函数：从RGB生成伪NIR
    
    Args:
        rgb: RGB图像
        method: 生成方法 'physical' | 'vegetation' | 'guided' | 'context' | 'ensemble'
        **kwargs: 各方法特定参数
        
    Returns:
        nir: 伪NIR通道
    """
    generators = {
        'physical': PhysicalNIRGenerator,
        'vegetation': VegetationIndexNIRGenerator,
        'guided': GuidedFilterNIRGenerator,
        'context': ContextAwareNIRGenerator,
        'ensemble': EnsembleNIRGenerator,
    }
    
    if method not in generators:
        raise ValueError(f"Unknown method: {method}. Available: {list(generators.keys())}")
    
    gen = generators[method](**kwargs)
    return gen(rgb)
