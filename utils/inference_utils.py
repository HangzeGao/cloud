"""
推理工具模块
============

提供多种推理模式的支持，包括滑动窗口、多尺度测试、整图推理等。
支持任意尺寸输入，适用于大图像的云层分割任务。

主要功能:
- 滑动窗口推理: 分块处理大图像
- 多尺度推理 (TTA): 多尺度融合提高精度
- 整图推理: 直接处理整个图像
- Patch-based推理: 带高斯融合的patch处理
- 高斯权重图生成

使用示例:
    >>> from utils.inference_utils import sliding_window_inference
    >>> pred = sliding_window_inference(model, image, window_size=512, stride=256)
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Callable, Union


# =============================================================================
# 滑动窗口推理
# =============================================================================

def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    window_size: int = 512,
    stride: int = 256,
    num_classes: int = 2,
    batch_size: int = 4,
    mirror_padding: bool = True
) -> torch.Tensor:
    """
    滑动窗口推理 - 支持任意尺寸输入
    
    将大图像分割成多个小窗口，分别推理后融合结果。
    支持镜像填充处理边界区域。
    
    Args:
        model: 分割模型
        image: 输入图像 [C, H, W] 或 [B, C, H, W]
        window_size: 窗口大小
        stride: 滑动步长
        num_classes: 类别数
        batch_size: 批处理大小
        mirror_padding: 是否使用镜像填充
        
    Returns:
        prediction: 预测结果 [num_classes, H, W] 或 [B, num_classes, H, W]
        
    Example:
        >>> image = torch.randn(3, 1024, 1024)
        >>> pred = sliding_window_inference(model, image, window_size=512, stride=256)
        >>> pred.shape  # torch.Size([2, 1024, 1024])
    """
    # 确保输入是4维
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    B, C, H, W = image.shape
    device = image.device
    
    # 输出累加器和计数器
    output = torch.zeros(B, num_classes, H, W, device=device)
    count = torch.zeros(B, 1, H, W, device=device)
    
    # 计算需要填充的尺寸
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    # 填充
    if mirror_padding:
        image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
    else:
        image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
    H_padded, W_padded = image_padded.shape[-2:]
    
    # 生成所有窗口坐标
    windows = []
    for y in range(0, H_padded - window_size + 1, stride):
        for x in range(0, W_padded - window_size + 1, stride):
            windows.append((y, x))
    
    # 分批处理
    model.eval()
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i+batch_size]
            
            # 收集窗口图像
            batch_images = []
            for y, x in batch_windows:
                window = image_padded[:, :, y:y+window_size, x:x+window_size]
                batch_images.append(window)
            
            batch_images = torch.cat(batch_images, dim=0)
            
            # 推理
            batch_output = model(batch_images)
            if isinstance(batch_output, dict):
                batch_logits = batch_output['logits']
            else:
                batch_logits = batch_output
            
            # 累加到输出
            for j, (y, x) in enumerate(batch_windows):
                logits = batch_logits[j:j+1]
                
                # 裁剪到原始区域
                y_end = min(y + window_size, H)
                x_end = min(x + window_size, W)
                h_valid = y_end - y
                w_valid = x_end - x
                
                output[:, :, y:y_end, x:x_end] += logits[:, :, :h_valid, :w_valid]
                count[:, :, y:y_end, x:x_end] += 1
    
    # 平均
    output = output / count.clamp(min=1)
    
    return output.squeeze(0) if B == 1 else output


# =============================================================================
# 整图推理
# =============================================================================

def whole_image_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    num_classes: int = 2,
    max_size: int = 2048
) -> torch.Tensor:
    """
    整图推理 - 直接处理整个图像
    
    如果图像尺寸超过max_size，会先进行下采样，推理后再上采样回原尺寸。
    
    Args:
        model: 分割模型
        image: 输入图像 [C, H, W] 或 [B, C, H, W]
        num_classes: 类别数
        max_size: 最大尺寸限制
        
    Returns:
        prediction: 预测结果 [num_classes, H, W] 或 [B, num_classes, H, W]
        
    Example:
        >>> image = torch.randn(3, 1024, 1024)
        >>> pred = whole_image_inference(model, image, max_size=2048)
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    _, _, H, W = image.shape
    
    # 如果图像太大，进行下采样
    scale = 1.0
    if max(H, W) > max_size:
        scale = max_size / max(H, W)
        new_h, new_w = int(H * scale), int(W * scale)
        image = F.interpolate(
            image,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )
    
    # 推理
    model.eval()
    with torch.no_grad():
        output = model(image)
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
    
    # 如果进行了下采样，上采样回原尺寸
    if scale < 1.0:
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
    
    return logits.squeeze(0) if logits.shape[0] == 1 else logits


# =============================================================================
# 多尺度推理 (TTA)
# =============================================================================

def multi_scale_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    scales: List[float] = None,
    num_classes: int = 2,
    flip: bool = True
) -> torch.Tensor:
    """
    多尺度测试时增强（TTA）
    
    在多个尺度上进行推理，并结合水平翻转增强，最后融合所有结果。
    通常能提高分割精度，但会增加推理时间。
    
    Args:
        model: 分割模型
        image: 输入图像 [C, H, W] 或 [B, C, H, W]
        scales: 多尺度列表，默认 [0.5, 1.0, 1.5]
        num_classes: 类别数
        flip: 是否进行水平翻转增强
        
    Returns:
        prediction: 融合后的预测结果 [num_classes, H, W] 或 [B, num_classes, H, W]
        
    Example:
        >>> image = torch.randn(3, 1024, 1024)
        >>> pred = multi_scale_inference(model, image, scales=[0.5, 1.0, 1.5], flip=True)
    """
    if scales is None:
        scales = [0.5, 1.0, 1.5]
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    _, _, H, W = image.shape
    device = image.device
    
    # 存储所有尺度的预测
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for scale in scales:
            # 调整尺寸
            if scale != 1.0:
                new_h, new_w = int(H * scale), int(W * scale)
                scaled_image = F.interpolate(
                    image,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled_image = image
            
            # 正常方向推理
            output = model(scaled_image)
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            
            # 上采样回原尺寸
            logits = F.interpolate(
                logits,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            all_predictions.append(logits)
            
            # 水平翻转推理
            if flip:
                flipped_image = torch.flip(scaled_image, dims=[3])
                output_flipped = model(flipped_image)
                if isinstance(output_flipped, dict):
                    logits_flipped = output_flipped['logits']
                else:
                    logits_flipped = output_flipped
                
                # 翻转回来
                logits_flipped = torch.flip(logits_flipped, dims=[3])
                logits_flipped = F.interpolate(
                    logits_flipped,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
                all_predictions.append(logits_flipped)
    
    # 平均所有预测
    final_prediction = torch.mean(torch.stack(all_predictions), dim=0)
    
    return final_prediction.squeeze(0) if final_prediction.shape[0] == 1 else final_prediction


# =============================================================================
# Patch-based 推理
# =============================================================================

def patch_based_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: int = 512,
    overlap: int = 128,
    num_classes: int = 2,
    blend_mode: str = 'gaussian'
) -> torch.Tensor:
    """
    基于patch的推理，使用高斯融合减少边界效应
    
    与 sliding_window_inference 类似，但使用高斯权重进行融合，
    在重叠区域产生更平滑的过渡。
    
    Args:
        model: 分割模型
        image: 输入图像 [C, H, W] 或 [B, C, H, W]
        patch_size: patch大小
        overlap: 重叠区域大小
        num_classes: 类别数
        blend_mode: 融合模式 ('average', 'gaussian')
        
    Returns:
        prediction: 预测结果
        
    Example:
        >>> image = torch.randn(3, 1024, 1024)
        >>> pred = patch_based_inference(model, image, patch_size=512, overlap=128)
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    B, C, H, W = image.shape
    device = image.device
    
    stride = patch_size - overlap
    
    # 创建高斯权重图
    if blend_mode == 'gaussian':
        weight = create_gaussian_weight(patch_size, patch_size, sigma=overlap//4)
        weight = torch.from_numpy(weight).float().to(device)
    else:
        weight = torch.ones(patch_size, patch_size, device=device)
    
    # 输出累加器和权重累加器
    output = torch.zeros(B, num_classes, H, W, device=device)
    weight_sum = torch.zeros(B, 1, H, W, device=device)
    
    model.eval()
    with torch.no_grad():
        # 计算所有patch的位置
        y_positions = list(range(0, H - patch_size + 1, stride))
        if y_positions[-1] + patch_size < H:
            y_positions.append(H - patch_size)
        
        x_positions = list(range(0, W - patch_size + 1, stride))
        if x_positions[-1] + patch_size < W:
            x_positions.append(W - patch_size)
        
        for y in y_positions:
            for x in x_positions:
                # 提取patch
                patch = image[:, :, y:y+patch_size, x:x+patch_size]
                
                # 推理
                patch_output = model(patch)
                if isinstance(patch_output, dict):
                    patch_logits = patch_output['logits']
                else:
                    patch_logits = patch_output
                
                # 加权累加
                for c in range(num_classes):
                    output[:, c:c+1, y:y+patch_size, x:x+patch_size] += \
                        patch_logits[:, c:c+1] * weight.unsqueeze(0).unsqueeze(0)
                
                weight_sum[:, :, y:y+patch_size, x:x+patch_size] += weight.unsqueeze(0).unsqueeze(0)
    
    # 归一化
    output = output / weight_sum.clamp(min=1e-8)
    
    return output.squeeze(0) if B == 1 else output


# =============================================================================
# 权重生成工具
# =============================================================================

def create_gaussian_weight(
    height: int,
    width: int,
    sigma: float = None
) -> np.ndarray:
    """
    创建高斯权重图，中心权重高，边缘权重低
    
    用于 patch-based 推理中的平滑融合。
    
    Args:
        height: 高度
        width: 宽度
        sigma: 高斯标准差，默认自动计算为 min(height, width) / 8
        
    Returns:
        weight: 高斯权重图 [H, W]
        
    Example:
        >>> weight = create_gaussian_weight(512, 512)
        >>> weight.shape  # (512, 512)
        >>> weight.max(), weight.min()  # 中心最大，边缘接近0
    """
    if sigma is None:
        sigma = min(height, width) / 8
    
    y = np.arange(height)
    x = np.arange(width)
    y, x = np.meshgrid(y, x, indexing='ij')
    
    center_y, center_x = height / 2, width / 2
    
    weight = np.exp(-((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * sigma ** 2))
    
    return weight


def create_linear_weight(
    height: int,
    width: int,
    direction: str = 'both'
) -> np.ndarray:
    """
    创建线性渐变权重图
    
    Args:
        height: 高度
        width: 宽度
        direction: 渐变方向 ('horizontal', 'vertical', 'both')
        
    Returns:
        weight: 线性权重图 [H, W]
    """
    if direction == 'horizontal':
        weight_x = np.linspace(0, 1, width)
        weight = np.tile(weight_x, (height, 1))
    elif direction == 'vertical':
        weight_y = np.linspace(0, 1, height).reshape(-1, 1)
        weight = np.tile(weight_y, (1, width))
    else:  # both - 中心到边缘的渐变
        weight_x = 1 - np.abs(np.linspace(-1, 1, width))
        weight_y = 1 - np.abs(np.linspace(-1, 1, height)).reshape(-1, 1)
        weight = weight_y * weight_x
    
    return weight


# =============================================================================
# 推理归一化集成
# =============================================================================

from .inference_normalizer import (
    normalize_for_inference,
    normalize_with_preset,
    batch_normalize,
    create_normalizer,
    NormalizationMethod,
    PRESET_CONFIGS
)


def apply_inference_normalization(
    image: torch.Tensor,
    method: str = 'adaptive',
    **kwargs
) -> torch.Tensor:
    """
    在推理阶段应用归一化
    
    支持多种归一化方法，确保输入数据符合模型期望的分布。
    适用于推理阶段的动态归一化，与训练时的静态归一化互补。
    
    Args:
        image: 输入图像张量 [C, H, W] 或 [B, C, H, W]，值范围通常为 [0, 1] 或 [0, 255]
        method: 归一化方法
                - 'minmax': 简单Min-Max归一化
                - 'standard': Z-score标准化
                - 'robust': 稳健归一化（基于分位数，抗异常值）
                - 'adaptive': 自适应归一化（自动选择最佳方法）
                - 'windowed': 窗口化局部归一化（处理光照不均）
                - 'log_scaling': 对数缩放（高动态范围）
        **kwargs: 传递给具体归一化器的参数
        
    Returns:
        归一化后的图像张量，维度与输入相同
        
    Examples:
        >>> # 自适应归一化（推荐用于未知场景）
        >>> normalized = apply_inference_normalization(image, method='adaptive')
        >>> 
        >>> # 稳健归一化（处理有云层/阴影的图像）
        >>> normalized = apply_inference_normalization(image, method='robust',
        ...                                            lower_percentile=2, upper_percentile=98)
    """
    # 保存原始设备和形状信息
    original_device = image.device
    
    # 确保输入为4D [B, C, H, W] 便于处理
    was_3d = image.dim() == 3
    if was_3d:
        image = image.unsqueeze(0)  # [1, C, H, W]
    
    B, C, H, W = image.shape
    
    # 处理预设方法
    if method == 'preset':
        preset = kwargs.get('preset', 'satellite')
        results = []
        for b in range(B):
            img_b = image[b]  # [C, H, W]
            normalized = normalize_with_preset(img_b, preset=preset)
            if isinstance(normalized, np.ndarray):
                normalized = torch.from_numpy(normalized).to(original_device)
            results.append(normalized)
        normalized_image = torch.stack(results, dim=0)
    else:
        # 批量归一化
        use_shared = kwargs.get('shared_stats', True)
        
        if use_shared and B > 1:
            # 使用第一张图像的统计参数归一化所有图像
            normalizer = create_normalizer(method, **kwargs)
            normalizer.fit(image[0])  # 从第一张图像学习参数
            
            results = []
            for b in range(B):
                img_normalized = normalizer.transform(image[b])
                if isinstance(img_normalized, np.ndarray):
                    img_normalized = torch.from_numpy(img_normalized).to(original_device)
                results.append(img_normalized)
            normalized_image = torch.stack(results, dim=0)
        else:
            # 独立归一化每个样本
            results = []
            for b in range(B):
                img_b = image[b]  # [C, H, W]
                normalized = normalize_for_inference(img_b, method=method, **kwargs)
                if isinstance(normalized, np.ndarray):
                    normalized = torch.from_numpy(normalized).to(original_device)
                results.append(normalized)
            normalized_image = torch.stack(results, dim=0)
    
    # 恢复原始维度
    if was_3d:
        normalized_image = normalized_image.squeeze(0)
    
    return normalized_image.to(original_device)


def get_normalization_recommendation(image: torch.Tensor) -> Dict[str, Any]:
    """
    根据图像特征推荐最佳归一化方法
    
    分析图像的统计特征，自动推荐最适合的归一化策略。
    
    Args:
        image: 输入图像 [C, H, W] 或 [B, C, H, W]
        
    Returns:
        Dict包含:
        - method: 推荐的方法名称
        - params: 推荐的参数
        - reason: 推荐原因
        - features: 图像特征统计
        
    Example:
        >>> image = torch.randn(3, 512, 512)
        >>> rec = get_normalization_recommendation(image)
        >>> print(f"推荐方法: {rec['method']}, 原因: {rec['reason']}")
    """
    if image.dim() == 4:
        image = image[0]  # 取第一个样本
    
    # 计算图像特征
    image_np = image.permute(1, 2, 0).cpu().numpy() if image.dim() == 3 else image.cpu().numpy()
    
    # 亮度分析
    if image.shape[0] == 3:
        luminance = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    else:
        luminance = image.mean(dim=0)
    
    luminance_np = luminance.cpu().numpy()
    
    # 计算统计特征
    mean_val = luminance_np.mean()
    std_val = luminance_np.std()
    p1, p99 = np.percentile(luminance_np, [1, 99])
    dynamic_range = p99 - p1
    
    # 检测异常值
    outlier_low = np.sum(luminance_np < p1) / luminance_np.size
    outlier_high = np.sum(luminance_np > p99) / luminance_np.size
    outlier_ratio = outlier_low + outlier_high
    
    # 决策逻辑
    recommendation = {}
    
    if outlier_ratio > 0.05:
        # 存在明显异常值（云层、阴影等）
        recommendation = {
            'method': 'robust',
            'params': {
                'lower_percentile': 2.0,
                'upper_percentile': 98.0,
                'clip': True
            },
            'reason': f'检测到异常值占比 {outlier_ratio*100:.1f}%，建议使用稳健归一化'
        }
    elif dynamic_range < 0.1:
        # 低对比度
        recommendation = {
            'method': 'standard',
            'params': {'per_channel': True},
            'reason': f'动态范围较小 ({dynamic_range:.3f})，建议使用标准归一化增强对比度'
        }
    elif dynamic_range > 0.8:
        # 高动态范围
        recommendation = {
            'method': 'log_scaling',
            'params': {},
            'reason': f'高动态范围 ({dynamic_range:.3f})，建议使用对数缩放'
        }
    elif std_val < 0.05:
        # 光照可能不均匀
        recommendation = {
            'method': 'windowed',
            'params': {'window_size': 64, 'method': 'mean_std'},
            'reason': '光照可能不均匀，建议使用窗口化局部归一化'
        }
    else:
        # 正常情况，使用自适应
        recommendation = {
            'method': 'adaptive',
            'params': {'auto_select': True},
            'reason': '图像特征正常，建议使用自适应归一化自动选择最佳策略'
        }
    
    recommendation['features'] = {
        'mean': float(mean_val),
        'std': float(std_val),
        'dynamic_range': float(dynamic_range),
        'outlier_ratio': float(outlier_ratio)
    }
    
    return recommendation


# =============================================================================
# 工具函数
# =============================================================================

def create_inference_transform(
    use_nir: bool = False,
    nir_method: str = 'guided',
    normalization: Optional[str] = None,
    normalization_params: Optional[Dict[str, Any]] = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    创建推理变换管道
    
    组合伪NIR生成和归一化的完整变换流程。
    
    Args:
        use_nir: 是否使用NIR通道
        nir_method: 伪NIR生成方法
        normalization: 归一化方法（None表示不归一化）
        normalization_params: 归一化参数
        
    Returns:
        变换函数，接收图像张量，返回处理后的张量
    """
    def transform(image: torch.Tensor) -> torch.Tensor:
        # 步骤1: 伪NIR生成
        if use_nir and image.shape[0] == 3:
            from .nir_generator import generate_pseudo_nir
            nir = generate_pseudo_nir(image, method=nir_method)
            if nir.dim() == 2:
                nir = nir.unsqueeze(0)
            image = torch.cat([image, nir], dim=0)
        
        # 步骤2: 归一化
        if normalization:
            params = normalization_params or {}
            image = apply_inference_normalization(image, method=normalization, **params)
        
        return image
    
    return transform


class InferenceMode:
    """
    推理模式枚举
    
    定义支持的推理模式常量。
    """
    SLIDING_WINDOW = 'sliding_window'
    MULTI_SCALE = 'multi_scale'
    WHOLE_IMAGE = 'whole'
    SIMPLE = 'simple'
    PATCH_BASED = 'patch_based'


class InferenceConfig:
    """
    推理配置数据类
    
    用于组织和传递推理参数。
    
    Attributes:
        mode: 推理模式
        window_size: 窗口大小（滑动窗口模式）
        stride: 滑动步长
        scales: 多尺度列表
        max_size: 整图推理最大尺寸
        use_tta: 是否使用测试时增强
        normalization: 归一化方法
    """
    
    def __init__(
        self,
        mode: str = InferenceMode.SLIDING_WINDOW,
        window_size: int = 512,
        stride: int = 256,
        scales: Optional[List[float]] = None,
        max_size: int = 2048,
        use_tta: bool = False,
        normalization: Optional[str] = None
    ):
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.scales = scales or [0.5, 1.0, 1.5]
        self.max_size = max_size
        self.use_tta = use_tta
        self.normalization = normalization
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'mode': self.mode,
            'window_size': self.window_size,
            'stride': self.stride,
            'scales': self.scales,
            'max_size': self.max_size,
            'use_tta': self.use_tta,
            'normalization': self.normalization
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'InferenceConfig':
        """从字典创建"""
        return cls(**config)
