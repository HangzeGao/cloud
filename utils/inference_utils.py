"""
任意尺寸推理工具
支持滑动窗口、整图推理、多尺度测试等
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


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
    """
    # 确保输入是4维
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    B, C, H, W = image.shape
    device = image.device
    
    # 输出尺寸
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


def whole_image_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    num_classes: int = 2,
    max_size: int = 2048
) -> torch.Tensor:
    """
    整图推理 - 直接处理整个图像
    如果图像太大，会先进行下采样
    
    Args:
        model: 分割模型
        image: 输入图像 [C, H, W] 或 [B, C, H, W]
        num_classes: 类别数
        max_size: 最大尺寸限制
        
    Returns:
        prediction: 预测结果
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    _, _, H, W = image.shape
    
    # 如果图像太大，进行下采样
    scale = 1.0
    if max(H, W) > max_size:
        scale = max_size / max(H, W)
        new_h, new_w = int(H * scale), int(W * scale)
        image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
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
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
    
    return logits.squeeze(0) if logits.shape[0] == 1 else logits


def multi_scale_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    scales: List[float] = [0.5, 1.0, 1.5],
    num_classes: int = 2,
    flip: bool = True
) -> torch.Tensor:
    """
    多尺度测试时增强（TTA）
    
    Args:
        model: 分割模型
        image: 输入图像 [C, H, W] 或 [B, C, H, W]
        scales: 多尺度列表
        num_classes: 类别数
        flip: 是否进行水平翻转增强
        
    Returns:
        prediction: 融合后的预测结果
    """
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
                scaled_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                scaled_image = image
            
            # 正常方向推理
            output = model(scaled_image)
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            
            # 上采样回原尺寸
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
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
                logits_flipped = F.interpolate(logits_flipped, size=(H, W), mode='bilinear', align_corners=False)
                all_predictions.append(logits_flipped)
    
    # 平均所有预测
    final_prediction = torch.mean(torch.stack(all_predictions), dim=0)
    
    return final_prediction.squeeze(0) if final_prediction.shape[0] == 1 else final_prediction


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
    
    Args:
        model: 分割模型
        image: 输入图像 [C, H, W] 或 [B, C, H, W]
        patch_size: patch大小
        overlap: 重叠区域大小
        num_classes: 类别数
        blend_mode: 融合模式 ('average', 'gaussian')
        
    Returns:
        prediction: 预测结果
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


def create_gaussian_weight(height: int, width: int, sigma: float = None) -> np.ndarray:
    """
    创建高斯权重图，中心权重高，边缘权重低
    
    Args:
        height: 高度
        width: 宽度
        sigma: 高斯标准差
        
    Returns:
        weight: 高斯权重图 [H, W]
    """
    if sigma is None:
        sigma = min(height, width) / 8
    
    y = np.arange(height)
    x = np.arange(width)
    y, x = np.meshgrid(y, x, indexing='ij')
    
    center_y, center_x = height / 2, width / 2
    
    weight = np.exp(-((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * sigma ** 2))
    
    return weight
