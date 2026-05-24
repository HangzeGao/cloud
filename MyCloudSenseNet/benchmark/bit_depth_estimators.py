"""
位深度估计器模块

包含三种位深度估计器实现，支持8-13位深度范围。
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalBitDepthEstimator(nn.Module):
    """
    极简版位深度估计器（最快，仅用于监控）
    
    只做前向传播，不参与梯度计算，用于监控和日志记录
    训练时可以完全关闭，推理时启用
    """
    
    def __init__(self, in_channels: int = 4, num_bit_depths: int = 6):
        super().__init__()
        self.num_bit_depths = num_bit_depths
        # 位深度范围 8-13
        self.register_buffer('bit_depths', torch.tensor([8, 9, 10, 11, 12, 13]).float())
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """不计算梯度，只返回估计值"""
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        
        # 只计算一个简单指标：动态范围
        dynamic_range = (x_flat.max(dim=2)[0] - x_flat.min(dim=2)[0]).mean(dim=1)  # [B]
        
        # 启发式规则：动态范围小的更可能是低比特
        # 8-bit通常<0.9, 10-bit通常>0.95, 12-bit通常接近1.0, 13-bit更接近1.0
        estimated = torch.clamp(8 + (dynamic_range - 0.8) * 20, 8, 13)
        
        # 创建伪logits（用于兼容性）
        logits = torch.zeros(B, self.num_bit_depths, device=x.device)
        # 在估计值附近设置高概率（索引范围0-5，对应8-13）
        idx = (estimated - 8).long().clamp(0, 5)
        logits.scatter_(1, idx.unsqueeze(1), 1.0)
        
        return logits, estimated


class ConvBitDepthEstimator(nn.Module):
    """
    轻量卷积位深度估计器（速度与精度的平衡）
    
    使用极轻量卷积网络
    复杂度：O(H*W)，但并行度高，实际很快
    """
    
    def __init__(self, in_channels: int = 4, num_bit_depths: int = 6):
        super().__init__()
        
        # 极轻量编码器：只下采样4次
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_bit_depths),
        )
        
        # 位深度范围 8-13
        self.register_buffer('bit_depths', torch.tensor([8, 9, 10, 11, 12, 13]).float())
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        logits = self.classifier(features)
        
        probs = F.softmax(logits, dim=1)
        bit_depths = self.bit_depths.to(x.device)
        estimated = (probs * bit_depths).sum(dim=1)
        
        return logits, estimated


class StatisticalBitDepthEstimator(nn.Module):
    """
    B1: 基于统计特征的位深度估计器（优化版）
    
    性能优化：
    - 移除昂贵的torch.quantile操作
    - 只使用O(1)复杂度的基本统计量
    - 增加直方图近似百分位数（向量化，无排序）
    """
    
    def __init__(self, in_channels: int = 4, hidden_dim: int = 32, num_bit_depths: int = 6):
        super().__init__()
        self.in_channels = in_channels
        self.num_bit_depths = num_bit_depths

        stat_dim = in_channels * 10

        self.mlp = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_bit_depths),
        )

        # 位深度值映射 (8-13)
        self.register_buffer('bit_depths', torch.tensor([8, 9, 10, 11, 12, 13]).float())
        self.register_buffer('hist_bins', torch.linspace(0, 1, 65))
        
    def extract_statistics(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        
        mean = x_flat.mean(dim=2)
        std = x_flat.std(dim=2)
        min_val = x_flat.min(dim=2)[0]
        max_val = x_flat.max(dim=2)[0]
        range_val = max_val - min_val
        
        low_density = (x_flat < 0.05).float().mean(dim=2)
        high_density = (x_flat > 0.95).float().mean(dim=2)
        mean_neighborhood = (torch.abs(x_flat - mean.unsqueeze(2)) < 0.1).float().mean(dim=2)
        
        x_expanded = x_flat.unsqueeze(-1)
        bins_expanded = self.hist_bins.view(1, 1, 1, -1).to(x.device)
        
        bin_width = 1.0 / (len(self.hist_bins) - 1)
        dist_to_bins = torch.abs(x_expanded - bins_expanded) / bin_width
        weights = torch.clamp(1 - dist_to_bins, 0, 1)
        
        hist = weights.sum(dim=2)
        hist = hist / (hist.sum(dim=2, keepdim=True) + 1e-8)
        
        entropy = -(hist * torch.log(hist + 1e-8)).sum(dim=2)
        
        hist_diff = hist[:, :, 1:-1] - hist[:, :, :-2]
        hist_diff_next = hist[:, :, 2:] - hist[:, :, 1:-1]
        peak_count = ((hist_diff > 0) & (hist_diff_next < 0)).float().sum(dim=2)
        
        stats = torch.cat([
            mean, std, min_val, max_val, range_val,
            low_density, high_density, mean_neighborhood,
            entropy, peak_count
        ], dim=1)
        
        return stats
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stats = self.extract_statistics(x)
        logits = self.mlp(stats)
        
        probs = F.softmax(logits, dim=1)
        bit_depths = self.bit_depths.to(x.device)
        estimated_bit_depth = (probs * bit_depths).sum(dim=1)
        
        return logits, estimated_bit_depth
