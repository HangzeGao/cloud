"""
位深度估计器 - 简化版

三种实现:
- minimal: 无参数, 仅统计特征
- conv: 轻量CNN (推荐)
- statistical: 统计+MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_BIT_DEPTHS = (8, 10, 12, 14, 16)
MIN_BIT_DEPTH, MAX_BIT_DEPTH = min(DEFAULT_BIT_DEPTHS), max(DEFAULT_BIT_DEPTHS)
STATISTICAL_FEATURES_PER_CHANNEL = 6


def normalize_bit_depth_values(bit_depths=None):
    if bit_depths is None:
        values = DEFAULT_BIT_DEPTHS
    elif torch.is_tensor(bit_depths):
        values = tuple(int(v) for v in bit_depths.detach().cpu().tolist())
    else:
        values = tuple(int(v) for v in bit_depths)
    if not values:
        raise ValueError("bit_depths must contain at least one value")
    if len(set(values)) != len(values):
        raise ValueError(f"bit_depths must be unique, got: {values}")
    return values


def bit_depths_to_indices(bit_depths, supported_bit_depths=None):
    supported = torch.as_tensor(
        normalize_bit_depth_values(supported_bit_depths),
        device=bit_depths.device,
        dtype=torch.float32,
    )
    values = bit_depths.to(dtype=torch.float32).view(-1, 1)
    return torch.argmin(torch.abs(values - supported.view(1, -1)), dim=1)


class MinimalEstimator(nn.Module):
    """极简统计估计器"""
    
    def __init__(self, in_channels, bit_depths=None):
        super().__init__()
        bit_depth_values = normalize_bit_depth_values(bit_depths)
        self.register_buffer('bit_depths', torch.tensor(bit_depth_values, dtype=torch.float32))
        self.register_buffer('min_bd', torch.tensor(float(min(bit_depth_values))))
        self.register_buffer('max_bd', torch.tensor(float(max(bit_depth_values))))
    
    @torch.no_grad()
    def forward(self, x):
        B = x.size(0)
        x_flat = x.view(B, -1)
        # 计算每个样本的动态范围（flatten后只有dim=1）
        dynamic_range = x_flat.max(dim=1)[0] - x_flat.min(dim=1)[0]
        
        # 启发式映射
        estimated = torch.clamp(
            self.min_bd + (dynamic_range - 0.8) * 20,
            float(self.min_bd), float(self.max_bd)
        )
        
        # 伪logits
        logits = torch.zeros(B, self.bit_depths.numel(), device=x.device)
        idx = bit_depths_to_indices(estimated, self.bit_depths)
        logits.scatter_(1, idx.unsqueeze(1), 1.0)
        
        return logits, estimated


class ConvEstimator(nn.Module):
    """轻量CNN估计器 (推荐)"""
    
    def __init__(self, in_channels, bit_depths=None):
        super().__init__()
        bit_depth_values = normalize_bit_depth_values(bit_depths)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, len(bit_depth_values)),
        )
        self.register_buffer('bit_depths', torch.tensor(bit_depth_values, dtype=torch.float32))
    
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)
        estimated = (probs * self.bit_depths.to(x.device)).sum(dim=1)
        return logits, estimated


class StatisticalEstimator(nn.Module):
    """统计特征MLP估计器"""
    
    def __init__(self, in_channels, hidden=32, bit_depths=None):
        super().__init__()
        bit_depth_values = normalize_bit_depth_values(bit_depths)
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * STATISTICAL_FEATURES_PER_CHANNEL, hidden),
            nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(hidden, len(bit_depth_values)),
        )
        self.register_buffer('bit_depths', torch.tensor(bit_depth_values, dtype=torch.float32))
        self.register_buffer('hist_bins', torch.linspace(0, 1, 65))
    
    def extract_stats(self, x):
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, -1)
        
        stats = [
            x_flat.mean(dim=2),
            x_flat.std(dim=2),
            x_flat.min(dim=2)[0],
            x_flat.max(dim=2)[0],
            (x_flat < 0.05).float().mean(dim=2),
            (x_flat > 0.95).float().mean(dim=2),
        ]
        return torch.cat(stats, dim=1)
    
    def forward(self, x):
        stats = self.extract_stats(x)
        logits = self.mlp(stats)
        probs = F.softmax(logits, dim=1)
        estimated = (probs * self.bit_depths.to(x.device)).sum(dim=1)
        return logits, estimated


class EstimatorFactory:
    """估计器工厂"""
    
    @staticmethod
    def create(estimator_type: str, in_channels, bit_depths=None):
        estimators = {
            "minimal": MinimalEstimator,
            "conv": ConvEstimator,
            "statistical": StatisticalEstimator,
        }
        if estimator_type not in estimators:
            raise ValueError(f"Unknown estimator: {estimator_type}")
        return estimators[estimator_type](in_channels, bit_depths=bit_depths)
