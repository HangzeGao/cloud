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


MIN_BIT_DEPTH, MAX_BIT_DEPTH = 8, 12


def get_num_bit_depths():
    return MAX_BIT_DEPTH - MIN_BIT_DEPTH + 1


class MinimalEstimator(nn.Module):
    """极简统计估计器"""
    
    def __init__(self, in_channels=4):
        super().__init__()
        self.register_buffer('min_bd', torch.tensor(float(MIN_BIT_DEPTH)))
    
    @torch.no_grad()
    def forward(self, x):
        B = x.size(0)
        x_flat = x.view(B, -1)
        # 计算每个样本的动态范围（flatten后只有dim=1）
        dynamic_range = x_flat.max(dim=1)[0] - x_flat.min(dim=1)[0]
        
        # 启发式映射
        estimated = torch.clamp(
            self.min_bd + (dynamic_range - 0.8) * 20,
            float(MIN_BIT_DEPTH), float(MAX_BIT_DEPTH)
        )
        
        # 伪logits
        logits = torch.zeros(B, get_num_bit_depths(), device=x.device)
        idx = (estimated - float(MIN_BIT_DEPTH)).long().clamp(0, get_num_bit_depths() - 1)
        logits.scatter_(1, idx.unsqueeze(1), 1.0)
        
        return logits, estimated


class ConvEstimator(nn.Module):
    """轻量CNN估计器 (推荐)"""
    
    def __init__(self, in_channels=4):
        super().__init__()
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
            nn.Linear(32, get_num_bit_depths()),
        )
        self.register_buffer('bit_depths', torch.arange(MIN_BIT_DEPTH, MAX_BIT_DEPTH + 1).float())
    
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)
        estimated = (probs * self.bit_depths.to(x.device)).sum(dim=1)
        return logits, estimated


class StatisticalEstimator(nn.Module):
    """统计特征MLP估计器"""
    
    def __init__(self, in_channels=4, hidden=32):
        super().__init__()
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 10, hidden),
            nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(hidden, get_num_bit_depths()),
        )
        self.register_buffer('bit_depths', torch.arange(MIN_BIT_DEPTH, MAX_BIT_DEPTH + 1).float())
        self.register_buffer('hist_bins', torch.linspace(0, 1, 65))
    
    def extract_stats(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        
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
    def create(estimator_type: str, in_channels=4):
        estimators = {
            "minimal": MinimalEstimator,
            "conv": ConvEstimator,
            "statistical": StatisticalEstimator,
        }
        if estimator_type not in estimators:
            raise ValueError(f"Unknown estimator: {estimator_type}")
        return estimators[estimator_type](in_channels)
