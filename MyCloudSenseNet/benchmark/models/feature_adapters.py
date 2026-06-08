"""
特征适配器 - 简化版

根据位深度信息调制特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitDepthAdapter(nn.Module):
    """
    位深度特征适配器
    
    使用通道注意力机制，根据位深度估计调制特征。
    """
    
    def __init__(self, feature_dim: int, num_bit_depths: int = 5):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 位深度嵌入
        self.bit_depth_emb = nn.Embedding(num_bit_depths, feature_dim)
        
        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 16, feature_dim),
            nn.Sigmoid(),
        )
        
        # 特征变换
        self.transform = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, features: torch.Tensor, bit_depth_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]
            bit_depth_logits: [B, num_bit_depths]
        
        Returns:
            调制后的特征 [B, C, H, W]
        """
        B, C, H, W = features.shape
        
        # 位深度软嵌入
        probs = F.softmax(bit_depth_logits, dim=1)
        bd_emb = torch.matmul(probs, self.bit_depth_emb.weight)  # [B, C]
        
        # 通道注意力
        base_attn = self.channel_attn(features)  # [B, C]
        modulated_attn = base_attn * torch.sigmoid(bd_emb)
        modulated_attn = modulated_attn.view(B, C, 1, 1)
        
        # 应用注意力
        attended = features * modulated_attn
        
        # 特征变换 + 残差
        transformed = self.transform(attended)
        return features + 0.1 * transformed


class HardGateBitDepthAdapter(nn.Module):
    """基于离散位深度类别的硬门控适配器。"""
    
    def __init__(self, feature_dim: int, num_bit_depths: int = 5):
        super().__init__()
        self.feature_dim = feature_dim
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(feature_dim, feature_dim), nn.Sigmoid(),
        )
        self.transform = nn.Conv2d(feature_dim, feature_dim, 1, bias=False)
        self.bit_depth_bias = nn.Parameter(torch.zeros(num_bit_depths))
    
    def forward(self, features, bit_depth_logits):
        B = features.size(0)
        idx = bit_depth_logits.argmax(dim=1)
        scale = self.bit_depth_bias[idx].view(B, 1, 1, 1)
        
        attn = self.channel_gate(features).view(B, -1, 1, 1)
        attn = attn * (1 + torch.sigmoid(scale))
        
        features_attended = features * attn
        adapted = self.transform(features_attended)
        return features + 0.1 * adapted


class LightBitDepthAdapter(nn.Module):
    """低参数量位深度缩放适配器。"""

    def __init__(self, feature_dim: int, num_bit_depths: int = 5):
        super().__init__()
        self.feature_dim = feature_dim
        self.bit_depth_scale = nn.Embedding(num_bit_depths, feature_dim)
        self.bit_depth_bias = nn.Embedding(num_bit_depths, feature_dim)
        nn.init.zeros_(self.bit_depth_scale.weight)
        nn.init.zeros_(self.bit_depth_bias.weight)

    def forward(self, features, bit_depth_logits):
        idx = bit_depth_logits.argmax(dim=1)
        scale = torch.sigmoid(self.bit_depth_scale(idx)).view(features.size(0), -1, 1, 1)
        bias = self.bit_depth_bias(idx).view(features.size(0), -1, 1, 1)
        return features + 0.1 * (features * scale + bias)


class AdapterFactory:
    """适配器工厂"""
    
    @staticmethod
    def create(adapter_type: str, feature_dim: int, num_bit_depths: int = 5):
        adapter_type = {
            "feature_adapter": "hard_gate",
            "multiscale_light": "hard_gate",
        }.get(adapter_type, adapter_type)
        adapters = {
            "default": BitDepthAdapter,
            "hard_gate": HardGateBitDepthAdapter,
            "light": LightBitDepthAdapter,
        }
        if adapter_type not in adapters:
            valid = ", ".join(sorted(adapters))
            raise ValueError(f"Unknown adapter: {adapter_type}. Valid adapters: {valid}")
        return adapters[adapter_type](feature_dim, num_bit_depths)
