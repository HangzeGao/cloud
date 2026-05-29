"""
空间-光谱注意力机制 (Spatial-Spectral Attention, SSA)

针对卫星影像设计，同时关注空间位置和光谱通道间的复杂关系。
特别适用于云检测任务，能够有效区分云层、阴影和地表。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class SpectralAttention(nn.Module):
    """
    光谱注意力模块
    
    学习通道间的关系，增强对云/阴影敏感的光谱波段响应。
    基于 SE-Net (Squeeze-and-Excitation) 思想，但针对多光谱卫星影像优化。
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        use_max_pool: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.use_max_pool = use_max_pool
        
        mid_channels = max(channels // reduction, 4)
        
        # 全局池化聚合
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        if use_max_pool:
            self.global_max = nn.AdaptiveMaxPool2d(1)
        
        # MLP 生成注意力权重
        self.mlp = nn.Sequential(
            nn.Linear(channels * (2 if use_max_pool else 1), mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(mid_channels, channels),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            光谱注意力权重 [B, C, 1, 1]
        """
        B, C, H, W = x.shape
        
        # 全局统计
        avg_feat = self.global_avg(x).view(B, C)
        
        if self.use_max_pool:
            max_feat = self.global_max(x).view(B, C)
            combined = torch.cat([avg_feat, max_feat], dim=1)
        else:
            combined = avg_feat
        
        # 生成注意力权重
        attn = self.mlp(combined).view(B, C, 1, 1)
        
        return attn


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    
    学习空间位置的重要性，关注云的形态和边界。
    使用大卷积核捕获大范围空间上下文。
    """
    
    def __init__(
        self,
        kernel_size: int = 7,
        use_avg_pool: bool = True,
        use_max_pool: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.use_avg_pool = use_avg_pool
        self.use_max_pool = use_max_pool
        
        # 通道压缩卷积
        in_channels = (1 if use_avg_pool else 0) + (1 if use_max_pool else 0)
        assert in_channels > 0, "At least one pooling method should be enabled"
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size, padding=self.padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            空间注意力权重 [B, 1, H, W]
        """
        # 通道维度池化
        features = []
        
        if self.use_avg_pool:
            features.append(torch.mean(x, dim=1, keepdim=True))
        if self.use_max_pool:
            features.append(torch.max(x, dim=1, keepdim=True)[0])
        
        combined = torch.cat(features, dim=1)
        
        # 生成空间注意力图
        attn = self.conv(combined)
        
        return attn


class BitDepthConditionedAttention(nn.Module):
    """
    位深度条件注意力
    
    将位深度信息融入注意力机制，处理不同位深度输入时的注意力调整。
    """
    
    def __init__(
        self,
        channels: int,
        num_bit_depths: int,
        attention_type: str = "both",  # spectral, spatial, both
    ):
        super().__init__()
        self.attention_type = attention_type
        self.num_bit_depths = num_bit_depths
        
        # 位深度嵌入
        self.bit_depth_emb = nn.Embedding(num_bit_depths, channels)
        
        # 位深度到调制参数的映射
        self.bit_depth_to_spectral = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, channels),
        )
        
        self.bit_depth_to_spatial = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 16, 1),
        )
        
        if attention_type in ("spectral", "both"):
            self.spectral_attn = SpectralAttention(channels, reduction=16)
        
        if attention_type in ("spatial", "both"):
            self.spatial_attn = SpatialAttention(kernel_size=7)
    
    def forward(
        self,
        x: torch.Tensor,
        bit_depth_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
            bit_depth_logits: 位深度 logits [B, num_bit_depths]
        
        Returns:
            调制后的特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 位深度软嵌入
        probs = F.softmax(bit_depth_logits, dim=1)
        bit_depth_idx = torch.arange(self.num_bit_depths, device=x.device)
        bit_depth_emb = self.bit_depth_emb(bit_depth_idx)  # [num_bit_depths, C]
        weighted_emb = torch.matmul(probs, bit_depth_emb)  # [B, C]
        
        # 光谱调制
        if self.attention_type in ("spectral", "both"):
            spectral_weight = self.spectral_attn(x)
            spectral_mod = torch.sigmoid(self.bit_depth_to_spectral(weighted_emb))
            spectral_mod = spectral_mod.view(B, C, 1, 1)
            x = x * spectral_weight * (1 + spectral_mod)
        
        # 空间调制
        if self.attention_type in ("spatial", "both"):
            spatial_weight = self.spatial_attn(x)
            spatial_mod = torch.sigmoid(self.bit_depth_to_spatial(weighted_emb))
            spatial_mod = spatial_mod.view(B, 1, 1, 1)
            x = x * spatial_weight * (1 + spatial_mod)
        
        return x


class SpatialSpectralAttention(nn.Module):
    """
    空间-光谱联合注意力 (SSA)
    
    同时建模空间位置和光谱通道的依赖关系，
    特别针对卫星影像的多光谱特性设计。
    
    Architecture:
        Input Feature
              ↓
        ┌──────┴──────┐
        ↓             ↓
    [Spectral]   [Spatial]
    Attention    Attention
        ↓             ↓
        └──────┬──────┘
               ↓
        [Fusion & Modulation]
               ↓
        Output Feature
    """
    
    def __init__(
        self,
        channels: int,
        num_bands: int = 4,
        reduction: int = 8,
        spatial_kernel: int = 7,
        bit_depth_dim: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.num_bands = num_bands
        self.reduction = reduction
        
        # 光谱注意力
        self.spectral_attn = SpectralAttention(
            channels=channels,
            reduction=reduction,
            use_max_pool=True,
        )
        
        # 空间注意力
        self.spatial_attn = SpatialAttention(
            kernel_size=spatial_kernel,
            use_avg_pool=True,
            use_max_pool=True,
        )
        
        # 光谱上下文编码（模拟波段间关系）
        if channels >= num_bands * 8:
            self.band_interaction = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, 1),
                nn.Sigmoid(),
            )
        else:
            self.band_interaction = None
        
        # 位深度条件调制（可选）
        if bit_depth_dim is not None:
            self.bit_depth_proj = nn.Sequential(
                nn.Linear(bit_depth_dim, channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels),
                nn.Sigmoid(),
            )
        else:
            self.bit_depth_proj = None
        
        # 融合门控
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, 2, 1),
            nn.Softmax(dim=1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        bit_depth_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
            bit_depth_probs: 位深度概率分布 [B, num_bit_depths] (可选)
        
        Returns:
            注意力增强的特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 光谱注意力
        spectral_weight = self.spectral_attn(x)  # [B, C, 1, 1]
        
        # 空间注意力
        spatial_weight = self.spatial_attn(x)  # [B, 1, H, W]
        
        # 位深度调制（如果提供）
        if bit_depth_probs is not None and self.bit_depth_proj is not None:
            bd_mod = self.bit_depth_proj(bit_depth_probs).view(B, C, 1, 1)
            spectral_weight = spectral_weight * (1 + bd_mod)
        
        # 联合应用注意力
        spectral_attended = x * spectral_weight
        spatial_attended = x * spatial_weight
        
        # 自适应融合门控
        concat = torch.cat([spectral_attended, spatial_attended], dim=1)
        gates = self.fusion_gate(concat)  # [B, 2, H, W]
        
        spectral_gate = gates[:, 0:1, :, :]
        spatial_gate = gates[:, 1:2, :, :]
        
        # 门控融合
        fused = spectral_attended * spectral_gate + spatial_attended * spatial_gate
        
        # 光谱间交互（如果启用）
        if self.band_interaction is not None:
            band_context = self.band_interaction(fused)
            fused = fused * band_context
        
        # 残差连接
        return x + 0.1 * fused


class BandSpecificAttention(nn.Module):
    """
    波段特异性注意力
    
    为每个输入光谱波段学习特定的空间注意力模式。
    适用于已知波段顺序（如 B02, B03, B04, B08）的场景。
    """
    
    def __init__(
        self,
        band_channels: List[int],
        kernel_size: int = 5,
    ):
        super().__init__()
        self.band_channels = band_channels
        self.num_bands = len(band_channels)
        
        # 每个波段的空间注意力
        self.band_attns = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 1, kernel_size, padding=kernel_size // 2),
                nn.Sigmoid(),
            )
            for ch in band_channels
        ])
        
        # 跨波段融合
        total_channels = sum(band_channels)
        self.cross_band_fusion = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // 4, total_channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]，C = sum(band_channels)
        
        Returns:
            波段调制后的特征 [B, C, H, W]
        """
        # 按波段分割
        band_features = []
        start = 0
        for ch in self.band_channels:
            band_features.append(x[:, start:start+ch, :, :])
            start += ch
        
        # 应用波段特异性注意力
        attended_bands = []
        for feat, attn in zip(band_features, self.band_attns):
            band_weight = attn(feat)
            attended_bands.append(feat * band_weight)
        
        # 融合
        fused = torch.cat(attended_bands, dim=1)
        
        # 跨波段交互
        cross_band_weight = self.cross_band_fusion(fused)
        
        return fused * cross_band_weight


# 实用函数

def create_ssa_modules(
    encoder_channels: List[int],
    num_bands: int = 4,
    reduction: int = 8,
    enable_bit_depth: bool = True,
    num_bit_depths: int = 5,
) -> nn.ModuleList:
    """
    为编码器的每个尺度创建 SSA 模块
    
    Args:
        encoder_channels: 每个尺度的通道数列表
        num_bands: 输入波段数
        reduction: 注意力降维比例
        enable_bit_depth: 是否启用位深度调制
        num_bit_depths: 位深度类别数
    
    Returns:
        SSA 模块列表
    """
    ssa_modules = nn.ModuleList()
    
    for ch in encoder_channels:
        ssa = SpatialSpectralAttention(
            channels=ch,
            num_bands=num_bands,
            reduction=reduction,
            spatial_kernel=7,
            bit_depth_dim=num_bit_depths if enable_bit_depth else None,
        )
        ssa_modules.append(ssa)
    
    return ssa_modules


if __name__ == "__main__":
    # 测试 SSA 模块
    print("Testing SpatialSpectralAttention...")
    
    batch_size = 2
    channels = 64
    height, width = 32, 32
    num_bit_depths = 5
    
    # 创建模块
    ssa = SpatialSpectralAttention(
        channels=channels,
        num_bands=4,
        reduction=8,
        spatial_kernel=7,
        bit_depth_dim=num_bit_depths,
    )
    
    # 模拟输入
    x = torch.randn(batch_size, channels, height, width)
    bit_depth_probs = torch.softmax(torch.randn(batch_size, num_bit_depths), dim=1)
    
    # 前向传播
    output = ssa(x, bit_depth_probs)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape should match input"
    print("✓ SpatialSpectralAttention test passed!")
    
    # 测试创建多个 SSA 模块
    print("\nTesting create_ssa_modules...")
    encoder_channels = [32, 24, 40, 112, 320]
    ssa_modules = create_ssa_modules(encoder_channels, num_bands=4)
    
    features = [torch.randn(batch_size, ch, 256 // (2 ** i), 256 // (2 ** i))
                for i, ch in enumerate(encoder_channels)]
    
    for i, (ssa, feat) in enumerate(zip(ssa_modules, features)):
        out = ssa(feat)
        print(f"  Scale {i}: {feat.shape} -> {out.shape}")
    
    print("✓ create_ssa_modules test passed!")
