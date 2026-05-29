"""
多尺度位深度感知特征融合模块 (MS-BDFF)

在不同分辨率层级注入位深度信息，实现更全面的特征调制。
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossScaleAttention(nn.Module):
    """
    跨尺度注意力模块
    实现高层特征与低层特征的有效融合
    """
    
    def __init__(self, low_channels: int, high_channels: int):
        super().__init__()
        self.low_channels = low_channels
        self.high_channels = high_channels
        
        # 高层特征下采样到与低层对齐
        self.high_to_low = nn.Sequential(
            nn.Conv2d(high_channels, low_channels, 1),
            nn.BatchNorm2d(low_channels),
        )
        
        # 注意力生成
        self.attention = nn.Sequential(
            nn.Conv2d(low_channels * 2, low_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_channels // 4, low_channels, 1),
            nn.Sigmoid(),
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(low_channels * 2, low_channels, 3, padding=1),
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, low_feat: torch.Tensor, high_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            low_feat: 低层特征 [B, C_low, H, W]
            high_feat: 高层特征 [B, C_high, H', W'] (已上采样到 low_feat 尺寸)
        
        Returns:
            融合后的特征 [B, C_low, H, W]
        """
        # 对齐通道数
        high_aligned = self.high_to_low(high_feat)
        
        # 生成注意力权重
        concat = torch.cat([low_feat, high_aligned], dim=1)
        attn = self.attention(concat)
        
        # 加权融合
        weighted_high = high_aligned * attn
        fused = torch.cat([low_feat, weighted_high], dim=1)
        
        return self.fusion(fused)


class ScaleAwareBitDepthModulator(nn.Module):
    """
    尺度感知位深度调制器
    为每个尺度生成专属的位深度调制信号
    """
    
    def __init__(
        self,
        channels: int,
        num_bit_depths: int,
        reduction: int = 16,
    ):
        super().__init__()
        
        # 位深度嵌入
        self.bit_depth_embedding = nn.Embedding(num_bit_depths, channels // reduction)
        
        # 轻量级调制网络
        self.modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        
        # 位深度条件投影
        self.condition_proj = nn.Linear(channels // reduction * 2, channels)
    
    def forward(
        self,
        feature: torch.Tensor,
        bit_depth_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feature: 特征图 [B, C, H, W]
            bit_depth_logits: 位深度 logits [B, num_bit_depths]
        
        Returns:
            调制后的特征 [B, C, H, W]
        """
        B, C, H, W = feature.shape
        
        # 基础调制信号
        base_mod = self.modulator(feature).view(B, C, 1, 1)
        
        # 位深度软嵌入
        probs = F.softmax(bit_depth_logits, dim=1)  # [B, num_bit_depths]
        bit_depth_idx = torch.arange(bit_depth_logits.size(1), device=feature.device)
        bit_depth_emb = self.bit_depth_embedding(bit_depth_idx)  # [num_bit_depths, C//reduction]
        
        # 加权嵌入 [B, C//reduction]
        weighted_emb = torch.matmul(probs, bit_depth_emb)
        
        # 结合空间统计与位深度信息
        spatial_stat = F.adaptive_avg_pool2d(feature, 1).view(B, C)
        spatial_emb = torch.sigmoid(nn.Linear(C, C // reduction).to(feature.device)(spatial_stat))
        
        combined = torch.cat([weighted_emb, spatial_emb], dim=1)
        condition = torch.sigmoid(self.condition_proj(combined)).view(B, C, 1, 1)
        
        # 应用调制
        modulated = feature * base_mod * condition
        
        # 残差连接
        return feature + 0.1 * modulated


class MultiscaleBitDepthFusion(nn.Module):
    """
    多尺度位深度感知特征融合模块 (MS-BDFF)
    
    核心组件：
    1. 为每个尺度配置专属的位深度调制器
    2. 跨尺度特征融合（自顶向下）
    3. 最终特征聚合
    """
    
    def __init__(
        self,
        encoder_channels: List[int],
        num_bit_depths: int,
        fusion_type: str = "attention",  # attention, sum, gate
        enable_residual: bool = True,
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.num_scales = len(encoder_channels)
        self.fusion_type = fusion_type
        self.enable_residual = enable_residual
        
        # 每个尺度的位深度调制器
        self.scale_modulators = nn.ModuleList([
            ScaleAwareBitDepthModulator(ch, num_bit_depths)
            for ch in encoder_channels
        ])
        
        # 跨尺度融合模块（自顶向下）
        if fusion_type == "attention":
            self.cross_scale_fusion = nn.ModuleList([
                CrossScaleAttention(encoder_channels[i], encoder_channels[i + 1])
                for i in range(len(encoder_channels) - 1)
            ])
        elif fusion_type == "gate":
            self.cross_scale_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(encoder_channels[i + 1], encoder_channels[i], 1),
                    nn.Sigmoid(),
                )
                for i in range(len(encoder_channels) - 1)
            ])
        
        # 特征聚合（用于 SegFormer 等需要聚合多尺度特征的架构）
        self.aggregate_features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, encoder_channels[-1], 1),
                nn.BatchNorm2d(encoder_channels[-1]),
                nn.ReLU(inplace=True),
            )
            for ch in encoder_channels[:-1]
        ])
    
    def forward(
        self,
        features: List[torch.Tensor],
        bit_depth_logits: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            features: 多尺度特征列表 [feat_s0, feat_s1, ..., feat_sn]
                     其中 feat_s0 分辨率最高
            bit_depth_logits: 位深度 logits [B, num_bit_depths]
        
        Returns:
            调制并融合后的多尺度特征列表
        """
        # 1. 多尺度位深度调制
        modulated = []
        for i, (feat, modulator) in enumerate(zip(features, self.scale_modulators)):
            mod_feat = modulator(feat, bit_depth_logits)
            modulated.append(mod_feat)
        
        # 2. 跨尺度融合（自顶向下）
        if self.fusion_type == "attention":
            fused = modulated.copy()
            for i in range(len(modulated) - 2, -1, -1):
                # 上采样高层特征
                high_upsampled = F.interpolate(
                    fused[i + 1],
                    size=modulated[i].shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
                fused[i] = self.cross_scale_fusion[i](modulated[i], high_upsampled)
        
        elif self.fusion_type == "gate":
            fused = modulated.copy()
            for i in range(len(modulated) - 2, -1, -1):
                high_upsampled = F.interpolate(
                    modulated[i + 1],
                    size=modulated[i].shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
                gate = self.cross_scale_gates[i](high_upsampled)
                fused[i] = modulated[i] * gate + high_upsampled * (1 - gate)
        
        elif self.fusion_type == "sum":
            fused = modulated.copy()
            for i in range(len(modulated) - 2, -1, -1):
                high_upsampled = F.interpolate(
                    modulated[i + 1],
                    size=modulated[i].shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
                fused[i] = modulated[i] + 0.5 * high_upsampled
        
        else:
            fused = modulated
        
        # 3. 如果需要残差连接
        if self.enable_residual:
            for i in range(len(fused)):
                if fused[i].shape == features[i].shape:
                    fused[i] = features[i] + 0.1 * fused[i]
        
        return fused
    
    def aggregate_to_single_scale(
        self,
        features: List[torch.Tensor],
        target_scale: int = -1,
    ) -> torch.Tensor:
        """
        将多尺度特征聚合到单一尺度
        
        Args:
            features: 多尺度特征列表
            target_scale: 目标尺度索引，-1 表示最高层（最低分辨率）
        
        Returns:
            聚合后的特征 [B, C_agg, H, W]
        """
        if target_scale == -1:
            target_scale = len(features) - 1
        
        target_h, target_w = features[target_scale].shape[2:]
        
        # 将所有特征对齐到目标尺度
        aligned_features = []
        for i, (feat, proj) in enumerate(zip(features, self.aggregate_features)):
            if i == target_scale:
                aligned_features.append(feat)
            elif i < len(self.aggregate_features):
                # 下采样并投影
                downsampled = F.adaptive_avg_pool2d(feat, (target_h, target_w))
                aligned_features.append(proj(downsampled))
            else:
                # 最高层直接取
                aligned_features.append(feat)
        
        # 拼接聚合
        return torch.cat(aligned_features, dim=1)


class MultiscaleFeatureAdapterWrapper(nn.Module):
    """
    多尺度特征适配器包装器
    将单尺度特征适配器扩展为多尺度版本
    """
    
    def __init__(
        self,
        base_adapter: nn.Module,
        encoder_channels: List[int],
        num_bit_depths: int,
        enable_ssa: bool = True,
    ):
        super().__init__()
        self.multiscale_fusion = MultiscaleBitDepthFusion(
            encoder_channels=encoder_channels,
            num_bit_depths=num_bit_depths,
            fusion_type="attention",
        )
        
        # 可选的空间-光谱注意力
        if enable_ssa:
            from benchmark.models.spatial_spectral_attention import SpatialSpectralAttention
            self.ssa_modules = nn.ModuleList([
                SpatialSpectralAttention(ch, num_bands=4)
                for ch in encoder_channels
            ])
        else:
            self.ssa_modules = None
        
        # 单尺度适配器（用于最后一层）
        self.base_adapter = base_adapter
    
    def forward(
        self,
        features: torch.Tensor | List[torch.Tensor],
        bit_depth_logits: torch.Tensor,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        前向传播
        
        Args:
            features: 单尺度或列表形式的特征
            bit_depth_logits: 位深度 logits
        
        Returns:
            适配后的特征
        """
        # 统一为列表形式
        if isinstance(features, torch.Tensor):
            feature_list = [features]
            return_single = True
        else:
            feature_list = features
            return_single = False
        
        # 多尺度调制
        modulated = self.multiscale_fusion(feature_list, bit_depth_logits)
        
        # 应用空间-光谱注意力（如果启用）
        if self.ssa_modules is not None:
            modulated = [
                ssa(feat, F.softmax(bit_depth_logits, dim=1))
                for ssa, feat in zip(self.ssa_modules, modulated)
            ]
        
        # 对最后一层应用基础适配器
        if self.base_adapter is not None:
            modulated[-1] = self.base_adapter(modulated[-1], bit_depth_logits)
        
        if return_single:
            return modulated[0]
        return modulated


if __name__ == "__main__":
    # 测试多尺度融合
    print("Testing MultiscaleBitDepthFusion...")
    
    # 模拟 5 层编码器特征 (UNet with EfficientNet-B0)
    batch_size = 2
    encoder_channels = [32, 24, 40, 112, 320]  # EfficientNet-B0 stages
    feature_shapes = [(batch_size, ch, 256 // (2 ** i), 256 // (2 ** i))
                      for i, ch in enumerate(encoder_channels)]
    
    features = [torch.randn(*shape) for shape in feature_shapes]
    bit_depth_logits = torch.randn(batch_size, 5)  # 8, 9, 10, 11, 12 bits
    
    # 创建模块
    fusion = MultiscaleBitDepthFusion(
        encoder_channels=encoder_channels,
        num_bit_depths=5,
        fusion_type="attention",
    )
    
    # 前向传播
    output = fusion(features, bit_depth_logits)
    
    print(f"Input scales: {[f.shape for f in features]}")
    print(f"Output scales: {[f.shape for f in output]}")
    print("✓ MultiscaleBitDepthFusion test passed!")
