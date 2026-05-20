"""
FPN (Feature Pyramid Network) 特征融合
经典的金字塔特征融合方案
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNFusion(nn.Module):
    """
    FPN特征金字塔融合模块
    
    功能：
    1. 自顶向下特征融合
    2. 横向连接
    3. 输出多尺度特征
    
    Args:
        in_channels: 输入特征通道列表 [c2, c3, c4, c5]
        out_channels: 输出特征通道数
        num_levels: 输出层数
    """
    
    def __init__(
        self,
        in_channels: list,
        out_channels: int = 256,
        num_levels: int = 4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        
        # 横向连接（1x1卷积调整通道数）
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 平滑卷积（3x3消除上采样伪影）
        self.smooth_convs = nn.ModuleList()
        for _ in range(num_levels):
            self.smooth_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
    def forward(self, features: list) -> list:
        """
        Args:
            features: [c2, c3, c4, c5] 不同尺度的特征
                     c2: 1/4, c3: 1/8, c4: 1/16, c5: 1/32
        Returns:
            fused_features: [p2, p3, p4, p5] 融合后的特征
        """
        assert len(features) == len(self.in_channels)
        
        # 横向连接调整通道
        laterals = []
        for i, feat in enumerate(features):
            laterals.append(self.lateral_convs[i](feat))
        
        # 自顶向下融合
        # 从最高层（最小分辨率）开始
        for i in range(len(laterals) - 1, 0, -1):
            # 上采样
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i-1].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            # 相加融合
            laterals[i-1] = laterals[i-1] + upsampled
        
        # 平滑卷积
        fused_features = []
        for i, lateral in enumerate(laterals):
            fused_features.append(self.smooth_convs[i](lateral))
        
        return fused_features
    
    def get_out_channels(self) -> int:
        """返回输出特征通道数"""
        return self.out_channels


class FPNFusionV2(nn.Module):
    """
    改进版FPN - 增加额外的特征增强
    """
    
    def __init__(
        self,
        in_channels: list,
        out_channels: int = 256,
        num_levels: int = 4,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.use_attention = use_attention
        
        # 基础FPN
        self.fpn = FPNFusion(in_channels, out_channels, num_levels)
        
        # 可选的注意力增强
        if use_attention:
            self.attention_modules = nn.ModuleList()
            for _ in range(num_levels):
                self.attention_modules.append(
                    ChannelAttention(out_channels)
                )
    
    def forward(self, features: list) -> list:
        fused = self.fpn(features)
        
        if self.use_attention:
            enhanced = []
            for i, feat in enumerate(fused):
                enhanced.append(self.attention_modules[i](feat))
            return enhanced
        
        return fused


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out
