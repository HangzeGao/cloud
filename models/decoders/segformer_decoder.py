"""
SegFormer Decoder - 轻量高效的MLP解码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_decoder import BaseDecoder


class SegFormerDecoder(BaseDecoder):
    """
    SegFormer MLP Decoder
    
    特点：
    1. 轻量级的MLP架构
    2. 统一融合多尺度特征
    3. 无需复杂的上采样操作
    
    Args:
        encoder_channels: 编码器通道数 [c2, c3, c4, c5]
        num_classes: 输出类别数
        embed_dim: 融合后的嵌入维度
    """
    
    def __init__(
        self,
        encoder_channels: list,
        num_classes: int = 2,
        embed_dim: int = 256
    ):
        super().__init__(encoder_channels, num_classes)
        
        self.embed_dim = embed_dim
        
        # 将所有编码器特征投影到统一维度
        self.linear_c4 = MLP(input_dim=encoder_channels[3], embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=encoder_channels[2], embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=encoder_channels[1], embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=encoder_channels[0], embed_dim=embed_dim)
        
        # 融合后的线性层
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        #  dropout
        self.dropout = nn.Dropout(0.1)
        
        # 分类头
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
    
    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: [c1, c2, c3, c4] 注意：输入顺序为从浅到深
        Returns:
            output: [B, num_classes, H, W]
        """
        c1, c2, c3, c4 = features
        
        # 获取目标尺寸（c1的尺寸，即1/4原图）
        target_size = c1.shape[-2:]
        
        # 投影并上采样到统一尺寸
        _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(_c4, size=target_size, mode='bilinear', align_corners=False)
        
        _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(_c3, size=target_size, mode='bilinear', align_corners=False)
        
        _c2 = self.linear_c2(c2)
        _c2 = F.interpolate(_c2, size=target_size, mode='bilinear', align_corners=False)
        
        _c1 = self.linear_c1(c1)
        # c1已经在目标尺寸
        
        # 拼接并融合
        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.linear_fuse(_c)
        _c = self.dropout(_c)
        
        # 预测
        output = self.linear_pred(_c)
        
        # 注意：输出尺寸是 target_size（c1尺寸），由主模型统一上采样到原图尺寸
        # 修复：移除固定scale_factor=4的上采样，因为c1尺寸不一定是1/4原图
        
        return output


class MLP(nn.Module):
    """
    MLP模块用于特征投影
    """
    
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SegFormerDecoderV2(BaseDecoder):
    """
    SegFormer 改进版 - 增加注意力机制
    """
    
    def __init__(
        self,
        encoder_channels: list,
        num_classes: int = 2,
        embed_dim: int = 256,
        use_attention: bool = True
    ):
        super().__init__(encoder_channels, num_classes)
        
        self.use_attention = use_attention
        self.embed_dim = embed_dim
        
        # 投影层
        self.linear_layers = nn.ModuleList([
            MLP(ch, embed_dim) for ch in encoder_channels
        ])
        
        if use_attention:
            # 特征注意力
            self.feature_attention = MultiScaleFeatureAttention(embed_dim, len(encoder_channels))
        
        # 融合层
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(encoder_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
    
    def forward(self, features: list) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        
        # 投影所有特征
        projected = []
        for i, (feat, linear) in enumerate(zip(features, self.linear_layers)):
            p = linear(feat)
            if i > 0:  # 上采样
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            projected.append(p)
        
        if self.use_attention:
            # 应用注意力
            projected = self.feature_attention(projected)
        
        # 融合
        concat = torch.cat(projected, dim=1)
        fused = self.linear_fuse(concat)
        fused = self.dropout(fused)
        
        output = self.linear_pred(fused)
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)
        
        return output


class MultiScaleFeatureAttention(nn.Module):
    """
    多尺度特征注意力 - 学习不同尺度特征的重要性
    """
    
    def __init__(self, embed_dim: int, num_scales: int):
        super().__init__()
        
        self.num_scales = num_scales
        
        # 全局池化 + MLP生成注意力权重
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.attention_mlp = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_scales),
            nn.Softmax(dim=1)
        )
    
    def forward(self, features: list) -> list:
        """
        Args:
            features: 多尺度特征列表
        Returns:
            weighted_features: 加权后的特征列表
        """
        B = features[0].shape[0]
        
        # 全局池化
        pooled = [self.global_pool(f).view(B, -1) for f in features]
        
        # 拼接并生成注意力权重
        concat = torch.cat(pooled, dim=1)
        weights = self.attention_mlp(concat)  # [B, num_scales]
        
        # 应用权重
        weighted = []
        for i, feat in enumerate(features):
            w = weights[:, i:i+1].view(B, 1, 1, 1)
            weighted.append(feat * w)
        
        return weighted
