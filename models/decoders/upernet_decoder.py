"""
UperNet Decoder - 金字塔池化 + FPN融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_decoder import BaseDecoder


class UperNetDecoder(BaseDecoder):
    """
    UperNet Decoder (Unified Perceptual Parsing Network)
    
    特点：
    1. PPM (Pyramid Pooling Module) 捕获全局上下文
    2. FPN融合多尺度特征
    
    Args:
        encoder_channels: 编码器通道数 [c2, c3, c4, c5]
        num_classes: 输出类别数
        pool_scales: PPM池化尺度
        channels: 中间通道数
    """
    
    def __init__(
        self,
        encoder_channels: list,
        num_classes: int = 2,
        pool_scales: list = [1, 2, 3, 6],
        channels: int = 512
    ):
        super().__init__(encoder_channels, num_classes)
        
        self.pool_scales = pool_scales
        self.channels = channels
        
        # PPM模块（作用于最后一层特征）
        self.ppm = PPM(
            in_channels=encoder_channels[-1],
            out_channels=channels,
            pool_scales=pool_scales
        )
        
        # FPN模块 - 将高层特征融合到低层
        self.fpn_in = nn.ModuleList()
        for in_ch in encoder_channels[:-1]:  # 不包括最后一层（已被PPM处理）
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, channels, 1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # FPN输出卷积
        self.fpn_out = nn.ModuleList()
        for _ in range(len(encoder_channels)):
            self.fpn_out.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 分类头
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)
    
    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: [c2, c3, c4, c5]
        Returns:
            output: [B, num_classes, H, W]
        """
        # PPM处理最深层特征
        ppm_out = self.ppm(features[-1])
        
        # FPN自顶向下路径
        fpn_features = [ppm_out]
        
        for i in range(len(features) - 2, -1, -1):
            # 投影当前层特征
            lateral = self.fpn_in[i](features[i])
            
            # 上采样高层特征
            upsampled = F.interpolate(
                fpn_features[-1],
                size=lateral.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            # 相加
            fused = lateral + upsampled
            fpn_features.append(fused)
        
        # 应用输出卷积（从浅层到深层）
        outs = []
        for i, feat in enumerate(reversed(fpn_features)):
            outs.append(self.fpn_out[i](feat))
        
        # 使用最浅层特征进行预测（包含最多空间信息）
        output = self.conv_seg(outs[0])
        
        # 上采样到原图尺寸
        output = F.interpolate(
            output,
            scale_factor=4,
            mode='bilinear',
            align_corners=False
        )
        
        return output


class PPM(nn.Module):
    """
    Pyramid Pooling Module
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_scales: list
    ):
        super().__init__()
        
        self.pool_scales = pool_scales
        
        # 各尺度的池化分支
        self.pools = nn.ModuleList()
        for scale in pool_scales:
            self.pools.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels // len(pool_scales), 1, bias=False),
                    nn.BatchNorm2d(out_channels // len(pool_scales)),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 原始特征分支
        self.conv_orig = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // len(pool_scales), 1, bias=False),
            nn.BatchNorm2d(out_channels // len(pool_scales)),
            nn.ReLU(inplace=True)
        )
        
        # 融合投影
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            fused: [B, C_out, H, W]
        """
        size = x.shape[-2:]
        
        # 原始特征
        orig = self.conv_orig(x)
        
        # 各池化分支
        pool_outs = [orig]
        for pool in self.pools:
            pooled = pool(x)
            # 上采样回原尺寸
            upsampled = F.interpolate(pooled, size=size, mode='bilinear', align_corners=False)
            pool_outs.append(upsampled)
        
        # 拼接
        concat = torch.cat(pool_outs, dim=1)
        
        # 融合
        output = self.conv_fuse(concat)
        
        return output


class SimpleDecoder(BaseDecoder):
    """
    简单解码器 - 用于快速实验和baseline
    仅使用最深层特征，上采样后分类
    """
    
    def __init__(
        self,
        encoder_channels: list,
        num_classes: int = 2,
        decoder_channels: int = 256
    ):
        super().__init__(encoder_channels, num_classes)
        
        # 简单的多层卷积上采样
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(decoder_channels, num_classes, 1)
        )
    
    def forward(self, features: list) -> torch.Tensor:
        # 使用最深层特征
        x = features[-1]
        
        # 解码
        output = self.decoder(x)
        
        # 上采样到原图尺寸（假设编码器下采样32倍）
        output = F.interpolate(
            output,
            scale_factor=32,
            mode='bilinear',
            align_corners=False
        )
        
        return output
