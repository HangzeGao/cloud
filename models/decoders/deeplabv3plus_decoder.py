"""
DeepLabV3+ Decoder - 空洞空间金字塔 + 低层特征融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_decoder import BaseDecoder


class DeepLabV3PlusDecoder(BaseDecoder):
    """
    DeepLabV3+ Decoder
    
    特点：
    1. ASPP模块捕获多尺度上下文
    2. 编码器-解码器结构
    3. 可调的输出步长（Output Stride）
    
    Args:
        encoder_channels: 编码器通道数 [c2, c3, c4, c5]
        num_classes: 输出类别数
        output_stride: 输出步长（16或8）
        decoder_channels: 解码器通道数
    """
    
    def __init__(
        self,
        encoder_channels: list,
        num_classes: int = 2,
        output_stride: int = 16,
        decoder_channels: int = 256
    ):
        super().__init__(encoder_channels, num_classes)
        
        self.output_stride = output_stride
        
        # 使用编码器的低层特征（c2，1/4尺度）
        low_level_channels = encoder_channels[0]
        
        # 低层特征投影
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # 解码器（融合ASPP输出和低层特征）
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_channels + 48, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 分类头
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: list, aspp_features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: [c2, c3, c4, c5]
            aspp_features: ASPP模块的输出（如果使用ASPP作为neck）
        Returns:
            output: [B, num_classes, H, W]
        """
        c2, c3, c4, c5 = features
        
        # 如果没有传入ASPP特征，使用c5
        if aspp_features is None:
            aspp_features = c5
        
        # 低层特征
        low_level_feat = self.low_level_conv(c2)
        
        # 上采样ASPP特征到低层特征尺寸
        aspp_up = F.interpolate(
            aspp_features,
            size=low_level_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        # 拼接并解码
        concat = torch.cat([aspp_up, low_level_feat], dim=1)
        decoder_out = self.decoder(concat)
        
        # 分类
        output = self.classifier(decoder_out)
        
        # 上采样到原图尺寸
        output = F.interpolate(
            output,
            scale_factor=4,
            mode='bilinear',
            align_corners=False
        )
        
        return output


class DeepLabV3PlusDecoderV2(BaseDecoder):
    """
    DeepLabV3+ 改进版 - 使用可分离卷积减少参数量
    """
    
    def __init__(
        self,
        encoder_channels: list,
        num_classes: int = 2,
        output_stride: int = 16,
        decoder_channels: int = 256
    ):
        super().__init__(encoder_channels, num_classes)
        
        low_level_channels = encoder_channels[0]
        
        # 使用可分离卷积的低层特征投影
        self.low_level_conv = nn.Sequential(
            SeparableConv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # 使用可分离卷积的解码器
        self.decoder = nn.Sequential(
            SeparableConv2d(decoder_channels + 48, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            
            SeparableConv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)
    
    def forward(self, features: list, aspp_features: torch.Tensor = None) -> torch.Tensor:
        c2, c3, c4, c5 = features
        
        if aspp_features is None:
            aspp_features = c5
        
        low_level_feat = self.low_level_conv(c2)
        
        aspp_up = F.interpolate(
            aspp_features,
            size=low_level_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        concat = torch.cat([aspp_up, low_level_feat], dim=1)
        decoder_out = self.decoder(concat)
        output = self.classifier(decoder_out)
        
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)
        
        return output


class SeparableConv2d(nn.Module):
    """可分离卷积（Depthwise + Pointwise）"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False
    ):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            bias=bias
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
