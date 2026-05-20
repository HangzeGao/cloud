"""
UNet++ Decoder - 嵌套跳跃连接，高精度分割
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_decoder import BaseDecoder


class UNetPPDecoder(BaseDecoder):
    """
    UNet++ Decoder
    
    特点：
    1. 嵌套跳跃连接（Nested Skip Connections）
    2. 密集连接
    3. 可选深度监督（Deep Supervision）
    4. 可选CBAM注意力
    
    Args:
        encoder_channels: 编码器各层通道数 [c2, c3, c4, c5]
        num_classes: 输出类别数
        deep_supervision: 是否使用深度监督
        use_cbam: 是否使用CBAM注意力
    """
    
    def __init__(
        self,
        encoder_channels: list,
        num_classes: int = 2,
        deep_supervision: bool = True,
        use_cbam: bool = True,
        decoder_channels: int = 256
    ):
        super().__init__(encoder_channels, num_classes)
        
        self.deep_supervision = deep_supervision
        self.use_cbam = use_cbam
        
        # 特征通道数（从深到浅）
        # encoder_channels: [c2, c3, c4, c5]
        c2, c3, c4, c5 = encoder_channels
        
        # 解码器通道配置
        # UNet++采用嵌套结构，X^(i,j) 表示第i层第j个节点
        
        # 节点 X^(0,0) -> X^(0,3) (最上层)
        # 节点 X^(1,0) -> X^(1,2)
        # 节点 X^(2,0) -> X^(2,1)
        # 节点 X^(3,0)
        
        # 上采样和卷积模块
        # 第0列（直接来自编码器）
        self.conv_00 = self._make_conv_block(c2, decoder_channels)
        self.conv_10 = self._make_conv_block(c3, decoder_channels)
        self.conv_20 = self._make_conv_block(c4, decoder_channels)
        self.conv_30 = self._make_conv_block(c5, decoder_channels)
        
        # 第1列
        self.up_01 = nn.ConvTranspose2d(decoder_channels, decoder_channels, 2, stride=2)
        self.conv_01 = self._make_conv_block(decoder_channels * 2, decoder_channels)
        
        self.up_11 = nn.ConvTranspose2d(decoder_channels, decoder_channels, 2, stride=2)
        self.conv_11 = self._make_conv_block(decoder_channels * 2, decoder_channels)
        
        self.up_21 = nn.ConvTranspose2d(decoder_channels, decoder_channels, 2, stride=2)
        self.conv_21 = self._make_conv_block(decoder_channels * 2, decoder_channels)
        
        # 第2列
        self.up_02 = nn.ConvTranspose2d(decoder_channels, decoder_channels, 2, stride=2)
        self.conv_02 = self._make_conv_block(decoder_channels * 3, decoder_channels)
        
        self.up_12 = nn.ConvTranspose2d(decoder_channels, decoder_channels, 2, stride=2)
        self.conv_12 = self._make_conv_block(decoder_channels * 3, decoder_channels)
        
        # 第3列
        self.up_03 = nn.ConvTranspose2d(decoder_channels, decoder_channels, 2, stride=2)
        self.conv_03 = self._make_conv_block(decoder_channels * 4, decoder_channels)
        
        # CBAM注意力（可选）
        if use_cbam:
            self.cbam_01 = CBAM(decoder_channels)
            self.cbam_02 = CBAM(decoder_channels)
            self.cbam_03 = CBAM(decoder_channels)
        
        # 输出层
        self.final_conv = nn.Conv2d(decoder_channels, num_classes, 1)
        
        # 深度监督输出（可选）
        if deep_supervision:
            self.aux_01 = nn.Conv2d(decoder_channels, num_classes, 1)
            self.aux_02 = nn.Conv2d(decoder_channels, num_classes, 1)
            self.aux_03 = nn.Conv2d(decoder_channels, num_classes, 1)
    
    def _make_conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """创建卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: [c2, c3, c4, c5]
        Returns:
            output: [B, num_classes, H, W]
        """
        c2, c3, c4, c5 = features
        
        # 第0列
        x_00 = self.conv_00(c2)
        x_10 = self.conv_10(c3)
        x_20 = self.conv_20(c4)
        x_30 = self.conv_30(c5)
        
        # 第1列
        x_01 = self.conv_01(torch.cat([x_00, self.up_01(x_10)], dim=1))
        x_11 = self.conv_11(torch.cat([x_10, self.up_11(x_20)], dim=1))
        x_21 = self.conv_21(torch.cat([x_20, self.up_21(x_30)], dim=1))
        
        if self.use_cbam:
            x_01 = self.cbam_01(x_01)
        
        # 第2列
        x_02 = self.conv_02(torch.cat([x_00, x_01, self.up_02(x_11)], dim=1))
        x_12 = self.conv_12(torch.cat([x_10, x_11, self.up_12(x_21)], dim=1))
        
        if self.use_cbam:
            x_02 = self.cbam_02(x_02)
        
        # 第3列
        x_03 = self.conv_03(torch.cat([x_00, x_01, x_02, self.up_03(x_12)], dim=1))
        
        if self.use_cbam:
            x_03 = self.cbam_03(x_03)
        
        # 最终输出（上采样到原图尺寸）
        output = self.final_conv(x_03)
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)
        
        if self.deep_supervision and self.training:
            # 深度监督分支
            aux1 = self.aux_01(x_01)
            aux1 = F.interpolate(aux1, scale_factor=4, mode='bilinear', align_corners=False)
            
            aux2 = self.aux_02(x_02)
            aux2 = F.interpolate(aux2, scale_factor=4, mode='bilinear', align_corners=False)
            
            aux3 = self.aux_03(x_03)
            aux3 = F.interpolate(aux3, scale_factor=4, mode='bilinear', align_corners=False)
            
            return output, [aux1, aux2, aux3]
        
        return output


class CBAM(nn.Module):
    """
    CBAM注意力模块 (Channel + Spatial)
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x
