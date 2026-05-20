"""
ASPP (Atrous Spatial Pyramid Pooling) 特征融合
DeepLab系列使用的空洞空间金字塔池化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPFusion(nn.Module):
    """
    ASPP模块 - 空洞空间金字塔池化
    
    功能：
    1. 多尺度特征提取（不同空洞率的卷积）
    2. 全局上下文编码（全局平均池化）
    3. 特征融合
    
    Args:
        in_channels: 输入特征通道数（通常使用encoder的最后一层）
        out_channels: 输出通道数
        rates: 空洞率列表 [6, 12, 18]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: list = [6, 12, 18],
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rates = rates
        
        # 1x1卷积分支（捕获局部信息）
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 空洞卷积分支（捕获多尺度上下文）
        self.branches = nn.ModuleList()
        for rate in rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 全局平均池化分支（捕获全局上下文）
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合后的投影层
        self.projection = nn.Sequential(
            nn.Conv2d(
                out_channels * (len(rates) + 2),  # 1x1 + rates + global
                out_channels,
                1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, features: list) -> list:
        """
        Args:
            features: 输入特征列表，通常只使用最后一层 [c5]
        Returns:
            fused_features: 融合后的特征列表（保持输入结构）
        """
        # ASPP通常作用于最后一层特征
        x = features[-1] if isinstance(features, list) else features
        
        # 获取输入尺寸
        size = x.shape[-2:]
        
        # 1x1卷积分支
        conv_1x1 = self.branch_1x1(x)
        
        # 空洞卷积分支
        atrous_outputs = []
        for branch in self.branches:
            atrous_outputs.append(branch(x))
        
        # 全局池化分支（需要上采样到原始尺寸）
        global_feat = self.global_branch(x)
        global_feat = F.interpolate(
            global_feat,
            size=size,
            mode='bilinear',
            align_corners=False
        )
        
        # 拼接所有分支
        concat = torch.cat([conv_1x1] + atrous_outputs + [global_feat], dim=1)
        
        # 投影融合
        output = self.projection(concat)
        
        # 保持列表输出格式（兼容其他neck接口）
        if isinstance(features, list):
            # 将融合特征放置到最深层位置
            return features[:-1] + [output]
        return [output]
    
    def get_out_channels(self) -> int:
        return self.out_channels


class ASPPFusionV2(nn.Module):
    """
    改进版ASPP - 增加可变形卷积和注意力机制
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        rates: list = [6, 12, 18],
        use_se: bool = True
    ):
        super().__init__()
        
        self.aspp = ASPPFusion(in_channels, out_channels, rates)
        self.use_se = use_se
        
        if use_se:
            # Squeeze-and-Excitation注意力
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 16, out_channels, 1),
                nn.Sigmoid()
            )
    
    def forward(self, features: list) -> list:
        output = self.aspp(features)
        
        if self.use_se:
            # 应用SE注意力
            se_weight = self.se(output[-1])
            output[-1] = output[-1] * se_weight
        
        return output


class LightASPP(nn.Module):
    """
    轻量级ASPP - 减少分支数量，适合边缘设备
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        rates: list = [6, 12]
    ):
        super().__init__()
        
        # 简化的ASPP：减少分支
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.atrous = nn.ModuleList()
        for rate in rates:
            self.atrous.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels // 4,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels // 4),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: list) -> list:
        x = features[-1] if isinstance(features, list) else features
        size = x.shape[-2:]
        
        # 各分支
        conv1 = self.conv_1x1(x)
        atrous_outs = [branch(x) for branch in self.atrous]
        
        global_out = self.global_pool(x)
        global_out = F.interpolate(global_out, size=size, mode='bilinear', align_corners=False)
        
        # 拼接
        concat = torch.cat([conv1] + atrous_outs + [global_out], dim=1)
        output = self.project(concat)
        
        if isinstance(features, list):
            return features[:-1] + [output]
        return [output]
