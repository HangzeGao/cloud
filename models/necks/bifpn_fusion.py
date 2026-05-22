"""
BiFPN (Bidirectional Feature Pyramid Network) 特征融合
EfficientDet提出的高效特征融合方案
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPNFusion(nn.Module):
    """
    BiFPN双向特征金字塔融合
    
    特点：
    1. 自顶向下 + 自底向上的双向融合
    2. 可学习的权重（加权融合）
    3. 跨尺度连接（跳跃连接）
    
    Args:
        in_channels: 输入特征通道列表
        out_channels: 输出特征通道数
        num_iterations: 双向融合迭代次数
    """
    
    def __init__(
        self,
        in_channels: list,
        out_channels: int = 256,
        num_iterations: int = 2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = len(in_channels)
        self.num_iterations = num_iterations
        
        # 首先将所有输入特征投影到统一通道数
        self.input_projections = nn.ModuleList()
        for in_ch in self.in_channels:
            self.input_projections.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 双向融合模块（可重复迭代）
        self.bifpn_blocks = nn.ModuleList()
        for _ in range(num_iterations):
            self.bifpn_blocks.append(
                BiFPNBlock(out_channels, self.num_levels)
            )
    
    def forward(self, features: list) -> list:
        """
        Args:
            features: [c2, c3, c4, c5]
        Returns:
            fused_features: 双向融合后的特征
        """
        # 投影到统一通道
        projected = []
        for i, feat in enumerate(features):
            projected.append(self.input_projections[i](feat))
        
        # 迭代双向融合
        for bifpn_block in self.bifpn_blocks:
            projected = bifpn_block(projected)
        
        return projected
    
    def get_out_channels(self) -> int:
        return self.out_channels


class BiFPNBlock(nn.Module):
    """
    单个BiFPN融合块
    包含：自顶向下路径 + 自底向上路径
    """
    
    def __init__(self, channels: int, num_levels: int):
        super().__init__()
        
        self.channels = channels
        self.num_levels = num_levels
        
        # 自顶向下路径的权重和卷积
        self.top_down_weights = nn.ParameterList()
        self.top_down_convs = nn.ModuleList()
        for i in range(num_levels - 1):
            # 每个节点有两个输入（来自同层和上层）
            self.top_down_weights.append(nn.Parameter(torch.ones(2, dtype=torch.float32)))
            self.top_down_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 自底向上路径的权重和卷积
        self.bottom_up_weights = nn.ParameterList()
        self.bottom_up_convs = nn.ModuleList()
        for i in range(num_levels - 1):
            self.bottom_up_weights.append(nn.Parameter(torch.ones(2, dtype=torch.float32)))
            self.bottom_up_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 使用ReLU确保权重为正
        self.eps = 0.0001
        self.relu = nn.ReLU()
    
    def forward(self, features: list) -> list:
        """
        Args:
            features: [P2, P3, P4, P5] 已经投影到统一通道
        """
        # 自顶向下路径
        top_down_features = [features[-1]]  # 从最高层开始
        
        for i in range(self.num_levels - 2, -1, -1):
            # 上采样上一层特征
            upsampled = F.interpolate(
                top_down_features[-1],
                size=features[i].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            # 加权融合
            weights = self.relu(self.top_down_weights[self.num_levels - 2 - i])
            weights = weights / (weights.sum() + self.eps)
            
            fused = weights[0] * features[i] + weights[1] * upsampled
            fused = self.top_down_convs[self.num_levels - 2 - i](fused)
            top_down_features.append(fused)
        
        top_down_features = top_down_features[::-1]  # 反转回[P2, P3, P4, P5]
        
        # 自底向上路径
        bottom_up_features = [top_down_features[0]]  # 从最底层开始
        
        for i in range(1, self.num_levels):
            # 下采样下一层特征
            downsampled = F.max_pool2d(
                bottom_up_features[-1],
                kernel_size=2,
                stride=2
            )
            
            # 加权融合
            weights = self.relu(self.bottom_up_weights[i - 1])
            weights = weights / (weights.sum() + self.eps)
            
            fused = weights[0] * top_down_features[i] + weights[1] * downsampled
            fused = self.bottom_up_convs[i - 1](fused)
            bottom_up_features.append(fused)
        
        return bottom_up_features


class FastBiFPN(nn.Module):
    """
    快速版BiFPN - 减少计算量
    适用于实时推理场景
    """
    
    def __init__(
        self,
        in_channels: list,
        out_channels: int = 256,
        num_iterations: int = 1
    ):
        super().__init__()
        
        self.out_channels = out_channels
        
        # 输入特征投影到统一通道数
        self.input_projections = nn.ModuleList()
        for in_ch in in_channels:
            self.input_projections.append(
                nn.Conv2d(in_ch, out_channels, 1)
            )
        
        # 加权特征融合
        self.fusion_weights = nn.Parameter(torch.ones(len(in_channels)))
        
        # 输出卷积
        self.output_convs = nn.ModuleList()
        for _ in range(len(in_channels)):
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )
    
    def forward(self, features: list) -> list:
        # 投影
        projected = [self.input_projections[i](f) for i, f in enumerate(features)]
        
        # 归一化权重
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # 多尺度特征融合
        outputs = []
        for i, feat in enumerate(projected):
            # 收集所有尺度的特征（上采样或下采样到当前尺度）
            resized_features = []
            for j, pf in enumerate(projected):
                if i == j:
                    resized_features.append(pf)
                elif i < j:
                    # 需要上采样
                    resized = F.interpolate(
                        pf,
                        size=feat.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    resized_features.append(resized)
                else:
                    # 需要下采样
                    resized = F.max_pool2d(pf, kernel_size=2**(i-j), stride=2**(i-j))
                    resized_features.append(resized)
            
            # 加权融合
            weighted_sum = sum(w * rf for w, rf in zip(weights, resized_features))
            outputs.append(self.output_convs[i](weighted_sum))
        
        return outputs
