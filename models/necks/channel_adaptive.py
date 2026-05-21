"""
通道自适应输入模块 - 多种实现方式

支持不同策略处理3/4通道输入：
- conv: 卷积投影（轻量高效）
- attention: 通道注意力机制
- transformer: 轻量Transformer编码
- physics: 物理先验+可学习调整
- hybrid: 混合策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BaseChannelAdaptive(nn.Module):
    """通道自适应基类"""
    
    def __init__(self, out_channels: int = 4):
        super().__init__()
        self.out_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W], C=3 or 4
        Returns:
            [B, out_channels, H, W]
        """
        raise NotImplementedError


class ConvChannelAdaptive(BaseChannelAdaptive):
    """
    卷积投影方式 - 轻量高效
    使用可学习卷积网络进行通道转换
    """
    
    def __init__(self, out_channels: int = 4, use_attention: bool = True):
        super().__init__(out_channels)
        self.use_attention = use_attention
        
        # 3通道→4通道投影
        self.rgb_to_4ch = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # 4通道精化
        self.refine_4ch = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # 通道注意力
        if use_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels // 2, out_channels),
                nn.Sigmoid()
            )
        else:
            self.channel_weights = nn.Parameter(torch.ones(out_channels))
        
        # 可学习残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_channels = x.shape[1]
        
        if in_channels == 3:
            transformed = self.rgb_to_4ch(x)
            # 将RGB填充为4通道（第4通道为0）
            residual = F.pad(x, (0, 0, 0, 0, 0, 1), value=0)
        elif in_channels == 4:
            transformed = self.refine_4ch(x)
            residual = x
        else:
            raise ValueError(f"Expected 3 or 4 channels, got {in_channels}")
        
        # 残差融合
        alpha = torch.sigmoid(self.residual_weight)
        out = alpha * transformed + (1 - alpha) * residual
        
        # 通道权重调整
        if self.use_attention:
            weights = self.channel_attention(out).view(out.size(0), out.size(1), 1, 1)
            out = out * weights
        else:
            out = out * self.channel_weights.view(1, -1, 1, 1)
        
        return out


class AttentionChannelAdaptive(BaseChannelAdaptive):
    """
    注意力机制方式 - 更强的特征提取
    使用空间注意力和通道注意力结合
    """
    
    def __init__(self, out_channels: int = 4):
        super().__init__(out_channels)
        
        # 3通道编码器
        self.encoder_3ch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 4通道编码器
        self.encoder_4ch = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(64, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 64),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_channels = x.shape[1]
        
        # 根据输入通道选择编码器
        if in_channels == 3:
            feat = self.encoder_3ch(x)
        elif in_channels == 4:
            feat = self.encoder_4ch(x)
        else:
            raise ValueError(f"Expected 3 or 4 channels, got {in_channels}")
        
        # 应用注意力
        spatial_w = self.spatial_attn(feat)
        channel_w = self.channel_attn(feat).view(feat.size(0), feat.size(1), 1, 1)
        
        # 注意力加权
        feat = feat * spatial_w * channel_w
        
        # 投影输出
        out = self.output_proj(feat)
        
        return out


class PhysicsChannelAdaptive(BaseChannelAdaptive):
    """
    物理先验方式 - 可解释性强
    基于物理模型生成伪NIR，加可学习调整
    """
    
    def __init__(self, out_channels: int = 4):
        super().__init__(out_channels)
        
        # 物理先验权重（初始化遵循物理模型）
        self.register_buffer('physics_weights', torch.tensor([0.7, 0.25, 0.05]))
        
        # 可学习调整参数
        self.adjustment = nn.Parameter(torch.zeros(4))  # RGB + NIR的微调
        
        # 精化网络（微调物理生成的结果）
        self.refine = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, 3, padding=1),
            nn.BatchNorm2d(4),
        )
        
        # 混合权重（物理 vs 学习）
        self.mix_weight = nn.Parameter(torch.tensor(0.3))  # 偏向物理
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_channels = x.shape[1]
        
        if in_channels == 3:
            # 物理模型生成NIR
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            nir_phy = (self.physics_weights[0] * r + 
                      self.physics_weights[1] * g + 
                      self.physics_weights[2] * b)
            x_4ch = torch.cat([x, nir_phy], dim=1)
        else:
            x_4ch = x
        
        # 应用可学习调整
        adjustment = torch.tanh(self.adjustment).view(1, 4, 1, 1)
        x_adjusted = x_4ch + adjustment
        
        # 精化
        refined = self.refine(x_4ch)
        
        # 混合物理和调整后结果
        mix = torch.sigmoid(self.mix_weight)
        out = mix * refined + (1 - mix) * x_adjusted
        
        return out


class HybridChannelAdaptive(BaseChannelAdaptive):
    """
    混合策略 - 综合多种方法优势
    结合卷积、注意力、物理先验
    """
    
    def __init__(self, out_channels: int = 4):
        super().__init__(out_channels)
        
        # 子模块
        self.conv_adaptive = ConvChannelAdaptive(out_channels, use_attention=True)
        self.physics_adaptive = PhysicsChannelAdaptive(out_channels)
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels * 2, 2),
            nn.Softmax(dim=1)
        )
        
        # 最终精化
        self.final_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 并行处理
        out_conv = self.conv_adaptive(x)
        out_physics = self.physics_adaptive(x)
        
        # 门控融合
        gate_input = torch.cat([out_conv, out_physics], dim=1)
        weights = self.gate(gate_input).view(x.size(0), 2, 1, 1)
        
        w_conv = weights[:, 0:1, :, :]
        w_physics = weights[:, 1:2, :, :]
        
        out = w_conv * out_conv + w_physics * out_physics
        
        # 最终精化
        out = self.final_refine(out)
        
        return out


class TransformerChannelAdaptive(BaseChannelAdaptive):
    """
    Transformer方式 - 全局建模能力强
    轻量级Vision Transformer处理通道关系
    """
    
    def __init__(self, out_channels: int = 4, embed_dim: int = 32, num_heads: int = 2):
        super().__init__(out_channels)
        
        self.embed_dim = embed_dim
        
        # Patch embedding (将图像分块)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 256, embed_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1),
        )
        
        # 上采样
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 如果是4通道输入，先投影到3通道
        if C == 4:
            x = x[:, :3, :, :]  # 简化：取RGB
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H//4, W//4]
        
        # 展平为序列
        B, C_emb, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H_p*W_p, C_emb]
        
        # 添加位置编码
        if x.size(1) <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :x.size(1), :]
        
        # Transformer编码
        x = self.transformer(x)
        
        # 恢复空间维度
        x = x.transpose(1, 2).view(B, C_emb, H_p, W_p)
        
        # 输出投影
        x = self.output_proj(x)
        
        # 上采样回原尺寸
        x = self.upsample(x)
        
        return x
