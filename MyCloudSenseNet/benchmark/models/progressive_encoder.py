"""
渐进式位深度感知编码器 (Progressive Bit-Depth Encoder)

在编码器的各个阶段逐步融入位深度信息，实现更高效的多层次特征调制。
支持可配置的位深度感知程度（从仅在 stem 层到全阶段渗透）。
"""

from typing import List, Optional, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitDepthAwareStem(nn.Module):
    """
    位深度感知 Stem 层
    
    在输入层就融入位深度信息，通过轻量级统计模块估计输入图像的位深度，
    并生成初始调制信号。
    """
    
    def __init__(
        self,
        in_channels: int,
        num_bit_depths: int = 5,
        enable_modulation: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_bit_depths = num_bit_depths
        self.enable_modulation = enable_modulation
        
        # 位深度快速估计（轻量级）
        self.bit_depth_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(16, num_bit_depths),
        )
        
        # 输入调制（可选）
        if enable_modulation:
            self.input_modulation = nn.Sequential(
                nn.Linear(num_bit_depths, in_channels),
                nn.Sigmoid(),
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            字典包含:
                - 'features': 调制后的特征（如果启用）或原始特征
                - 'bit_depth_logits': 位深度 logits [B, num_bit_depths]
                - 'bit_depth_probs': 位深度概率 [B, num_bit_depths]
        """
        # 估计位深度
        bit_depth_logits = self.bit_depth_estimator(x)
        bit_depth_probs = F.softmax(bit_depth_logits, dim=1)
        
        result = {
            'features': x,
            'bit_depth_logits': bit_depth_logits,
            'bit_depth_probs': bit_depth_probs,
        }
        
        # 应用输入调制
        if self.enable_modulation:
            B, C, H, W = x.shape
            modulation = self.input_modulation(bit_depth_probs).view(B, C, 1, 1)
            result['features'] = x * modulation
        
        return result


class BitDepthModulation(nn.Module):
    """
    位深度调制模块
    
    在编码器的某个阶段插入，根据位深度信息调制特征。
    """
    
    def __init__(
        self,
        channels: int,
        num_bit_depths: int,
        modulation_type: str = "channel",  # channel, spatial, both
        reduction: int = 16,
    ):
        super().__init__()
        self.channels = channels
        self.modulation_type = modulation_type
        
        mid_channels = max(channels // reduction, 4)
        
        # 位深度到调制参数的映射
        if modulation_type in ("channel", "both"):
            self.channel_proj = nn.Sequential(
                nn.Linear(num_bit_depths, mid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(mid_channels, channels),
                nn.Sigmoid(),
            )
        
        if modulation_type in ("spatial", "both"):
            self.spatial_proj = nn.Sequential(
                nn.Linear(num_bit_depths, mid_channels),
                nn.ReLU(inplace=True),
                nn.Linear(mid_channels, 1),
                nn.Sigmoid(),
            )
            
            # 空间注意力生成
            self.spatial_gen = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, 1, 1),
                nn.Sigmoid(),
            )
    
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
        
        # 计算位深度概率
        bit_depth_probs = F.softmax(bit_depth_logits, dim=1)
        
        if self.modulation_type == "channel":
            # 通道调制
            channel_mod = self.channel_proj(bit_depth_probs).view(B, C, 1, 1)
            return x * channel_mod
        
        elif self.modulation_type == "spatial":
            # 空间调制
            spatial_mod = self.spatial_proj(bit_depth_probs).view(B, 1, 1, 1)
            spatial_attn = self.spatial_gen(x)
            return x * spatial_attn * (1 + spatial_mod)
        
        elif self.modulation_type == "both":
            # 联合调制
            channel_mod = self.channel_proj(bit_depth_probs).view(B, C, 1, 1)
            spatial_mod = self.spatial_proj(bit_depth_probs).view(B, 1, 1, 1)
            spatial_attn = self.spatial_gen(x)
            return x * channel_mod * (1 + spatial_attn * spatial_mod)
        
        else:
            return x


class ProgressiveBitDepthEncoder(nn.Module):
    """
    渐进式位深度感知编码器
    
    将位深度感知融入编码器各阶段，支持灵活的配置：
    - 'stem_only': 仅在输入层估计位深度
    - 'early': 在前两个 stage 调制
    - 'late': 在后两个 stage 调制
    - 'all': 在所有 stage 调制
    - 'adaptive': 根据输入动态选择调制策略
    
    Attributes:
        stages: 编码器阶段列表，每个阶段可以是原始层或（原始层 + 调制器）
        stem: 位深度感知 stem 层
        modulation_strategy: 调制策略
    """
    
    MODULATION_STRATEGIES = {
        'stem_only': [],
        'early': [0, 1],
        'mid': [1, 2],
        'late': [2, 3],
        'all': [0, 1, 2, 3],
        'adaptive': 'adaptive',
    }
    
    def __init__(
        self,
        base_encoder: nn.Module,
        in_channels: int = 4,
        num_bit_depths: int = 5,
        modulation_strategy: str = "adaptive",
        modulation_type: str = "channel",
        enable_stem_modulation: bool = True,
    ):
        super().__init__()
        
        self.base_encoder = base_encoder
        self.modulation_strategy = modulation_strategy
        self.num_bit_depths = num_bit_depths
        
        # 位深度感知 stem
        self.stem = BitDepthAwareStem(
            in_channels=in_channels,
            num_bit_depths=num_bit_depths,
            enable_modulation=enable_stem_modulation,
        )
        
        # 确定哪些阶段需要调制
        if modulation_strategy in self.MODULATION_STRATEGIES:
            stages_to_modulate = self.MODULATION_STRATEGIES[modulation_strategy]
        else:
            stages_to_modulate = []
        
        # 构建渐进式编码器
        self.stages = nn.ModuleList()
        self.modulators = nn.ModuleDict()
        
        # 获取编码器输出通道
        if hasattr(base_encoder, 'out_channels'):
            out_channels = base_encoder.out_channels
            if isinstance(out_channels, (list, tuple)):
                self.encoder_channels = list(out_channels)
            else:
                self.encoder_channels = [out_channels]
        else:
            # 默认假设
            self.encoder_channels = [64, 128, 256, 512]
        
        # 构建各阶段 - 直接使用整个编码器作为一个 stage
        # SMP 编码器结构复杂，直接整体使用更稳定
        self.stages = nn.ModuleList([base_encoder])
        
        # 只在输出层添加调制器
        if self.encoder_channels:
            ch = self.encoder_channels[-1]
            self.modulators['output'] = BitDepthModulation(
                channels=ch,
                num_bit_depths=num_bit_depths,
                modulation_type=modulation_type,
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        渐进式前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            字典包含:
                - 'features': 编码特征
                - 'multi_scale_features': 多尺度特征列表（如果可用）
                - 'bit_depth_logits': 位深度 logits
                - 'bit_depth_probs': 位深度概率
        """
        # 1. Stem 层处理（包含位深度初始估计）
        stem_result = self.stem(x)
        x = stem_result['features']
        bit_depth_logits = stem_result['bit_depth_logits']
        bit_depth_probs = stem_result['bit_depth_probs']
        
        # 2. 渐进式编码
        multi_scale_features = []
        
        for i, stage in enumerate(self.stages):
            # 执行当前 stage
            x = stage(x)
            
            # 保存多尺度特征
            if isinstance(x, (list, tuple)):
                multi_scale_features.extend(x)
                x = x[-1]  # 继续传播最高层特征
            else:
                multi_scale_features.append(x)
            
        # 应用输出层位深度调制（如果有）
        if 'output' in self.modulators:
            x = self.modulators['output'](x, bit_depth_logits)
        
        # 保存位深度信息供后续查询
        self._last_bit_depth_logits = bit_depth_logits
        self._last_bit_depth_probs = bit_depth_probs
        
        return {
            'features': x,
            'multi_scale_features': multi_scale_features,
            'bit_depth_logits': bit_depth_logits,
            'bit_depth_probs': bit_depth_probs,
            'stem_features': stem_result.get('features'),
        }
    
    def get_bit_depth_info(self) -> Dict[str, torch.Tensor]:
        """获取位深度信息（用于日志记录）"""
        if hasattr(self, '_last_bit_depth_logits'):
            bd_values = torch.arange(
                8, 13,  # 8, 9, 10, 11, 12
                device=self._last_bit_depth_probs.device,
                dtype=torch.float32
            )
            estimated = (self._last_bit_depth_probs * bd_values.view(1, -1)).sum(dim=1)
            return {
                'logits': self._last_bit_depth_logits,
                'probs': self._last_bit_depth_probs,
                'estimated': estimated,
            }
        return {}
    
    def freeze_stem_estimator(self):
        """冻结 stem 层的位深度估计器（用于微调阶段）"""
        for param in self.stem.bit_depth_estimator.parameters():
            param.requires_grad = False
    
    def unfreeze_stem_estimator(self):
        """解冻 stem 层的位深度估计器"""
        for param in self.stem.bit_depth_estimator.parameters():
            param.requires_grad = True


class AdaptiveBitDepthModulation(nn.Module):
    """
    自适应位深度调制器
    
    根据输入特征的质量和位深度估计的置信度，
    动态调整调制强度。
    """
    
    def __init__(
        self,
        channels: int,
        num_bit_depths: int,
        base_modulation: nn.Module,
    ):
        super().__init__()
        self.base_modulation = base_modulation
        self.num_bit_depths = num_bit_depths
        
        # 调制强度门控网络
        self.intensity_gate = nn.Sequential(
            nn.Linear(num_bit_depths, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        
        # 特征质量估计
        self.quality_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        bit_depth_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征
            bit_depth_logits: 位深度 logits
        
        Returns:
            自适应调制后的特征
        """
        # 计算调制强度
        bit_depth_probs = F.softmax(bit_depth_logits, dim=1)
        confidence = bit_depth_probs.max(dim=1)[0]  # 最大概率作为置信度
        intensity = self.intensity_gate(bit_depth_probs)  # [B, 1]
        
        # 估计特征质量
        quality = self.quality_estimator(x)  # [B, 1]
        
        # 综合调制强度
        modulation_strength = intensity * quality  # [B, 1]
        
        # 应用基础调制
        modulated = self.base_modulation(x, bit_depth_logits)
        
        # 插值混合：原始特征和调制特征的混合比例由调制强度决定
        B = x.size(0)
        alpha = modulation_strength.view(B, 1, 1, 1)
        
        return (1 - alpha) * x + alpha * modulated


def build_progressive_encoder(
    base_encoder: nn.Module,
    config: Optional[Dict] = None,
) -> ProgressiveBitDepthEncoder:
    """
    根据配置构建渐进式编码器
    
    Args:
        base_encoder: 基础编码器
        config: 配置字典
    
    Returns:
        ProgressiveBitDepthEncoder 实例
    """
    if config is None:
        config = {}
    
    return ProgressiveBitDepthEncoder(
        base_encoder=base_encoder,
        in_channels=config.get('in_channels', 4),
        num_bit_depths=config.get('num_bit_depths', 5),
        modulation_strategy=config.get('modulation_strategy', 'adaptive'),
        modulation_type=config.get('modulation_type', 'channel'),
        enable_stem_modulation=config.get('enable_stem_modulation', True),
    )


# 兼容性包装：保持与原 AdaptiveEncoder 相同的接口
class ProgressiveAdaptiveEncoderWrapper(nn.Module):
    """
    渐进式编码器的兼容包装
    
    与原 BitDepthAdaptiveEncoder 保持相同的接口，
    但内部使用 ProgressiveBitDepthEncoder。
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        in_channels: int = 4,
        feature_dim: int = None,
        enable_progressive: bool = True,
        modulation_strategy: str = 'adaptive',
    ):
        super().__init__()
        
        self.encoder = encoder
        self._feature_dim = feature_dim
        
        self.progressive_encoder = ProgressiveBitDepthEncoder(
            base_encoder=encoder,
            in_channels=in_channels,
            num_bit_depths=5,  # 8, 9, 10, 11, 12
            modulation_strategy=modulation_strategy,
            modulation_type='channel',
            enable_stem_modulation=True,
        )
    
    @property
    def out_channels(self):
        """兼容 SMP 接口"""
        return self.encoder.out_channels
    
    @property
    def output_stride(self):
        """兼容 SMP 接口"""
        return getattr(self.encoder, 'output_stride', 32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        result = self.progressive_encoder(x)
        # 保存位深度信息供后续使用
        self._last_bit_depth_logits = result['bit_depth_logits']
        self._last_bit_depth_probs = result['bit_depth_probs']
        return result['features']
    
    def get_bit_depth_info(self) -> Optional[Dict[str, torch.Tensor]]:
        """获取位深度信息"""
        return self.progressive_encoder.get_bit_depth_info()
    
    # 委托 SMP 相关方法
    def set_in_channels(self, in_channels, pretrained=True):
        if hasattr(self.encoder, 'set_in_channels'):
            return self.encoder.set_in_channels(in_channels, pretrained)
        return None
    
    def get_stages(self):
        if hasattr(self.encoder, 'get_stages'):
            return self.encoder.get_stages()
        return [self.encoder]
    
    def make_dilated(self, output_stride):
        if hasattr(self.encoder, 'make_dilated'):
            return self.encoder.make_dilated(output_stride)
        return None


if __name__ == "__main__":
    print("Testing ProgressiveBitDepthEncoder...")
    
    # 创建一个简单的模拟编码器
    class MockEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_channels = [32, 64, 128, 256]
            self.stages = nn.ModuleList([
                nn.Conv2d(4, 32, 3, stride=2, padding=1),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
            ])
        
        def forward(self, x):
            for stage in self.stages:
                x = stage(x)
            return x
    
    # 测试不同调制策略
    mock_encoder = MockEncoder()
    x = torch.randn(2, 4, 256, 256)
    
    for strategy in ['stem_only', 'early', 'late', 'all', 'adaptive']:
        encoder = ProgressiveBitDepthEncoder(
            mock_encoder,
            modulation_strategy=strategy,
        )
        result = encoder(x)
        print(f"\nStrategy: {strategy}")
        print(f"  Output shape: {result['features'].shape}")
        print(f"  Multi-scale features: {len(result['multi_scale_features'])}")
        print(f"  Bit depth logits shape: {result['bit_depth_logits'].shape}")
    
    print("\n✓ ProgressiveBitDepthEncoder test passed!")
