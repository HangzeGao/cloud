"""
位深度自适应编码器模块

包含两种自适应编码器实现：
1. BitDepthAdaptiveEncoder: 完整版，支持特征适配
2. SimpleBitDepthAdaptiveEncoder: 简化版，只做位深度估计

支持8-13位深度范围（6个类别）。
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bit_depth_estimators import create_bit_depth_estimator
from .feature_adapters import create_feature_adapter
from .bit_depth_config import get_num_bit_depths


class BitDepthAdaptiveEncoder(nn.Module):
    """
    组合B1+C3的自适应编码器（完整版）
    
    支持多种位深度估计器和特征适配器
    支持8-13位深度范围（6个类别）
    """
    
    def __init__(
        self, 
        base_encoder: nn.Module, 
        in_channels: int = 4, 
        feature_dim: int = None,
        estimator_type: str = 'conv',
        adapter_type: str = 'ultra_light',
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.estimator_type = estimator_type
        self.adapter_type = adapter_type

        # 初始化位深度估计器
        self.bit_depth_estimator = create_bit_depth_estimator(
            estimator_type=estimator_type,
            in_channels=in_channels,
        )
        
        # 自动检测特征维度
        if feature_dim is None:
            if hasattr(base_encoder, 'out_channels'):
                out_ch = base_encoder.out_channels
                if isinstance(out_ch, (list, tuple)):
                    feature_dim = out_ch[-1]
                else:
                    feature_dim = out_ch
            else:
                feature_dim = 512

        # 初始化特征适配器
        self.feature_adapter = create_feature_adapter(
            adapter_type=adapter_type,
            feature_dim=feature_dim,
            num_bit_depths=get_num_bit_depths()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：先估计位深度，然后通过编码器，最后适配特征
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            编码特征（经位深度适配，如果启用）
        """
        bit_depth_logits, estimated_bd = self.bit_depth_estimator(x)
        features = self.base_encoder(x)
        
        if self.feature_adapter is not None:
            if isinstance(features, (list, tuple)):
                adapted_features = [feat for feat in features]
                if len(adapted_features) > 0:
                    last_feat = adapted_features[-1]
                    adapted_features[-1] = self._adapt_feature(last_feat, bit_depth_logits)
                features = adapted_features if len(adapted_features) > 1 else adapted_features[0]
            else:
                features = self._adapt_feature(features, bit_depth_logits)
        
        self._last_bit_depth_logits = bit_depth_logits
        self._last_estimated_bd = estimated_bd
        
        return features
    
    def _adapt_feature(self, feature: torch.Tensor, bit_depth_logits: torch.Tensor) -> torch.Tensor:
        """适配单尺度特征"""
        if self.feature_adapter is None:
            return feature
        
        if hasattr(self.feature_adapter, 'feature_dim') and feature.size(1) != self.feature_adapter.feature_dim:
            return feature
        
        return self.feature_adapter(feature, bit_depth_logits)
    
    def get_bit_depth_info(self) -> Optional[Dict[str, torch.Tensor]]:
        """获取最近一次前向传播的位深度信息"""
        if hasattr(self, '_last_bit_depth_logits'):
            return {
                'logits': self._last_bit_depth_logits,
                'estimated': self._last_estimated_bd,
                'probs': F.softmax(self._last_bit_depth_logits, dim=1),
            }
        return None


# 预定义配置
ENCODER_CONFIGS = {
    'bit_depth': {
        'class': BitDepthAdaptiveEncoder,
        'description': '完整版，支持位深度估计和特征适配',
        'features': 'estimate + adapt',
    },
}


def create_adaptive_encoder(
    encoder_type: str,
    base_encoder: nn.Module,
    in_channels: int = 4,
    feature_dim: int = None,
    estimator_type: str = 'conv',
    adapter_type: str = 'ultra_light',
):
    if encoder_type not in ENCODER_CONFIGS:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. "
            f"Choose from {list(ENCODER_CONFIGS.keys())}"
        )
    
    encoder_class = ENCODER_CONFIGS[encoder_type]['class']

    return encoder_class(
        base_encoder=base_encoder,
        in_channels=in_channels,
        feature_dim=feature_dim,
        estimator_type=estimator_type,
        adapter_type=adapter_type,
    )
