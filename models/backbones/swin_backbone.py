"""
Swin Transformer Backbone - 使用 timm 真实模型
支持任意尺寸输入
"""
import torch
import torch.nn as nn
from .base_backbone import BaseBackbone

try:
    import timm
    from timm import create_model
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# Swin Transformer 各变体的配置
SWIN_CHANNELS = {
    'swin_tiny_patch4_window7_224': [96, 192, 384, 768],
    'swin_tiny_patch4_window7_384': [96, 192, 384, 768],
    'swin_small_patch4_window7_224': [96, 192, 384, 768],
    'swin_small_patch4_window7_384': [96, 192, 384, 768],
    'swin_base_patch4_window7_224': [128, 256, 512, 1024],
    'swin_base_patch4_window7_384': [128, 256, 512, 1024],
    'swin_base_patch4_window12_384': [128, 256, 512, 1024],
    'swin_large_patch4_window7_224': [192, 384, 768, 1536],
    'swin_large_patch4_window7_384': [192, 384, 768, 1536],
}


class SwinBackbone(BaseBackbone):
    """
    Swin Transformer Backbone - 使用 timm 真实模型
    输出多尺度特征 [1/4, 1/8, 1/16, 1/32]
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        drop_path_rate: float = 0.2,
        in_channels: int = 3
    ):
        super().__init__(pretrained=pretrained)

        self.model_name = model_name

        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for Swin Transformer backbone. "
                "Install with: pip install timm"
            )

        # 验证模型名称
        if model_name not in SWIN_CHANNELS:
            valid_models = list(SWIN_CHANNELS.keys())
            raise ValueError(
                f"Unknown Swin model: {model_name}. "
                f"Valid models: {valid_models}"
            )

        self.feature_channels = SWIN_CHANNELS[model_name]
        self.strides = [4, 8, 16, 32]

        # 使用 timm 创建模型
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4],  # 4个stage的输出
            in_chans=in_channels,
            drop_path_rate=drop_path_rate,
        )

        print(f"[SwinBackbone] {model_name} loaded from timm")
        print(f"[SwinBackbone] Feature channels: {self.feature_channels}")
        print(f"[SwinBackbone] Drop path rate: {drop_path_rate}")
        print(f"[SwinBackbone] Pretrained: {pretrained}")

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            features: [c2, c3, c4, c5] 对应 1/4, 1/8, 1/16, 1/32
        """
        features = self.model(x)
        return features

    def get_feature_channels(self) -> list:
        return self.feature_channels
