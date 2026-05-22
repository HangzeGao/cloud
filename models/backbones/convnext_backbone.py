"""
ConvNeXt Backbone - 使用 timm 真实模型
现代CNN架构，支持任意尺寸输入
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


# ConvNeXt 各变体的配置
CONVNEXT_CHANNELS = {
    'convnext_tiny': [96, 192, 384, 768],
    'convnext_small': [96, 192, 384, 768],
    'convnext_base': [128, 256, 512, 1024],
    'convnext_large': [192, 384, 768, 1536],
    'convnext_large_mlp': [192, 384, 768, 1536],
    'convnext_xlarge': [256, 512, 1024, 2048],
    'convnextv2_tiny': [96, 192, 384, 768],
    'convnextv2_small': [96, 192, 384, 768],
    'convnextv2_base': [128, 256, 512, 1024],
    'convnextv2_large': [192, 384, 768, 1536],
    'convnextv2_huge': [352, 704, 1408, 2816],
}


class ConvNeXtBackbone(BaseBackbone):
    """
    ConvNeXt Backbone - 使用 timm 真实模型
    结合了Transformer的设计理念与CNN的效率
    输出多尺度特征 [1/4, 1/8, 1/16, 1/32]
    """

    def __init__(
        self,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
        in_channels: int = 3
    ):
        super().__init__(pretrained=pretrained)

        self.model_name = model_name

        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for ConvNeXt backbone. "
                "Install with: pip install timm"
            )

        # 验证模型名称
        if model_name not in CONVNEXT_CHANNELS:
            valid_models = list(CONVNEXT_CHANNELS.keys())
            raise ValueError(
                f"Unknown ConvNeXt model: {model_name}. "
                f"Valid models: {valid_models}"
            )

        self.feature_channels = CONVNEXT_CHANNELS[model_name]
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

        print(f"[ConvNeXtBackbone] {model_name} loaded from timm")
        print(f"[ConvNeXtBackbone] Feature channels: {self.feature_channels}")
        print(f"[ConvNeXtBackbone] Drop path rate: {drop_path_rate}")
        print(f"[ConvNeXtBackbone] Pretrained: {pretrained}")

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
