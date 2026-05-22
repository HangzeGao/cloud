"""
EfficientNet Backbone - 使用 timm 真实模型
高效轻量，支持多种变体 (B0-B7)
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


# EfficientNet 各变体的特征通道配置
EFFICIENTNET_CHANNELS = {
    'efficientnet_b0': [24, 40, 112, 320],
    'efficientnet_b1': [24, 40, 112, 320],
    'efficientnet_b2': [24, 48, 120, 352],
    'efficientnet_b3': [32, 48, 136, 384],
    'efficientnet_b4': [32, 56, 160, 448],
    'efficientnet_b5': [40, 64, 176, 512],
    'efficientnet_b6': [40, 72, 200, 576],
    'efficientnet_b7': [48, 80, 224, 640],
    'efficientnet_b8': [48, 88, 248, 704],
    'efficientnet_l2': [72, 104, 272, 800],
}


class EfficientNetBackbone(BaseBackbone):
    """
    EfficientNet Backbone - 使用 timm 真实模型
    轻量高效，适合部署
    输出多尺度特征 [1/4, 1/8, 1/16, 1/32]
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b3",
        pretrained: bool = True,
        in_channels: int = 3
    ):
        super().__init__(pretrained=pretrained)

        self.model_name = model_name

        if not TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for EfficientNet backbone. "
                "Install with: pip install timm"
            )

        # 验证模型名称
        if model_name not in EFFICIENTNET_CHANNELS:
            valid_models = list(EFFICIENTNET_CHANNELS.keys())
            raise ValueError(
                f"Unknown EfficientNet model: {model_name}. "
                f"Valid models: {valid_models}"
            )

        self.feature_channels = EFFICIENTNET_CHANNELS[model_name]
        self.strides = [4, 8, 16, 32]

        # 使用 timm 创建模型，features_only=True 返回5个尺度特征 (stem + 4 blocks)
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[1, 2, 3, 4],  # 跳过 Level 0，去掉 stride=2 的浅层特征，减少计算量，且与检测/分割 FPN 对齐
            in_chans=in_channels,
        )

        print(f"[EfficientNetBackbone] {model_name} loaded from timm")
        print(f"[EfficientNetBackbone] Feature channels: {self.feature_channels}")
        print(f"[EfficientNetBackbone] Pretrained: {pretrained}")

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            features: [c2, c3, c4, c5] 对应 1/4, 1/8, 1/16, 1/32
        """
        # timm 的 features_only=True 返回多尺度特征列表
        features = self.model(x)
        return features

    def get_feature_channels(self) -> list:
        return self.feature_channels
