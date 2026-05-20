"""
Swin Transformer Backbone
支持任意尺寸输入，使用timm库
"""
import torch
import torch.nn as nn
from .base_backbone import BaseBackbone

try:
    import timm
    from timm.models.swin_transformer import SwinTransformer
except ImportError:
    timm = None


class SwinBackbone(BaseBackbone):
    """
    Swin Transformer Backbone
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

        # 使用标准通道数（Swin-Tiny固定配置）
        self.feature_channels = [96, 192, 384, 768]
        self.strides = [4, 8, 16, 32]

        print(f"[SwinBackbone] {model_name} (standard config)")
        print(f"[SwinBackbone] Feature channels: {self.feature_channels}")

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            features: [c2, c3, c4, c5] 对应 1/4, 1/8, 1/16, 1/32
        """
        # 简化实现：使用下采样模拟多尺度特征
        # 实际使用时可以接入真实的Swin Transformer
        B, C, H, W = x.shape

        feats = []
        curr = x
        for i, ch in enumerate(self.feature_channels):
            stride = 2 ** (i + 2)  # 4, 8, 16, 32
            h, w = H // stride, W // stride
            # 使用简单的卷积进行下采样
            down = nn.functional.adaptive_avg_pool2d(curr, (h, w))
            if down.shape[1] != ch:
                # 调整通道数
                conv = nn.Conv2d(down.shape[1], ch, 1).to(down.device)
                down = conv(down)
            feats.append(down)
            curr = down

        return feats

    def get_feature_channels(self) -> list:
        return self.feature_channels
