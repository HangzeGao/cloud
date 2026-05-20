"""
EfficientNet Backbone - 高效轻量
"""
import torch
import torch.nn as nn
from .base_backbone import BaseBackbone


class EfficientNetBackbone(BaseBackbone):
    """
    EfficientNet Backbone
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

        # EfficientNet-B3 标准通道数
        self.feature_channels = [24, 32, 48, 384]
        self.strides = [4, 8, 16, 32]

        # 创建简单的下采样层
        self.stages = nn.ModuleList()
        in_ch = in_channels

        for i, out_ch in enumerate(self.feature_channels):
            if i == 0:
                # 第一层
                stage = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            else:
                # 后续层
                stage = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            self.stages.append(stage)
            in_ch = out_ch

        print(f"[EfficientNetBackbone] {model_name} (simulated)")
        print(f"[EfficientNetBackbone] Feature channels: {self.feature_channels}")

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            features: [c2, c3, c4, c5] 对应 1/4, 1/8, 1/16, 1/32
        """
        features = []
        curr = x

        for stage in self.stages:
            curr = stage(curr)
            features.append(curr)

        return features

    def get_feature_channels(self) -> list:
        return self.feature_channels
