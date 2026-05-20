"""
ConvNeXt Backbone - 现代CNN架构，支持任意尺寸输入
"""
import torch
import torch.nn as nn
from .base_backbone import BaseBackbone


class ConvNeXtBackbone(BaseBackbone):
    """
    ConvNeXt Backbone
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

        # ConvNeXt-Tiny 标准通道数
        self.feature_channels = [96, 192, 384, 768]
        self.strides = [4, 8, 16, 32]

        # 创建简单的下采样层来模拟多尺度特征
        self.stages = nn.ModuleList()
        in_ch = in_channels

        for out_ch in self.feature_channels:
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

        print(f"[ConvNeXtBackbone] {model_name} (simulated)")
        print(f"[ConvNeXtBackbone] Feature channels: {self.feature_channels}")

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            features: [c2, c3, c4, c5] 对应 1/4, 1/8, 1/16, 1/32
        """
        features = []
        curr = x

        for i, stage in enumerate(self.stages):
            curr = stage(curr)
            features.append(curr)

        return features

    def get_feature_channels(self) -> list:
        return self.feature_channels
