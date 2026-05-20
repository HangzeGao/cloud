"""
ResNet Backbone - 经典架构
"""
import torch
import torch.nn as nn
from torchvision import models
from .base_backbone import BaseBackbone


class ResNetBackbone(BaseBackbone):
    """
    ResNet Backbone
    经典可靠的CNN架构
    输出多尺度特征 [1/4, 1/8, 1/16, 1/32]
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        in_channels: int = 3
    ):
        super().__init__(pretrained=pretrained)
        
        self.model_name = model_name
        
        # 加载预训练模型
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            self.feature_channels = [64, 128, 256, 512]
        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            self.feature_channels = [64, 128, 256, 512]
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.feature_channels = [256, 512, 1024, 2048]
        elif model_name == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            resnet = models.resnet101(weights=weights)
            self.feature_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")
        
        # 提取各层
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        
        self.strides = [4, 8, 16, 32]
        
        # 修改第一层以支持不同的输入通道数
        if in_channels != 3:
            self.stem[0] = nn.Conv2d(
                in_channels, 
                self.stem[0].out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
        
        print(f"[ResNetBackbone] {model_name} loaded")
        print(f"[ResNetBackbone] Feature channels: {self.feature_channels}")
    
    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            features: [c2, c3, c4, c5] 对应 1/4, 1/8, 1/16, 1/32
        """
        x = self.stem(x)
        
        c2 = self.layer1(x)   # 1/4
        c3 = self.layer2(c2)  # 1/8
        c4 = self.layer3(c3)  # 1/16
        c5 = self.layer4(c4)  # 1/32
        
        return [c2, c3, c4, c5]
    
    def get_feature_channels(self) -> list:
        return self.feature_channels


class ResNeXtBackbone(BaseBackbone):
    """
    ResNeXt Backbone - 使用分组卷积改进的ResNet
    """
    
    def __init__(
        self,
        model_name: str = "resnext50_32x4d",
        pretrained: bool = True,
        in_channels: int = 3
    ):
        super().__init__(pretrained=pretrained)
        
        self.model_name = model_name
        
        if model_name == "resnext50_32x4d":
            weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
            resnext = models.resnext50_32x4d(weights=weights)
            self.feature_channels = [256, 512, 1024, 2048]
        elif model_name == "resnext101_32x8d":
            weights = models.ResNeXt101_32X8D_Weights.DEFAULT if pretrained else None
            resnext = models.resnext101_32x8d(weights=weights)
            self.feature_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown ResNeXt model: {model_name}")
        
        self.stem = nn.Sequential(
            resnext.conv1,
            resnext.bn1,
            resnext.relu,
            resnext.maxpool
        )
        
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4
        
        self.strides = [4, 8, 16, 32]
        
        if in_channels != 3:
            self.stem[0] = nn.Conv2d(
                in_channels,
                self.stem[0].out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
        
        print(f"[ResNeXtBackbone] {model_name} loaded")
        print(f"[ResNeXtBackbone] Feature channels: {self.feature_channels}")
    
    def forward(self, x: torch.Tensor) -> list:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]
