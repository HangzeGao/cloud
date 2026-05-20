"""
Decoder基类 - 统一接口
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List


class BaseDecoder(nn.Module, ABC):
    """所有decoder的基类"""
    
    def __init__(self, encoder_channels: List[int], num_classes: int):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.num_classes = num_classes
    
    @abstractmethod
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 编码器输出的多尺度特征列表
        Returns:
            segmentation_map: [B, num_classes, H, W]
        """
        pass
    
    def get_num_classes(self) -> int:
        return self.num_classes
