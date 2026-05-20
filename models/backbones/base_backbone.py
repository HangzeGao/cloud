"""
Backbone基类 - 统一接口
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseBackbone(nn.Module, ABC):
    """所有backbone的基类，统一输出接口"""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.pretrained = pretrained
        self.feature_channels = []  # 各阶段特征通道数
        self.strides = []  # 各阶段下采样倍数
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        返回多尺度特征列表，从浅层到深层
        Returns: [c1, c2, c3, c4, c5] 对应不同下采样倍数
        """
        pass
    
    def get_feature_channels(self) -> List[int]:
        """返回各阶段特征通道数"""
        return self.feature_channels
    
    def get_strides(self) -> List[int]:
        """返回各阶段下采样倍数"""
        return self.strides
