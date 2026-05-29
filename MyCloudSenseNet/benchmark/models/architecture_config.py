"""
模型配置 - 简化版

核心配置参数，去除冗余嵌套。
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """
    云检测模型配置
    
    简化设计，核心参数直接可访问。
    """
    
    # 模型架构
    model_name: str = "unet"  # unet, segformer, deeplabv3+
    backbone: str = "timm-efficientnet-b0"
    encoder_weights: str = "imagenet"
    in_channels: int = 4
    num_classes: int = 3
    
    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 4
    max_epochs: int = 100
    num_workers: int = 4
    warmup_epochs: int = 5
    
    # 优化器参数
    encoder_lr_scale: float = 0.5
    bit_depth_lr_scale: float = 2.0
    weight_decay: float = 1e-4
    
    # 学习率调度
    lr_scheduler_type: str = "cosine_warmup"  # plateau, cosine, cosine_warmup, one_cycle
    
    # 位深度自适应
    bit_depth_enabled: bool = True
    bit_depth_estimator: str = "conv"  # minimal, conv, statistical
    bit_depth_adapter: str = "multiscale_light"  # ultra_light, light, multiscale_light
    bit_depth_multiscale: bool = True
    bit_depth_ssa: bool = True  # 空间-光谱注意力
    
    # 损失函数
    loss_focal: bool = True
    loss_focal_alpha: float = 0.25
    loss_focal_gamma: float = 2.0
    loss_boundary: bool = True
    loss_boundary_weight: float = 2.0
    loss_dynamic_weighting: bool = True
    
    # TTA 推理
    tta_strategy: str = "adaptive"  # none, adaptive, light, standard
    tta_threshold: float = 0.9
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)
    
    def save(self, path: Path):
        """保存配置"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        """加载配置"""
        with open(path) as f:
            return cls(**json.load(f))


class Configs:
    """预定义配置模板"""
    
    @staticmethod
    def fast() -> ModelConfig:
        """快速实验配置"""
        return ModelConfig(
            model_name="unet",
            backbone="timm-efficientnet-b0",
            learning_rate=2e-4,
            max_epochs=8,
            bit_depth_adapter="ultra_light",
            bit_depth_multiscale=False,
            bit_depth_ssa=False,
            loss_focal=False,
            loss_boundary=False,
            lr_scheduler_type="one_cycle",
            warmup_epochs=2,
            num_workers=2,
        )
    
    @staticmethod
    def balanced() -> ModelConfig:
        """平衡配置 - 推荐默认"""
        return ModelConfig()
