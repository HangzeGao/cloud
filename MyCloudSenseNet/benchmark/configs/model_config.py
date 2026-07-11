"""
模型配置 - 简化版

核心配置参数，去除冗余嵌套。
"""

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    """
    云检测模型配置
    
    简化设计，核心参数直接可访问。
    """
    
    # 模型架构
    model_name: str = "segformer"  # unet, segformer, deeplabv3+, fpn
    backbone: str = "mit_b2" # mit_b0 ~ mit_b5 / resnet34, resnet50 / timm-efficientnet-b0 ~ timm-efficientnet-b7
    encoder_weights: str = "imagenet"
    bands: list[str] = field(default_factory=lambda: ["B02", "B03", "B04"])
    num_classes: int = 2
    normalization_stats_path: str | None = None
    
    # 训练参数
    learning_rate: float = 1e-3
    batch_size: int = 24
    max_epochs: int = 100
    patience: int = max_epochs // 5
    num_workers: int = 16
    
    # 优化器参数
    encoder_lr_scale: float = 0.5
    bit_depth_lr_scale: float = 2.0
    weight_decay: float = 1e-4
    
    # 学习率调度
    lr_scheduler_type: str = "cosine_warmup"  # plateau, cosine, cosine_warmup, one_cycle
    warmup_epochs: int = max_epochs // 10
    
    # 位深度自适应
    bit_depth_enabled: bool = True
    bit_depth_estimator: str = "conv"  # minimal, conv, statistical
    bit_depth_adapter: str = "hard_gate"  # default, hard_gate, light
    bit_depth_classes: list[int] = field(default_factory=lambda: [8, 10, 12, 14, 16])
    bit_depth_loss_weight: float = 0.1
    
    # 损失函数
    loss_focal: bool = True
    loss_focal_alpha: float = 0.25
    loss_focal_gamma: float = 2.0
    loss_boundary: bool = True
    loss_boundary_weight: float = 2.0
    loss_dynamic_weighting: bool = True

    # Metrics
    iou_class_weights: list[float] = field(default_factory=lambda: [1.0, 1.5])
    
    # TTA 推理
    use_tta: bool = True
    tta_strategy: str = "light"  # none, adaptive, light, standard
    tta_threshold: float = 0.9

    # TensorBoard test visualization
    log_test_images: bool = True
    test_image_log_max_samples: int = 64

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate config values early so failures are explicit."""
        if self.model_name not in {"unet", "segformer", "deeplabv3+", "fpn"}:
            raise ValueError(f"Unsupported model_name: {self.model_name}")
        if not self.bands:
            raise ValueError("bands must contain at least one spectral band")
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if len(self.iou_class_weights) != self.num_classes:
            raise ValueError("iou_class_weights length must match num_classes")
        if any(weight < 0 for weight in self.iou_class_weights):
            raise ValueError("iou_class_weights must be non-negative")
        if sum(self.iou_class_weights) <= 0:
            raise ValueError("iou_class_weights must contain at least one positive weight")
        if self.test_image_log_max_samples < 0:
            raise ValueError("test_image_log_max_samples must be >= 0")
        if not self.bit_depth_classes:
            raise ValueError("bit_depth_classes must contain at least one value")
        if len(set(self.bit_depth_classes)) != len(self.bit_depth_classes):
            raise ValueError("bit_depth_classes must be unique")
        if any(bit_depth <= 0 for bit_depth in self.bit_depth_classes):
            raise ValueError("bit_depth_classes must contain positive values")
        if self.bit_depth_loss_weight < 0:
            raise ValueError("bit_depth_loss_weight must be >= 0")

    @property
    def in_channels(self) -> int:
        """输入通道数由 bands 唯一推导，避免配置不一致。"""
        return len(self.bands)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)


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
            bit_depth_adapter="hard_gate",
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

    @staticmethod
    def high_accuracy() -> ModelConfig:
        """高精度配置"""
        return ModelConfig(
            backbone="timm-efficientnet-b4",
            learning_rate=5e-5,
            batch_size=2,
            max_epochs=150,
            warmup_epochs=10,
            encoder_lr_scale=0.3,
            bit_depth_estimator="statistical",
            bit_depth_adapter="default",
            loss_boundary_weight=3.0,
            tta_threshold=0.95,
        )
