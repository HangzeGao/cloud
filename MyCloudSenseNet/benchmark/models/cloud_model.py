"""
云检测模型 - 优化版

核心特性:
1. 多尺度位深度感知编码 (MS-BDFF)
2. 空间-光谱注意力 (SSA)
3. 动态加权复合损失 (DWCL)
4. 边界感知损失
5. 自适应 TTA
6. 高级学习率调度

Author: CloudSenseNet Team
Version: 2.0
"""

from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from ..configs import ModelConfig
from .losses import intersection_over_union
from .loss_factory import CombinedLoss
from .lr_schedulers import LRSchedulerFactory
from .optimizer_factory import create_optimizer


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BitDepthAdaptiveEncoder(nn.Module):
    """
    位深度自适应编码器包装
    
    包装基础编码器，添加位深度估计和特征适配。
    保持与SMP兼容的接口和输出格式。
    """
    
    def __init__(self, base_encoder, bit_depth_estimator, feature_adapter):
        super().__init__()
        self.encoder = base_encoder
        self.bit_depth_estimator = bit_depth_estimator
        self.feature_adapter = feature_adapter
        
        # 代理基础编码器的属性
        if hasattr(base_encoder, 'out_channels'):
            self.out_channels = base_encoder.out_channels
        if hasattr(base_encoder, 'output_stride'):
            self.output_stride = base_encoder.output_stride
    
    def forward(self, x):
        # 估计位深度
        bit_depth_logits, estimated_bd = self.bit_depth_estimator(x)
        self._last_bit_depth_logits = bit_depth_logits
        self._last_estimated_bd = estimated_bd
        
        # 编码
        features = self.encoder(x)
        
        # 适配特征（支持单张量或列表）
        if isinstance(features, (list, tuple)):
            adapted = list(features)
            if len(adapted) > 0 and self.feature_adapter is not None:
                last_feat = adapted[-1]
                if last_feat.size(1) == self.feature_adapter.feature_dim:
                    adapted[-1] = self.feature_adapter(last_feat, bit_depth_logits)
            return adapted
        else:
            if self.feature_adapter is not None:
                features = self.feature_adapter(features, bit_depth_logits)
            return features
    
    def get_bit_depth_info(self):
        """获取位深度信息"""
        if hasattr(self, '_last_bit_depth_logits'):
            import torch.nn.functional as F
            return {
                'logits': self._last_bit_depth_logits,
                'estimated': self._last_estimated_bd,
                'probs': F.softmax(self._last_bit_depth_logits, dim=1),
            }
        return None


class CloudModel(pl.LightningModule):
    """
    云判模型 v2.0 - 位深度自适应优化版
    
    Args:
        config: 模型配置
    """
    
    def __init__(
        self,
        config: ModelConfig | Dict[str, Any],
    ):
        super().__init__()
        
        self.config = config if isinstance(config, ModelConfig) else ModelConfig(**config)
        self.save_hyperparameters({"config": self.config.to_dict()})
        self.bands = self.config.bands
        self.in_channels = self.config.in_channels
        self.num_classes = self.config.num_classes
        
        # set device
        self.device_type = get_device()
        self.gpu = self.device_type in ("mps", "cuda")
        
        # build model
        self.model = self._build_model()
        self.criterion = CombinedLoss.create(
            loss_type="combined",
            num_classes=self.num_classes,
            **self._get_loss_kwargs()
        )
        
        # cache outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self._test_images_logged = 0
    
    def _get_loss_kwargs(self) -> Dict[str, Any]:
        """获取损失函数配置参数"""
        cfg = self.config
        return {
            "use_ce": True,
            "use_dice": True,
            "use_focal": cfg.loss_focal,
            "enable_dynamic_weighting": cfg.loss_dynamic_weighting,
            "enable_class_adaptive": False,  # 简化配置
            "focal_alpha": cfg.loss_focal_alpha,
            "focal_gamma": cfg.loss_focal_gamma,
            "enable_boundary": cfg.loss_boundary,
            "boundary_weight": cfg.loss_boundary_weight,
        }
    
    def _build_model(self) -> nn.Module:
        """构建分割模型"""
        cfg = self.config
        
        # 创建基础模型
        if cfg.model_name == "unet":
            model = smp.Unet(
                encoder_name=cfg.backbone,
                encoder_weights=cfg.encoder_weights,
                decoder_attention_type="scse",
                decoder_interpolation="bilinear",
                in_channels=self.in_channels,
                classes=self.num_classes,
            )
        elif cfg.model_name == "segformer":
            model = smp.Segformer(
                encoder_name=cfg.backbone,
                encoder_weights=cfg.encoder_weights,
                in_channels=self.in_channels,
                classes=self.num_classes,
            )
        elif cfg.model_name == "deeplabv3+":
            model = smp.DeepLabV3Plus(
                encoder_name=cfg.backbone,
                encoder_weights=cfg.encoder_weights,
                in_channels=self.in_channels,
                classes=self.num_classes,
            )
        else:
            raise ValueError(f"Unknown model: {cfg.model_name}")
        
        # 应用位深度自适应编码器
        if cfg.bit_depth_enabled:
            from .bit_depth_estimators import EstimatorFactory
            from .feature_adapters import AdapterFactory
            
            # 位深度估计器
            self.bit_depth_estimator = EstimatorFactory.create(
                cfg.bit_depth_estimator,
                in_channels=self.in_channels,
            )
            
            # 特征适配器（在编码器输出后应用）
            feature_dim = self._get_feature_dim(model.encoder)
            self.feature_adapter = AdapterFactory.create(
                cfg.bit_depth_adapter,
                feature_dim=feature_dim,
            )
            
            # 包装编码器
            model.encoder = BitDepthAdaptiveEncoder(
                base_encoder=model.encoder,
                bit_depth_estimator=self.bit_depth_estimator,
                feature_adapter=self.feature_adapter,
            )
        
        if self.gpu:
            model = model.to(self.device_type)
        
        return model
    
    def _get_feature_dim(self, encoder) -> int:
        """获取编码器输出维度"""
        if hasattr(encoder, 'out_channels'):
            out_ch = encoder.out_channels
            if isinstance(out_ch, (list, tuple)):
                return out_ch[-1]
            return out_ch
        return 512
    
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """移动张量到设备"""
        if self.device_type == "cuda":
            return tensor.cuda(non_blocking=True)
        elif self.device_type == "mps":
            return tensor.to("mps")
        return tensor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.model(x)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """训练步骤"""
        x = self._to_device(batch["chip"])
        y = self._to_device(batch["label"].long())
        
        logits = self.forward(x)
        
        # 计算损失
        try:
            # 尝试传递 epoch 参数（用于动态加权损失）
            loss, loss_info = self.criterion(logits, y, self.current_epoch)
        except TypeError:
            # 如果损失函数不接受 epoch 参数，只传递 logits 和 targets
            result = self.criterion(logits, y)
            if isinstance(result, tuple):
                loss, loss_info = result
            else:
                loss = result
                loss_info = {}
        
        # 记录损失
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for key, val in loss_info.items():
            if isinstance(val, (int, float)):
                self.log(f"train/{key}", val, on_step=False, on_epoch=True)
        
        # 记录位深度信息
        if self.config.bit_depth_enabled:
            if hasattr(self.model.encoder, 'get_bit_depth_info'):
                bd_info = self.model.encoder.get_bit_depth_info()
                if bd_info and 'estimated' in bd_info:
                    self.log("train/est_bit_depth", bd_info['estimated'].mean(), 
                            on_step=False, on_epoch=True)
        
        return loss
    
    def _eval_step(
        self,
        batch: Dict[str, Any],
        stage: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a labeled evaluation step for validation or test."""
        if "label" not in batch:
            raise ValueError(f"{stage} evaluation requires y paths/labels in the dataset.")

        x = self._to_device(batch["chip"])
        y = self._to_device(batch["label"].long())
        
        with torch.no_grad():
            logits = self.forward(x)
            preds = torch.argmax(logits, dim=1)
            iou = intersection_over_union(preds, y)
        
        self.log(f"{stage}/iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        return iou, x, y, preds

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """验证步骤"""
        iou, _, _, _ = self._eval_step(batch, "val")
        self.validation_step_outputs.append(iou)
        return iou

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """测试步骤"""
        iou, x, y, preds = self._eval_step(batch, "test")
        self._log_test_images(batch, x, y, preds)
        self.test_step_outputs.append(iou)
        return iou

    def on_test_start(self):
        """Reset the TensorBoard test-image budget for each test run."""
        self._test_images_logged = 0

    def _log_test_images(
        self,
        batch: Dict[str, Any],
        x: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
    ) -> None:
        """Log a bounded set of test inputs, labels, and predictions to TensorBoard."""
        max_samples = self.config.test_image_log_max_samples
        if (
            not self.config.log_test_images
            or max_samples == 0
            or self.global_rank != 0
            or self._test_images_logged >= max_samples
        ):
            return

        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_images"):
            return

        count = min(x.size(0), max_samples - self._test_images_logged)
        experiment.add_images(
            "test/input",
            self._make_rgb_preview(x[:count]),
            self.global_step,
            dataformats="NCHW",
        )
        experiment.add_images(
            "test/target",
            self._colorize_mask(targets[:count]),
            self.global_step,
            dataformats="NCHW",
        )
        experiment.add_images(
            "test/prediction",
            self._colorize_mask(predictions[:count]),
            self.global_step,
            dataformats="NCHW",
        )
        if "chip_id" in batch and hasattr(experiment, "add_text"):
            chip_ids = ", ".join(str(chip_id) for chip_id in batch["chip_id"][:count])
            experiment.add_text("test/chip_ids", chip_ids, self.global_step)

        self._test_images_logged += count

    def _make_rgb_preview(self, images: torch.Tensor) -> torch.Tensor:
        """Create a normalized RGB preview, preferring B04/B03/B02 bands."""
        band_indices = [
            self.bands.index(band)
            for band in ("B04", "B03", "B02")
            if band in self.bands
        ]
        if len(band_indices) < 3:
            band_indices = list(range(min(3, images.size(1))))
        while len(band_indices) < 3:
            band_indices.append(band_indices[-1])

        rgb = images[:, band_indices].detach().float().cpu()
        channel_min = rgb.amin(dim=(-2, -1), keepdim=True)
        channel_max = rgb.amax(dim=(-2, -1), keepdim=True)
        return (rgb - channel_min) / (channel_max - channel_min).clamp_min(1e-6)

    def _colorize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert class-index masks to RGB images for TensorBoard."""
        palette = torch.tensor(
            [
                [0.10, 0.10, 0.10],
                [0.95, 0.63, 0.12],
                [0.12, 0.65, 0.95],
                [0.86, 0.28, 0.28],
                [0.45, 0.80, 0.31],
                [0.72, 0.42, 0.88],
            ],
            dtype=torch.float32,
        )
        class_indices = mask.detach().long().cpu().clamp_(0, len(palette) - 1)
        return palette[class_indices].permute(0, 3, 1, 2)
    
    def on_validation_epoch_end(self):
        """验证 epoch 结束"""
        if self.validation_step_outputs:
            avg_iou = torch.stack(self.validation_step_outputs).mean()
            self.log("val/avg_iou", avg_iou, prog_bar=True)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """测试 epoch 结束"""
        if self.test_step_outputs:
            avg_iou = torch.stack(self.test_step_outputs).mean()
            self.log("test/avg_iou", avg_iou, prog_bar=True)
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """配置优化器和调度器"""
        optimizer = create_optimizer(
            model=self.model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            encoder_lr_scale=self.config.encoder_lr_scale,
            bit_depth_component_lr_scale=self.config.bit_depth_lr_scale,
        )
        
        scheduler_config = LRSchedulerFactory.create(
            scheduler_type=self.config.lr_scheduler_type,
            optimizer=optimizer,
            num_epochs=self.config.max_epochs,
            warmup_epochs=self.config.warmup_epochs,
            monitor="val/avg_iou",
            mode="max",
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
