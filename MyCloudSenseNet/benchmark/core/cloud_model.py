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

from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import albumentations as A
except ImportError:
    A = None

from benchmark.core.cloud_dataset import CloudDataset
from benchmark.models.losses import intersection_over_union
from benchmark.models.architecture_config import ModelConfig
from benchmark.models.advanced_losses import CombinedLoss
from benchmark.models.training_utils import LRSchedulerFactory, create_optimizer


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
        bands: 光谱波段列表
        config: 模型配置
        x_train/y_train: 训练数据路径
        x_val/y_val: 验证数据路径
    """
    
    def __init__(
        self,
        bands: List[str],
        config: ModelConfig,
        x_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['x_train', 'y_train', 'x_val', 'y_val'])
        
        self.config = config
        self.bands = bands
        self.in_channels = len(bands)
        self.num_classes = config.num_classes
        
        # 设备
        self.device_type = get_device()
        self.gpu = self.device_type in ("mps", "cuda")
        
        # 数据集
        self.transforms = self._create_transforms()
        self._init_datasets(x_train, y_train, x_val, y_val)
        
        # 构建模型
        self.model = self._build_model()
        self.criterion = CombinedLoss.create(
            loss_type="combined",
            num_classes=self.num_classes,
            **self._get_loss_kwargs()
        )
        
        # 验证输出缓存
        self.validation_step_outputs = []
    
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
    
    def _init_datasets(self, x_train, y_train, x_val, y_val):
        """初始化数据集"""
        if x_train is not None and y_train is not None:
            self.train_dataset = CloudDataset(
                x_paths=x_train,
                bands=self.bands,
                y_paths=y_train,
                transforms=self.transforms,
            )
        else:
            self.train_dataset = None
        
        if x_val is not None and y_val is not None:
            self.val_dataset = CloudDataset(
                x_paths=x_val,
                bands=self.bands,
                y_paths=y_val,
                transforms=None,
            )
        else:
            self.val_dataset = None
    
    def _create_transforms(self):
        """创建数据增强"""
        if A:
            return A.Compose([A.D4(p=0.5)])
        return None
    
    def _build_model(self) -> nn.Module:
        """构建分割模型"""
        cfg = self.config
        
        # 创建基础模型
        if cfg.model_name == "unet":
            model = smp.Unet(
                encoder_name=cfg.backbone,
                encoder_weights=cfg.encoder_weights,
                decoder_attention_type="scse",
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
            from benchmark.models.bit_depth_estimators import EstimatorFactory
            from benchmark.models.feature_adapters import AdapterFactory
            
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
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """验证步骤"""
        x = self._to_device(batch["chip"])
        y = self._to_device(batch["label"].long())
        
        with torch.no_grad():
            logits = self.forward(x)
            preds = torch.argmax(logits, dim=1)
            iou = intersection_over_union(preds, y)
        
        self.validation_step_outputs.append(iou)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return iou
    
    def on_validation_epoch_end(self):
        """验证 epoch 结束"""
        if self.validation_step_outputs:
            avg_iou = torch.stack(self.validation_step_outputs).mean()
            self.log("val/avg_iou", avg_iou, prog_bar=True)
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """配置优化器和调度器"""
        optimizer = create_optimizer(
            model=self.model,
            learning_rate=self.config.learning_rate,
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
    
    def train_dataloader(self):
        """训练数据加载器"""
        if self.train_dataset is None:
            return None
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.gpu,
            persistent_workers=self.config.num_workers > 0,
            prefetch_factor=2 if self.config.num_workers > 0 else None,
        )
    
    def val_dataloader(self):
        """验证数据加载器"""
        if self.val_dataset is None:
            return None
        
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.gpu,
        )
    
    def predict_with_tta(
        self,
        x: torch.Tensor,
        tta_strategy: str = "adaptive"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """使用 TTA 预测"""
        self.model.eval()
        
        def forward_fn(img):
            return self.forward(img)
        
        return predict_with_advanced_tta(
            model=forward_fn,
            x=x,
            tta_strategy=tta_strategy,
        )
