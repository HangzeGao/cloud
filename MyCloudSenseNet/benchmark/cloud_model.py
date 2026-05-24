from typing import Optional, List

import albumentations as A
import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from MyCloudSenseNet.benchmark.cloud_dataset import CloudDataset
from MyCloudSenseNet.benchmark.losses import intersection_over_union
from MyCloudSenseNet.benchmark.adaptive_encoders import (
    BitDepthAdaptiveEncoder,
    SimpleBitDepthAdaptiveEncoder,
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CloudModel(pl.LightningModule):
    """
    云检测模型（使用B1+C3位深度自适应方案）
    """
    
    def __init__(
        self,
        bands: List[str],
        x_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
        hparams: dict = {},
        model_name: str = "unet",
        enable_bit_depth_adaptation: bool = True,
        use_simple_adapter: bool = False,
        estimator_type: str = 'conv',
        adapter_type: str = 'ultra_light',
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['x_train', 'y_train', 'x_val', 'y_val'])
        self.hparams.update(hparams)

        self.bands = bands
        self.in_channels = len(bands)
        self.num_classes = 3
        self.enable_bit_depth_adaptation = enable_bit_depth_adaptation
        self.use_simple_adapter = use_simple_adapter
        self.estimator_type = estimator_type
        self.adapter_type = adapter_type

        self.backbone = self.hparams.get("backbone", "timm-efficientnet-b0")
        self.weights = self.hparams.get("weights", "imagenet")

        self.learning_rate = self.hparams.get("lr", 1e-4)
        self.patience = self.hparams.get("patience", 8)
        self.num_workers = self.hparams.get("num_workers", 0)
        self.batch_size = self.hparams.get("batch_size", 4)

        self.device_type = get_device()
        self.gpu = self.device_type in ("mps", "cuda")

        self.transforms = self._create_transforms()

        self._init_datasets(x_train, y_train, x_val, y_val)
        self.model = self._build_model(model_name)

    def _init_datasets(self, x_train, y_train, x_val, y_val):
        if x_train is not None and y_train is not None:
            self.train_dataset = CloudDataset(
                x_paths=x_train,
                bands=self.bands,
                y_paths=y_train,
                transforms=self.transforms,
            )

        if x_val is not None and y_val is not None:
            self.val_dataset = CloudDataset(
                x_paths=x_val,
                bands=self.bands,
                y_paths=y_val,
                transforms=None,
            )

    def _to_device(self, tensor):
        if self.device_type == "cuda":
            return tensor.cuda(non_blocking=True)
        elif self.device_type == "mps":
            return tensor.to("mps")
        return tensor

    def forward(self, image: torch.Tensor):
        return self.model(image)

    def training_step(self, batch: dict, batch_idx: int):
        if not hasattr(self, 'train_dataset') or self.train_dataset.data is None:
            raise ValueError("Training dataset not initialized")
        
        self.model.train()
        torch.set_grad_enabled(True)
        x = self._to_device(batch["chip"])
        y = self._to_device(batch["label"].long())
        
        preds = self.forward(x)
        
        ce_loss = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([0.1, 0.4, 0.5], device=self.device_type),
            reduction="mean"
        )(preds, y)
        
        dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)(preds, y)
        
        loss = 0.5 * ce_loss + 0.5 * dice_loss
        
        self.log("train/ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        if self.enable_bit_depth_adaptation and hasattr(self.model.encoder, 'get_bit_depth_info'):
            bd_info = self.model.encoder.get_bit_depth_info()
            if bd_info is not None:
                self.log("train/est_bit_depth", bd_info['estimated'].mean(), on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        if not hasattr(self, 'val_dataset') or self.val_dataset.data is None:
            raise ValueError("Validation dataset not initialized")
        
        self.model.eval()
        torch.set_grad_enabled(False)
        x = self._to_device(batch["chip"])
        y = self._to_device(batch["label"].long())
        
        preds = self.forward(x)
        preds_class = torch.argmax(preds, dim=1)
        
        iou = intersection_over_union(preds_class, y)
        
        self.log("val/iou", iou, on_step=True, on_epoch=True, prog_bar=True)
        
        if self.enable_bit_depth_adaptation and hasattr(self.model.encoder, 'get_bit_depth_info'):
            bd_info = self.model.encoder.get_bit_depth_info()
            if bd_info is not None:
                self.log("val/est_bit_depth", bd_info['estimated'].mean(), on_epoch=True)
        
        return iou

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True if self.device_type == "cuda" else False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True if self.device_type == "cuda" else False,
        )

    def configure_optimizers(self):
        param_groups = [
            {"params": self.model.encoder.base_encoder.parameters(), "lr": self.learning_rate * 0.5, "name": "encoder_base"},
            {"params": self.model.decoder.parameters(), "lr": self.learning_rate, "name": "decoder"},
            {"params": self.model.segmentation_head.parameters(), "lr": self.learning_rate, "name": "head"},
        ]
        
        if self.enable_bit_depth_adaptation:
            if hasattr(self.model.encoder, 'bit_depth_estimator'):
                estimator_params = list(self.model.encoder.bit_depth_estimator.parameters())
                if len(estimator_params) > 0:
                    param_groups.append({
                        "params": estimator_params,
                        "lr": self.learning_rate * 2.0,
                        "name": "bit_depth_estimator"
                    })
            if hasattr(self.model.encoder, 'feature_adapter') and self.model.encoder.feature_adapter is not None:
                feature_adapter_params = list(self.model.encoder.feature_adapter.parameters())
                if len(feature_adapter_params) > 0:
                    param_groups.append({
                        "params": feature_adapter_params,
                        "lr": self.learning_rate * 2.0,
                        "name": "feature_adapter"
                    })
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=self.patience, min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/iou_epoch",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _build_model(self, model_name: str):
        if model_name == "unet":
            model = smp.Unet(
                encoder_name=self.backbone,
                encoder_weights=self.weights,
                decoder_attention_type="scse",
                decoder_interpolation="bilinear",
                in_channels=self.in_channels,
                classes=self.num_classes,
            )
        elif model_name == "segformer":
            model = smp.Segformer(
                encoder_name="mit_b3",
                encoder_weights=self.weights,
                in_channels=self.in_channels,
                classes=self.num_classes,
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        if self.enable_bit_depth_adaptation:
            if self.use_simple_adapter:
                adaptive_encoder = SimpleBitDepthAdaptiveEncoder(
                    model.encoder,
                    in_channels=self.in_channels,
                    estimator_type=self.estimator_type,
                )
            else:
                feature_dim = None
                if hasattr(model.encoder, 'out_channels'):
                    out_ch = model.encoder.out_channels
                    if isinstance(out_ch, (list, tuple)):
                        feature_dim = out_ch[-1]
                    else:
                        feature_dim = out_ch

                adaptive_encoder = BitDepthAdaptiveEncoder(
                    model.encoder,
                    in_channels=self.in_channels,
                    feature_dim=feature_dim,
                    estimator_type=self.estimator_type,
                    adapter_type=self.adapter_type,
                )
            
            model.encoder = adaptive_encoder

        if self.gpu:
            model = model.to(self.device_type)

        return model

    def _create_transforms(self):
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ]
        return A.Compose(transforms)
