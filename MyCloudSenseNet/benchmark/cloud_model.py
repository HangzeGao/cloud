from typing import Optional, List, Tuple, Dict

import albumentations as A
import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from MyCloudSenseNet.benchmark.cloud_dataset import CloudDataset
from MyCloudSenseNet.benchmark.losses import intersection_over_union

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class BitDepthEstimator(nn.Module):
    """
    轻量级位深度估计器
    通过分析输入图像的统计特征估计位深度
    支持 8-bit 到 16-bit 范围 (覆盖常见卫星图像: 8-bit RGB, 10-bit GF1, 12-bit Sentinel-2, 16-bit 高比特图像)
    """
    def __init__(self, in_channels: int = 4, hidden_dim: int = 64, num_bit_depths: int = 9):
        super().__init__()
        # 使用全局统计特征而非空间特征
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # 位深度分类器: 8, 9, 10, 11, 12, 13, 14, 15, 16-bit
        self.num_classes = num_bit_depths
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_bit_depths),  # 9 classes for 8-16 bit depths
        )

        # 位深度值: 8, 9, 10, 11, 12, 13, 14, 15, 16
        self.bit_depths = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15, 16])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: 位深度分类logits [B, 5]
            estimated_bit_depth: 估计的位深度值 [B]
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        # 软估计: 使用softmax加权平均
        probs = F.softmax(logits, dim=1)
        bit_depths = self.bit_depths.to(x.device).float()
        estimated_bit_depth = torch.sum(probs * bit_depths, dim=1)

        return logits, estimated_bit_depth


class BitDepthAdaptiveLayer(nn.Module):
    """
    位深度自适应层
    根据输入位深度动态调整特征提取
    支持 8-bit 到 16-bit (9个类别)
    """
    def __init__(self, in_channels: int, out_channels: int, num_bit_depths: int = 9):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bit_depths = num_bit_depths

        # 多尺度卷积核，对应不同位深度
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_bit_depths)
        ])

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * num_bit_depths, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 自适应注意力
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, num_bit_depths),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, bit_depth_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
            bit_depth_logits: 位深度分类logits [B, num_bit_depths]
        Returns:
            自适应特征 [B, out_channels, H, W]
        """
        # 应用多尺度卷积
        multi_scale_features = [conv(x) for conv in self.convs]

        if bit_depth_logits is not None:
            # 基于位深度估计的软选择
            bit_depth_probs = F.softmax(bit_depth_logits, dim=1)
            # 将概率扩展到空间维度
            weights = bit_depth_probs.view(-1, self.num_bit_depths, 1, 1, 1)
            weighted_features = torch.stack(multi_scale_features, dim=1)  # [B, num_scales, C, H, W]
            weighted_features = (weighted_features * weights).sum(dim=1)  # [B, C, H, W]
            return weighted_features
        else:
            # 如果没有位深度信息，使用注意力机制融合
            stacked = torch.cat(multi_scale_features, dim=1)  # [B, num_scales*C, H, W]
            fused = self.fusion(stacked)

            # 计算注意力权重
            attn_weights = self.attention(fused)  # [B, num_bit_depths]
            attn_weights = attn_weights.view(-1, self.num_bit_depths, 1, 1, 1)

            # 加权融合
            features_tensor = torch.stack(multi_scale_features, dim=1)
            output = (features_tensor * attn_weights).sum(dim=1)
            return output


class AdaptiveInputNorm(nn.Module):
    """
    自适应输入归一化层
    根据估计的位深度动态调整归一化参数
    支持 8-bit 到 16-bit (9个类别)
    """
    def __init__(self, num_channels: int = 4, bit_depths: List[int] = None):
        if bit_depths is None:
            bit_depths = list(range(8, 17))  # [8, 9, 10, 11, 12, 13, 14, 15, 16]
        super().__init__()
        self.num_channels = num_channels
        self.bit_depths = bit_depths
        self.num_bit_depths = len(bit_depths)

        # 为每个位深度学习一组gamma和beta
        self.gamma = nn.Parameter(torch.ones(self.num_bit_depths, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(self.num_bit_depths, num_channels, 1, 1))

        # 学习位深度映射系数
        self.register_buffer('max_vals', torch.tensor([2**bd - 1 for bd in bit_depths]).float())

        # 运行均值和方差
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.momentum = 0.1

    def forward(self, x: torch.Tensor, bit_depth_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        自适应归一化

        Args:
            x: 输入图像 [B, C, H, W]，值范围通常为 [0, 1]
            bit_depth_logits: 位深度分类logits [B, num_bit_depths]
        """
        if bit_depth_logits is not None:
            # 基于位深度估计的软归一化
            probs = F.softmax(bit_depth_logits, dim=1).view(-1, self.num_bit_depths, 1, 1, 1)

            # 加权gamma和beta
            gamma_weighted = (self.gamma * probs).sum(dim=1)  # [B, C, 1, 1]
            beta_weighted = (self.beta * probs).sum(dim=1)

            # 计算归一化统计量
            mean = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True, unbiased=False)

            # 应用自适应归一化
            x_norm = (x - mean) / torch.sqrt(var + 1e-5)
            x_norm = x_norm * gamma_weighted + beta_weighted

            return x_norm
        else:
            # 标准BatchNorm行为
            mean = x.mean(dim=[0, 2, 3])
            var = x.var(dim=[0, 2, 3], unbiased=False)

            if self.training:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                    self.num_batches_tracked += 1
            else:
                mean = self.running_mean
                var = self.running_var

            # 使用第一个位深度的参数作为默认值
            x_norm = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + 1e-5)
            x_norm = x_norm * self.gamma[0] + self.beta[0]

            return x_norm


class AdaptiveEncoder(nn.Module):
    """
    自适应编码器 - 包装现有的smp编码器，添加位深度自适应能力
    支持 8-bit 到 16-bit (9个类别)
    """
    def __init__(self, base_encoder: nn.Module, in_channels: int = 4, num_bit_depths: int = 9):
        super().__init__()
        self.base_encoder = base_encoder
        self.in_channels = in_channels

        # 位深度估计器
        self.bit_depth_estimator = BitDepthEstimator(in_channels=in_channels)

        # 自适应输入归一化
        self.adaptive_norm = AdaptiveInputNorm(num_channels=in_channels)

        # 如果编码器有单独的stem，替换它
        if hasattr(base_encoder, '_conv_stem'):
            # EfficientNet等模型
            stem_out_channels = base_encoder._conv_stem.out_channels
            self.adaptive_stem = nn.Sequential(
                BitDepthAdaptiveLayer(in_channels, stem_out_channels // 2, num_bit_depths),
                BitDepthAdaptiveLayer(stem_out_channels // 2, stem_out_channels, num_bit_depths),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            编码特征
        """
        # 估计位深度
        bit_depth_logits, estimated_bit_depth = self.bit_depth_estimator(x)

        # 应用自适应归一化
        x = self.adaptive_norm(x, bit_depth_logits)

        # 如果使用自适应stem，应用它
        if hasattr(self, 'adaptive_stem'):
            # 这里我们需要处理多尺度特征
            # 为简化，我们使用估计的位深度进行缩放
            scale_factor = 10.0 / estimated_bit_depth.view(-1, 1, 1, 1)
            x = x * scale_factor

        # 通过基础编码器
        features = self.base_encoder(x)

        return features

    def get_bit_depth_info(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取位深度信息，用于监控"""
        logits, estimated = self.bit_depth_estimator(x)
        probs = F.softmax(logits, dim=1)
        return {
            'logits': logits,
            'estimated': estimated,
            'probs': probs,
        }


class CloudModel(pl.LightningModule):
    def __init__(
        self,
        bands: List[str],
        x_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
        hparams: dict = {},
        model_name: str = "unet",
    ):
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        # required
        self.bands = bands
        self.in_channels = len(bands)
        self.num_classes = 3

        # optional modeling params
        self.backbone = self.hparams.get("backbone", "timm-efficientnet-b0") # resnet34 timm-efficientnet-b3
        self.weights = self.hparams.get("weights", "imagenet") # "imagenet"
        self.learning_rate = self.hparams.get("lr", 1e-4) # 1e-3 3e-4
        self.patience = self.hparams.get("patience", 8)
        self.num_workers = self.hparams.get("num_workers", 0)
        self.batch_size = self.hparams.get("batch_size", 4)
        self.device_type = get_device()
        self.gpu = self.device_type in ("mps", "cuda")
        self.transform = None # self._create_transforms()
        self.additional_transform = self._create_additional_transforms()

        # Instantiate datasets, model, and trainer params if provided
        self._init_datasets(x_train, y_train, x_val, y_val)
        self.model = self._build_model(model_name=model_name)

    def _init_datasets(self, x_train, y_train, x_val, y_val):
        if x_train is not None and y_train is not None:
            self.train_dataset = CloudDataset(
                x_paths=x_train,
                bands=self.bands,
                y_paths=y_train,
                transforms=self.transform,
                additional_transforms=self.additional_transform,
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
        if self.train_dataset.data is None:
            raise ValueError(
                "x_train and y_train must be specified when CloudModel is instantiated to run training"
            )
        self.model.train()
        torch.set_grad_enabled(True)
        x = self._to_device(batch["chip"])
        y = self._to_device(batch["label"].long())
        # bit_depth_loss = 0
        # if torch.rand(1).item() < 0.5:
        #     # 混合位深度训练：随机模拟不同位深度
        #     # 随机选择目标位深度: 8, 9, 10, 11, 12
        #     low_bd, high_bd = 8, 13
        #     target_bd = torch.randint(low_bd, high_bd, (1,)).item()
        #     x = self._simulate_bit_depth(x, target_bd)
        #     if hasattr(self.model.encoder, 'bit_depth_estimator'):
        #         bit_depth_logits, estimated_bd = self.model.encoder.bit_depth_estimator(x)
        #         # 简化映射：8-bit -> class 0, 9-bit -> class 1, ..., 16-bit -> class 8
        #         target_class = target_bd - 8  # 直接映射到 0-8
        #         target_class = max(0, min(target_class, 8))  # 确保在有效范围内
        #
        #         target_class_tensor = torch.tensor([target_class] * x.size(0), device=x.device).long()
        #         bit_depth_loss = F.cross_entropy(bit_depth_logits, target_class_tensor)
        preds = self.forward(x)
        ce_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.4, 0.5], device=self.device_type), reduction="mean")(preds, y)
        dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)(preds, y)
        loss = 0.5 * ce_loss + 0.5 * dice_loss# + 0.001 * bit_depth_loss
        self.log(name="ce_loss", value=ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        self.log(name="dice_loss", value=dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        # self.log(name="bit_depth_loss", value=bit_depth_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        self.log(name="loss", value=loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        if self.val_dataset.data is None:
            raise ValueError(
                "x_val and y_val must be specified when CloudModel is instantiated to run validation"
            )
        self.model.eval()
        torch.set_grad_enabled(False)
        x = self._to_device(batch["chip"])
        y = self._to_device(batch["label"].long())
        preds = self.forward(x)
        # preds = torch.softmax(preds, dim=1)[:, 1]
        # preds = (preds > 0.5) * 1
        preds = torch.argmax(preds, dim=1)
        iou = intersection_over_union(preds, y)
        self.log(
            "iou", iou, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return iou

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True if self.device_type=="cuda" else False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True if self.device_type=="cuda" else False,
        )

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        encoder_params = self.model.encoder.parameters()
        decoder_params = list(self.model.decoder.parameters()) + list(self.model.segmentation_head.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.learning_rate * 0.5},
                {"params": decoder_params, "lr": self.learning_rate}
            ],
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=self.patience, min_lr=1e-6)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "iou_epoch",
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

        adaptive_encoder = AdaptiveEncoder(
            model.encoder,
            in_channels=self.in_channels,
        )
        model.encoder = adaptive_encoder

        if self.gpu:
            model = model.to(self.device_type)

        return model

    def _create_transforms(self):
        transforms = [
            # A.ShiftScaleRotate(shift_limit=0.0625, rotate_limit=15, p=0.5),
            # A.GridDistortion(p=0.35),
            # A.RandomCrop(512, 512, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.GaussianBlur(p=0.25),
            # A.RandomBrightnessContrast(
            #     brightness_limit=(-0.2, 0.2),
            #     contrast_limit=(0.2, 0.2),
            #     p=0.5
            # ),
        ]

        return A.Compose(transforms)

    def _create_additional_transforms(self):
        from benchmark.bit_depth_transform import BitDepthSimulation

        additional_transforms = [
            BitDepthSimulation(bit_depth_range=(8, 12), p=1.0),
        ]

        return A.Compose(additional_transforms)

    def _simulate_bit_depth(self, x: torch.Tensor, target_bit_depth: int = 8) -> torch.Tensor:
        """
        模拟特定位深度图像
        用于混合位深度训练

        Args:
            x: 输入图像 [B, C, H, W]，范围 [0, 1]
            target_bit_depth: 目标位深度
        """
        if target_bit_depth > 12:
            return x

        # 将 [0, 1] 映射到 [0, 2^bit_depth - 1]
        max_val = 2 ** target_bit_depth - 1
        x_quantized = torch.floor(x * max_val + 0.5) / max_val

        # 添加量化噪声
        noise = torch.rand_like(x) / max_val
        x_simulated = x_quantized + noise * 0.5

        return torch.clamp(x_simulated, 0, 1)
