"""Lightning datamodule for cloud segmentation chips."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import pytorch_lightning as pl
import torch

from ..configs import ModelConfig
from .dataset import CloudDataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None


def create_train_transforms():
    """Create stochastic training augmentations when albumentations is installed."""
    if A is None or ToTensorV2 is None:
        return None

    return A.Compose(
        [
            A.D4(p=1),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                p=0.5,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406, 0.0],
                std=[0.229, 0.224, 0.225, 1.0],
                max_pixel_value=1.0
            ),
            ToTensorV2(),
        ]
    )


def create_val_transforms():
    """Create deterministic evaluation preprocessing when albumentations is installed."""
    if A is None or ToTensorV2 is None:
        return None

    return A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406, 0.0],
                std=[0.229, 0.224, 0.225, 1.0],
                max_pixel_value=1.0
            ),
            ToTensorV2(),
        ]
    )


class CloudDataModule(pl.LightningDataModule):
    """Own dataset construction and DataLoader settings for train/val/test splits."""

    def __init__(
        self,
        config: ModelConfig,
        x_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
        x_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.config = config
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.x_train is not None and self.y_train is not None:
            self.train_dataset = CloudDataset(
                x_paths=self.x_train,
                bands=self.config.bands,
                y_paths=self.y_train,
                transforms=create_train_transforms(),
            )

        if self.x_val is not None and self.y_val is not None:
            self.val_dataset = CloudDataset(
                x_paths=self.x_val,
                bands=self.config.bands,
                y_paths=self.y_val,
                transforms=create_val_transforms(),
            )

        if self.x_test is not None:
            self.test_dataset = CloudDataset(
                x_paths=self.x_test,
                bands=self.config.bands,
                y_paths=self.y_test,
                transforms=None,
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            return None

        loader_kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "shuffle": True,
            "pin_memory": torch.cuda.is_available(),
        }
        if self.config.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        return torch.utils.data.DataLoader(self.train_dataset, **loader_kwargs)

    def val_dataloader(self):
        if self.val_dataset is None:
            return None

        return self._eval_dataloader(self.val_dataset)

    def test_dataloader(self):
        if self.test_dataset is None:
            return None

        return self._eval_dataloader(self.test_dataset)

    def _eval_dataloader(self, dataset: CloudDataset):
        loader_kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "shuffle": False,
            "pin_memory": torch.cuda.is_available(),
        }
        if self.config.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        return torch.utils.data.DataLoader(dataset, **loader_kwargs)
