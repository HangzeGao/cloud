"""Factory for configured segmentation losses."""

import torch.nn as nn

from .boundary_losses import BoundaryAwareLoss
from .compound_losses import DynamicWeightedCompoundLoss
from .loss_base import DiceLoss, FocalLoss


class CombinedLoss(nn.Module):
    """Create loss modules from configuration-friendly names."""

    @staticmethod
    def create(
        loss_type: str = "dwcl",
        num_classes: int = 3,
        **kwargs,
    ) -> nn.Module:
        if loss_type == "ce":
            return nn.CrossEntropyLoss(
                weight=kwargs.get("class_weights"),
                ignore_index=kwargs.get("ignore_index", 255),
            )

        if loss_type == "dice":
            return DiceLoss(
                mode="multiclass",
                from_logits=True,
                ignore_index=kwargs.get("ignore_index", 255),
            )

        if loss_type == "focal":
            return FocalLoss(
                mode="multiclass",
                alpha=kwargs.get("alpha", 0.25),
                gamma=kwargs.get("gamma", 2.0),
                ignore_index=kwargs.get("ignore_index", 255),
            )

        if loss_type == "dwcl":
            return DynamicWeightedCompoundLoss(
                num_classes=num_classes,
                use_ce=kwargs.get("use_ce", True),
                use_dice=kwargs.get("use_dice", True),
                use_focal=kwargs.get("use_focal", True),
                enable_dynamic_weighting=kwargs.get("enable_dynamic_weighting", True),
                enable_class_adaptive=kwargs.get("enable_class_adaptive", True),
                focal_alpha=kwargs.get("focal_alpha", 0.25),
                focal_gamma=kwargs.get("focal_gamma", 2.0),
                class_weights=kwargs.get("class_weights"),
            )

        if loss_type == "boundary":
            base = (
                DynamicWeightedCompoundLoss(num_classes=num_classes)
                if kwargs.get("use_advanced", True)
                else None
            )
            return BoundaryAwareLoss(
                base_loss=base,
                boundary_width=kwargs.get("boundary_width", 5),
                boundary_weight=kwargs.get("boundary_weight", 2.0),
            )

        if loss_type == "combined":
            dwcl = DynamicWeightedCompoundLoss(
                num_classes=num_classes,
                use_ce=kwargs.get("use_ce", True),
                use_dice=kwargs.get("use_dice", True),
                use_focal=kwargs.get("use_focal", True),
                enable_dynamic_weighting=kwargs.get("enable_dynamic_weighting", True),
                enable_class_adaptive=kwargs.get("enable_class_adaptive", True),
            )

            if kwargs.get("enable_boundary", True):
                return BoundaryAwareLoss(
                    base_loss=dwcl,
                    boundary_width=kwargs.get("boundary_width", 5),
                    boundary_weight=kwargs.get("boundary_weight", 2.0),
                )

            return dwcl

        raise ValueError(f"Unknown loss type: {loss_type}")
