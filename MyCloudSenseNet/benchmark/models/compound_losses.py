"""Compound segmentation losses."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .loss_base import DiceLoss, FocalLoss
from .loss_weights import ClassAdaptiveWeights


class DynamicWeightedCompoundLoss(nn.Module):
    """
    Cross-entropy, Dice and focal loss with optional uncertainty weighting.

    Reference:
        Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """

    def __init__(
        self,
        num_classes: int = 3,
        mode: str = "multiclass",
        use_ce: bool = True,
        use_dice: bool = True,
        use_focal: bool = True,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.5,
        enable_dynamic_weighting: bool = True,
        enable_class_adaptive: bool = True,
        class_weight_power: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1e-5,
        dice_log_loss: bool = False,
        ignore_index: int = 255,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.mode = mode
        self.enable_dynamic_weighting = enable_dynamic_weighting
        self.enable_class_adaptive = enable_class_adaptive
        self.class_weight_power = class_weight_power
        self.use_ce = use_ce
        self.use_dice = use_dice
        self.use_focal = use_focal
        self.ignore_index = ignore_index

        if enable_dynamic_weighting:
            self.log_vars = nn.Parameter(torch.zeros(3))
        else:
            self.register_buffer("log_vars", torch.zeros(3))
            self.log_vars[0] = -torch.log(torch.tensor(ce_weight))
            self.log_vars[1] = -torch.log(torch.tensor(dice_weight))
            self.log_vars[2] = -torch.log(torch.tensor(focal_weight))

        if use_ce:
            base_weight = torch.tensor(class_weights) if class_weights else None
            self.ce_loss = nn.CrossEntropyLoss(
                weight=base_weight,
                ignore_index=ignore_index,
                reduction="none",
            )

        if use_dice:
            self.dice_loss = DiceLoss(
                mode=mode,
                from_logits=True,
                smooth=dice_smooth,
                log_loss=dice_log_loss,
                ignore_index=ignore_index,
            )

        if use_focal:
            self.focal_loss = FocalLoss(
                mode=mode,
                alpha=focal_alpha,
                gamma=focal_gamma,
                ignore_index=ignore_index,
                reduction="none",
            )

        if enable_class_adaptive:
            self.class_adaptive = ClassAdaptiveWeights(num_classes=num_classes)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        epoch: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        total_loss = 0.0

        if self.enable_class_adaptive and self.training:
            class_weights = self.class_adaptive(targets)
        else:
            class_weights = None

        losses = []

        if self.use_ce:
            ce = self.ce_loss(logits, targets)

            if class_weights is not None:
                target_weights = class_weights[targets.clamp(0, self.num_classes - 1)]
                valid_mask = targets != self.ignore_index
                ce = ce[valid_mask]
                target_weights = target_weights[valid_mask]
                ce = (ce * target_weights).mean()
            else:
                ce = ce.mean()

            losses.append(ce)
            loss_dict["ce_loss"] = ce.item()

        if self.use_dice:
            dice = self.dice_loss(logits, targets)
            losses.append(dice)
            loss_dict["dice_loss"] = dice.item()

        if self.use_focal:
            focal = self.focal_loss(logits, targets)
            losses.append(focal)
            loss_dict["focal_loss"] = focal.item()

        if self.enable_dynamic_weighting:
            for i, loss in enumerate(losses):
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * loss + self.log_vars[i]
                total_loss = total_loss + weighted_loss
                loss_dict[f"weighted_loss_{i}"] = weighted_loss.item()
                loss_dict[f"precision_{i}"] = precision.item()
        else:
            weights = torch.exp(-self.log_vars)
            for loss, weight in zip(losses, weights):
                total_loss = total_loss + weight * loss

        for i in range(len(losses)):
            loss_dict[f"log_var_{i}"] = self.log_vars[i].item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
