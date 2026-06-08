"""Basic segmentation loss implementations."""

from typing import List, Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for binary or multiclass segmentation."""

    def __init__(
        self,
        mode: str = "multiclass",
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != self.ignore_index

        if self.mode == "multiclass":
            probs = F.softmax(logits, dim=1)
            _, num_classes, _, _ = logits.shape
            targets_one_hot = F.one_hot(
                targets.clamp(0, num_classes - 1),
                num_classes=num_classes,
            ).permute(0, 3, 1, 2).float()

            p_t = (probs * targets_one_hot).sum(dim=1)
            p_t = p_t[valid_mask]
            focal_weight = (1 - p_t) ** self.gamma

            ce_loss = F.cross_entropy(
                logits,
                targets,
                ignore_index=self.ignore_index,
                reduction="none",
            )
            focal_loss = self.alpha * focal_weight * ce_loss[valid_mask]
        elif self.mode == "binary":
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1 - p_t) ** self.gamma
            ce_loss = F.binary_cross_entropy_with_logits(
                logits,
                targets.float(),
                reduction="none",
            )
            focal_loss = self.alpha * focal_weight * ce_loss
        else:
            raise ValueError(f"Unknown focal loss mode: {self.mode}")

        if self.reduction == "mean":
            if focal_loss.numel() == 0:
                return torch.tensor(0.0, device=logits.device)
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DiceLoss(smp.losses.DiceLoss):
    """Thin wrapper around SMP DiceLoss with ignore-index short circuiting."""

    def __init__(
        self,
        mode: str = "multiclass",
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 1e-5,
        ignore_index: Optional[int] = 255,
        eps: float = 1e-7,
    ):
        super().__init__(
            mode=mode,
            classes=classes,
            log_loss=log_loss,
            from_logits=from_logits,
            smooth=smooth,
            eps=eps,
        )
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device)

        return super().forward(logits, targets)
