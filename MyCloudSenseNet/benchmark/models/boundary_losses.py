"""Boundary-aware segmentation losses."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAwareLoss(nn.Module):
    """Apply a larger loss weight around label boundaries."""

    def __init__(
        self,
        base_loss: Optional[nn.Module] = None,
        boundary_width: int = 5,
        boundary_weight: float = 2.0,
        kernel_size: int = 3,
        mode: str = "multiclass",
        ignore_index: int = 255,
    ):
        super().__init__()

        self.base_loss = base_loss
        self.boundary_width = boundary_width
        self.boundary_weight = boundary_weight
        self.mode = mode
        self.ignore_index = ignore_index

        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3),
        )

        dilation_size = boundary_width * 2 + 1
        self.register_buffer(
            "dilation_kernel",
            torch.ones(1, 1, dilation_size, dilation_size),
        )

    def detect_boundaries(self, targets: torch.Tensor) -> torch.Tensor:
        batch_size, height, width = targets.shape
        device = targets.device
        boundary_masks = []
        unique_classes = torch.unique(targets[targets != self.ignore_index])

        for cls in unique_classes:
            binary_mask = (targets == cls).float().unsqueeze(1)
            edge_x = F.conv2d(binary_mask, self.sobel_x, padding=1)
            edge_y = F.conv2d(binary_mask, self.sobel_y, padding=1)
            edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
            boundary_masks.append((edges > 0.1).float())

        if boundary_masks:
            all_boundaries = torch.stack(boundary_masks, dim=0).max(dim=0)[0]
        else:
            all_boundaries = torch.zeros(batch_size, 1, height, width, device=device)

        boundary_mask = F.max_pool2d(
            all_boundaries,
            kernel_size=self.dilation_kernel.shape[-1],
            stride=1,
            padding=self.boundary_width,
        )
        return boundary_mask.squeeze(1)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            boundary_mask = self.detect_boundaries(targets)

        # Keep this per-pixel so boundary weights can be applied consistently.
        base_loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        weights = torch.ones_like(base_loss)
        weights[boundary_mask > 0] = self.boundary_weight
        weighted_loss = (base_loss * weights).mean()

        info = {
            "boundary_ratio": boundary_mask.mean().item(),
            "boundary_weight": self.boundary_weight,
        }
        return weighted_loss, info
