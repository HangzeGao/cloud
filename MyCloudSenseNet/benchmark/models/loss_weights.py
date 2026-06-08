"""Class weighting utilities for segmentation losses."""

import torch
import torch.nn as nn


class ClassAdaptiveWeights(nn.Module):
    """Maintain an EMA estimate of class frequencies and return inverse weights."""

    def __init__(
        self,
        num_classes: int,
        min_weight: float = 0.1,
        max_weight: float = 5.0,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.momentum = momentum

        self.register_buffer("class_freq", torch.ones(num_classes) / num_classes)
        self.register_buffer("num_batches", torch.tensor(0))

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        batch_freq = torch.bincount(
            targets.flatten(),
            minlength=self.num_classes,
        ).float()
        batch_freq = batch_freq / (batch_freq.sum() + 1e-8)

        if self.training:
            self.num_batches += 1
            momentum = min(self.momentum, 1 - 1 / (self.num_batches + 1))
            self.class_freq = momentum * self.class_freq + (1 - momentum) * batch_freq

        weights = 1.0 / (self.class_freq + 1e-8)
        weights = weights / weights.sum() * self.num_classes
        return torch.clamp(weights, self.min_weight, self.max_weight)

    def reset(self):
        self.class_freq.fill_(1.0 / self.num_classes)
        self.num_batches.zero_()


def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int = 3,
    mode: str = "inverse_freq",
    min_weight: float = 0.1,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Compute static class weights from a label tensor."""
    counts = torch.bincount(targets.flatten(), minlength=num_classes).float()

    if mode == "inverse_freq":
        weights = 1.0 / (counts + 1e-8)
    elif mode == "effective_num":
        beta = 0.9999
        weights = (1.0 - beta) / (1.0 - beta ** (counts + 1e-8))
    elif mode == "sqrt_inv":
        weights = 1.0 / torch.sqrt(counts + 1e-8)
    else:
        weights = torch.ones(num_classes)

    weights = weights / weights.sum() * num_classes
    return torch.clamp(weights, min_weight, max_weight)
