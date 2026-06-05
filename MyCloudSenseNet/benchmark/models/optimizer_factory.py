"""Optimizer creation and parameter grouping helpers."""

import torch
import torch.nn as nn
from torch.optim import Optimizer


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    encoder_lr_scale: float = 0.5,
    bit_depth_component_lr_scale: float = 2.0,
    **kwargs,
) -> Optimizer:
    """Create an optimizer with separate learning rates for model submodules."""
    param_groups = []

    base_encoder = _unwrap_encoder(getattr(model, "encoder", None))
    if base_encoder is not None:
        param_groups.append(
            {
                "params": list(base_encoder.parameters()),
                "lr": learning_rate * encoder_lr_scale,
                "name": "encoder",
                "weight_decay": weight_decay,
            }
        )

    if hasattr(model, "decoder"):
        param_groups.append(
            {
                "params": list(model.decoder.parameters()),
                "lr": learning_rate,
                "name": "decoder",
                "weight_decay": weight_decay,
            }
        )

    if hasattr(model, "segmentation_head"):
        param_groups.append(
            {
                "params": list(model.segmentation_head.parameters()),
                "lr": learning_rate,
                "name": "head",
                "weight_decay": weight_decay,
            }
        )

    _append_bit_depth_group(
        param_groups,
        model,
        attr_name="bit_depth_estimator",
        group_name="bit_depth_estimator",
        learning_rate=learning_rate * bit_depth_component_lr_scale,
        weight_decay=weight_decay * 0.1,
    )
    _append_bit_depth_group(
        param_groups,
        model,
        attr_name="feature_adapter",
        group_name="feature_adapter",
        learning_rate=learning_rate * bit_depth_component_lr_scale,
        weight_decay=weight_decay * 0.1,
    )

    other_params = _collect_ungrouped_parameters(model, param_groups)
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": learning_rate,
                "name": "other",
                "weight_decay": weight_decay,
            }
        )

    optimizer_type = optimizer_type.lower()
    if optimizer_type == "adamw":
        return torch.optim.AdamW(param_groups)
    if optimizer_type == "adam":
        return torch.optim.Adam(param_groups)
    if optimizer_type == "sgd":
        return torch.optim.SGD(
            param_groups,
            momentum=kwargs.get("momentum", 0.9),
            nesterov=kwargs.get("nesterov", True),
        )

    raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def _unwrap_encoder(encoder):
    if encoder is None:
        return None

    base_encoder = encoder
    while hasattr(base_encoder, "encoder") and base_encoder is not base_encoder.encoder:
        base_encoder = base_encoder.encoder
    return base_encoder


def _find_encoder_component(model: nn.Module, attr_name: str):
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return None

    if hasattr(encoder, attr_name):
        return getattr(encoder, attr_name)
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, attr_name):
        return getattr(encoder.encoder, attr_name)
    return None


def _append_bit_depth_group(
    param_groups: list,
    model: nn.Module,
    attr_name: str,
    group_name: str,
    learning_rate: float,
    weight_decay: float,
):
    component = _find_encoder_component(model, attr_name)
    if component is None:
        return

    grouped_param_ids = {
        id(param)
        for group in param_groups
        for param in group["params"]
    }
    params = [
        param
        for param in component.parameters()
        if id(param) not in grouped_param_ids
    ]
    if params:
        param_groups.append(
            {
                "params": params,
                "lr": learning_rate,
                "name": group_name,
                "weight_decay": weight_decay,
            }
        )


def _collect_ungrouped_parameters(model: nn.Module, param_groups: list):
    grouped_param_ids = {
        id(param)
        for group in param_groups
        for param in group["params"]
    }
    return [
        param
        for _, param in model.named_parameters()
        if param.requires_grad and id(param) not in grouped_param_ids
    ]
