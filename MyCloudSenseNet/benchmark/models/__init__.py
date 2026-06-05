"""Benchmark model architectures and model-building components."""

from ..configs import Configs, ModelConfig

__all__ = ["ModelConfig", "Configs", "CloudModel", "BitDepthAdaptiveEncoder", "get_device"]


def __getattr__(name: str):
    if name in {"CloudModel", "BitDepthAdaptiveEncoder", "get_device"}:
        from .cloud_model import BitDepthAdaptiveEncoder, CloudModel, get_device

        exports = {
            "CloudModel": CloudModel,
            "BitDepthAdaptiveEncoder": BitDepthAdaptiveEncoder,
            "get_device": get_device,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
