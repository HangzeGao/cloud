"""
Model components module for cloud detection benchmark.

Contains model-specific components:
- Adaptive encoders (adaptive_encoders)
- Bit depth estimators (bit_depth_estimators)
- Feature adapters (feature_adapters)
- Loss functions (losses)
- Model architectures (unet/, segformer/)
"""

from benchmark.models.losses import intersection_over_union
from benchmark.models.adaptive_encoders import AdaptiveEncoderFactory
from benchmark.models.bit_depth_estimators import (
    get_bit_depth_range,
    get_num_bit_depths,
    get_bit_depth_values,
    get_bit_depth_tensor,
    pretty_print_dict,
)

__all__ = [
    "intersection_over_union",
    "AdaptiveEncoderFactory",
    "get_bit_depth_range",
    "get_num_bit_depths",
    "get_bit_depth_values",
    "get_bit_depth_tensor",
    "pretty_print_dict",
]
