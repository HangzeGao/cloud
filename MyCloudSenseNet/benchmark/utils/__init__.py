"""
Utilities module for cloud detection benchmark.

Contains general-purpose utility functions:
- General utilities (utils)
- Test-time augmentation (tta)
- Visualization tools (visualization)
"""

from benchmark.utils.utils import (
    mask2label,
    stretch,
    ensure_dir,
    round_up,
    calculate_bit_depth,
    is_label_valid,
    create_rgb_composite,
    display_thumbnail,
)
from benchmark.utils.tta import (
    normalize_tta_modes,
    apply_tta,
    undo_tta,
    predict_with_tta,
    get_tta_mode_count,
)
# Note: visualization is imported separately to avoid circular imports
# from benchmark.utils.visualization import ...

__all__ = [
    # Utils
    "mask2label",
    "stretch",
    "ensure_dir",
    "round_up",
    "calculate_bit_depth",
    "is_label_valid",
    "create_rgb_composite",
    "display_thumbnail",
    # TTA
    "normalize_tta_modes",
    "apply_tta",
    "undo_tta",
    "predict_with_tta",
    "get_tta_mode_count",
    # Visualization (import separately from benchmark.utils.visualization)
]
