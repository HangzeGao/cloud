"""Evaluation metrics and mask alignment helpers."""

from .metrics import (
    calculate_class_distribution,
    evaluate_batch,
    intersection_over_union,
    intersection_over_union_and_coverage,
    read_prediction_and_aligned_true,
)

__all__ = [
    "calculate_class_distribution",
    "evaluate_batch",
    "intersection_over_union",
    "intersection_over_union_and_coverage",
    "read_prediction_and_aligned_true",
]
