"""Small filesystem and shape utilities."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure

    Returns:
        The ensured directory path
    """
    path.mkdir(exist_ok=True, parents=True)
    return path


def round_up(value: int, divisor: int) -> int:
    """
    Round up a value to the nearest multiple of divisor.

    Args:
        value: Value to round up
        divisor: Divisor to round to

    Returns:
        Rounded up value
    """
    if divisor <= 1:
        return value
    return ((value + divisor - 1) // divisor) * divisor

