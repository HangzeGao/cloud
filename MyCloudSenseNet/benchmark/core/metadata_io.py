"""
Metadata I/O module for cloud detection.

This module handles loading and saving of chip metadata, including
reading from directories and CSV files.
"""

from pathlib import Path
from typing import List, Set

import pandas as pd

from benchmark.core.config import BANDS, logger


def required_chip_columns(bands: List[str] = BANDS) -> Set[str]:
    """
    Get the set of required column names for chip metadata.

    Args:
        bands: List of band names

    Returns:
        Set of required column names
    """
    return {"chip_id", *(f"{band}_path" for band in bands)}


def get_chip_metadata(
    features_dir: Path,
    bands: List[str] = BANDS,
) -> pd.DataFrame:
    """
    Scan a directory and build chip metadata from band files.

    Looks for band files (e.g., B02.tif, B03.tif, etc.) either directly in
    the features_dir or in subdirectories.

    Args:
        features_dir: Directory containing chip images
        bands: List of band names to look for

    Returns:
        DataFrame with columns: chip_id, {band}_path for each band

    Raises:
        ValueError: If no valid chips are found or directory doesn't exist
    """
    features_dir = Path(features_dir)
    if not features_dir.exists():
        raise ValueError(f"features_dir does not exist: {features_dir}")

    rows = []

    # Check if band files are directly in the directory
    if all((features_dir / f"{band}.tif").exists() for band in bands):
        rows.append({
            "chip_id": features_dir.name,
            **{
                f"{band}_path": str((features_dir / f"{band}.tif").resolve())
                for band in bands
            },
        })
        return pd.DataFrame(rows)

    # Look for band files in subdirectories
    chip_dirs = [
        path for path in features_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]

    for chip_dir in sorted(chip_dirs):
        if not all((chip_dir / f"{band}.tif").exists() for band in bands):
            logger.warning(f"Skipping {chip_dir.name}: missing one or more band files")
            continue

        row = {"chip_id": chip_dir.name}
        for band in bands:
            row[f"{band}_path"] = str((chip_dir / f"{band}.tif").resolve())
        rows.append(row)

    if not rows:
        band_files = ", ".join(f"{band}.tif" for band in bands)
        raise ValueError(
            f"No valid chip folders found in {features_dir}. "
            f"Expected either {band_files} directly, "
            "or subfolders containing those files."
        )

    return pd.DataFrame(rows)


def load_chip_metadata(
    features: Path | pd.DataFrame,
    bands: List[str] = BANDS,
) -> pd.DataFrame:
    """
    Load chip metadata from a directory, CSV file, or DataFrame.

    Args:
        features: Path to features directory or CSV, or existing DataFrame
        bands: List of band names

    Returns:
        DataFrame with chip metadata

    Raises:
        ValueError: If required columns are missing from CSV
    """
    if isinstance(features, pd.DataFrame):
        return features.copy()

    features = Path(features)

    if features.suffix.lower() == ".csv":
        metadata = pd.read_csv(features)
    else:
        metadata = get_chip_metadata(features, bands=bands)

    # Validate required columns
    required = required_chip_columns(bands)
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Chip metadata is missing required columns: {sorted(missing)}")

    return metadata


def save_chip_metadata(
    metadata: pd.DataFrame,
    output_path: Path,
    index: bool = False,
) -> None:
    """
    Save chip metadata to a CSV file.

    Args:
        metadata: DataFrame with chip metadata
        output_path: Output CSV path
        index: Whether to include DataFrame index in output
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=index)
    logger.info(f"Saved chip metadata to {output_path}")


def merge_chip_metadata(
    metadata_list: List[pd.DataFrame],
    validate_columns: bool = True,
    bands: List[str] = BANDS,
) -> pd.DataFrame:
    """
    Merge multiple chip metadata DataFrames.

    Args:
        metadata_list: List of DataFrames to merge
        validate_columns: Whether to validate column consistency
        bands: List of band names for validation

    Returns:
        Merged DataFrame

    Raises:
        ValueError: If validation fails or metadata_list is empty
    """
    if not metadata_list:
        raise ValueError("Cannot merge empty list of metadata DataFrames")

    if len(metadata_list) == 1:
        return metadata_list[0].copy()

    # Check for consistent columns
    if validate_columns:
        required = required_chip_columns(bands)
        for i, metadata in enumerate(metadata_list):
            missing = required - set(metadata.columns)
            if missing:
                raise ValueError(
                    f"Metadata DataFrame at index {i} missing required columns: {missing}"
                )

    return pd.concat(metadata_list, ignore_index=True)


def filter_valid_chips(
    metadata: pd.DataFrame,
    label_column: str = "label_path",
) -> pd.DataFrame:
    """
    Filter metadata to only include chips with valid labels.

    Args:
        metadata: DataFrame with chip metadata
        label_column: Column name containing label paths

    Returns:
        Filtered DataFrame with only existing labels
    """
    if label_column not in metadata.columns:
        logger.warning(f"Label column '{label_column}' not found, returning all chips")
        return metadata.copy()

    valid_mask = metadata[label_column].apply(lambda x: Path(x).exists())
    valid_count = valid_mask.sum()
    total_count = len(metadata)

    if valid_count < total_count:
        logger.info(f"Filtered {total_count - valid_count} chips with missing labels")

    return metadata[valid_mask].copy()


def get_chip_statistics(metadata: pd.DataFrame) -> dict:
    """
    Get statistics about chip metadata.

    Args:
        metadata: DataFrame with chip metadata

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_chips": len(metadata),
        "columns": list(metadata.columns),
        "chip_id_unique": metadata["chip_id"].nunique() if "chip_id" in metadata.columns else 0,
    }

    # Check for path columns
    path_columns = [col for col in metadata.columns if col.endswith("_path")]
    stats["path_columns"] = path_columns

    for col in path_columns:
        existing = metadata[col].apply(lambda x: Path(x).exists()).sum()
        stats[f"{col}_existing"] = int(existing)
        stats[f"{col}_missing"] = len(metadata) - int(existing)

    return stats
