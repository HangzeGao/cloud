"""
Metadata I/O module for cloud detection.

This module handles loading and saving of chip metadata, including
reading from directories and CSV files.
"""

from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Set

import pandas as pd

from ..configs.defaults import BANDS, NUM_CLASSES, logger
from .raster_utils import label_class_stats

DEFAULT_RASTER_EXTENSIONS = (".tif", ".tiff")
DEFAULT_LABEL_NAMES = ("label", "labels", "mask", "cloud", "cloud_mask")


def required_chip_columns(bands: List[str] = BANDS) -> Set[str]:
    """
    Get the set of required column names for chip metadata.

    Args:
        bands: List of band names

    Returns:
        Set of required column names
    """
    return {"chip_id", *(f"{band}_path" for band in bands)}


def _is_raster_file(path: Path, extensions: Sequence[str]) -> bool:
    return path.is_file() and path.suffix.lower() in {ext.lower() for ext in extensions}


def _path_to_csv_value(path: Path, relative_to: Path | None) -> str:
    if relative_to is None:
        return str(path.resolve())
    try:
        return path.resolve().relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _strip_band_token(stem: str, bands: Sequence[str]) -> tuple[str, str] | None:
    """Return (chip_stem, band) for names like chip_B02 or B02_chip."""
    for band in bands:
        for sep in ("_", "-", "."):
            suffix = f"{sep}{band}"
            prefix = f"{band}{sep}"
            if stem.endswith(suffix):
                chip_stem = stem[: -len(suffix)]
                return chip_stem, band
            if stem.startswith(prefix):
                chip_stem = stem[len(prefix):]
                return chip_stem, band
    return None


def _find_named_raster(directory: Path, stem: str, extensions: Sequence[str]) -> Path | None:
    for ext in extensions:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _find_matching_label(
    label_root: Path,
    chip_id: str,
    image_rel_path: Path | None,
    extensions: Sequence[str],
    required: bool,
) -> Path | None:
    candidates: list[Path] = []
    chip_path = Path(chip_id)

    for ext in extensions:
        candidates.append(label_root / chip_path.with_suffix(ext))
        candidates.append(label_root / chip_path / f"{chip_path.name}{ext}")
        for label_name in DEFAULT_LABEL_NAMES:
            candidates.append(label_root / chip_path / f"{label_name}{ext}")

        if image_rel_path is not None:
            candidates.append(label_root / image_rel_path.with_suffix(ext))
            candidates.append(label_root / image_rel_path.parent / f"{image_rel_path.stem}{ext}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if required:
        raise FileNotFoundError(
            f"No label found for chip_id='{chip_id}' under {label_root}"
        )
    return None


def build_dataset_metadata(
    dataset_name: str | None = None,
    data_root: Path | str | None = None,
    images_dir: Path | str | None = None,
    labels_dir: Path | str | None = None,
    output_csv: Path | str | None = None,
    bands: List[str] = BANDS,
    image_extensions: Sequence[str] = DEFAULT_RASTER_EXTENSIONS,
    label_extensions: Sequence[str] = DEFAULT_RASTER_EXTENSIONS,
    relative_to: Path | str | None = None,
    include_labels: bool = True,
    strict: bool = True,
    allow_multiband: bool = True,
    add_label_distribution: bool = False,
    num_classes: int = NUM_CLASSES,
) -> pd.DataFrame:
    """
    Build RICE2-style chip metadata from mirrored image/label directories.

    The returned DataFrame uses the project CSV format:
    ``chip_id,B02_path,B03_path,B04_path,B08_path,label_path``.

    Supported image layouts include:
    - ``images/<dataset>/<chip_id>/<band>.tif`` (RICE2-style chip folders)
    - ``images/<dataset>/<chip_id>_<band>.tif`` (band token in filename)
    - ``images/<dataset>/<chip_id>.tif`` (one multiband file; all band columns
      point to the same file so callers can post-process or use a multiband
      reader)

    Labels are matched by the same relative chip position under
    ``labels/<dataset>``. For example, ``images/oncloudn/a/001.tif`` will look
    for ``labels/oncloudn/a/001.tif``.

    Args:
        dataset_name: Dataset folder name such as ``oncloudn``, ``RICE2``, or
            ``HRC_WHU``. Required when using ``data_root`` defaults.
        data_root: Root containing ``images/``, ``labels/``, and optionally
            ``metadata/``.
        images_dir: Explicit image directory. Overrides ``data_root/images``.
        labels_dir: Explicit label directory. Overrides ``data_root/labels``.
        output_csv: Optional CSV path to write.
        bands: Band names to create ``*_path`` columns for.
        image_extensions: Raster extensions to scan for images.
        label_extensions: Raster extensions to scan for labels.
        relative_to: If provided, paths written to the DataFrame are relative to
            this directory. Otherwise absolute paths are used.
        include_labels: Whether to add and validate ``label_path``.
        strict: Raise on missing labels or incomplete band groups. If false,
            skip incomplete band groups and leave missing labels absent.
        allow_multiband: Treat leftover image files as single multiband images.
        add_label_distribution: Add ``dominant_label_class``,
            ``label_non_background_ratio``, and ``label_class_*_ratio`` columns
            by reading each ``label_path``.
        num_classes: Number of label classes used for distribution columns.

    Returns:
        Metadata DataFrame sorted by ``chip_id``.
    """
    if images_dir is None:
        if data_root is None or dataset_name is None:
            raise ValueError("Provide images_dir, or both data_root and dataset_name.")
        images_dir = Path(data_root) / "images" / dataset_name
    else:
        images_dir = Path(images_dir)

    if labels_dir is None and include_labels:
        if data_root is None or dataset_name is None:
            raise ValueError("Provide labels_dir, or both data_root and dataset_name.")
        labels_dir = Path(data_root) / "labels" / dataset_name
    elif labels_dir is not None:
        labels_dir = Path(labels_dir)

    image_root = Path(images_dir)
    label_root = Path(labels_dir) if labels_dir is not None else None
    relative_base = Path(relative_to) if relative_to is not None else None

    if not image_root.exists():
        raise ValueError(f"images_dir does not exist: {image_root}")
    if include_labels and label_root is not None and not label_root.exists():
        raise ValueError(f"labels_dir does not exist: {label_root}")

    image_files = sorted(
        path for path in image_root.rglob("*")
        if _is_raster_file(path, image_extensions)
    )
    if not image_files:
        raise ValueError(f"No image files found in {image_root}")

    rows: list[dict[str, str]] = []
    used_paths: set[Path] = set()

    chip_dirs = sorted({path.parent for path in image_files})
    for chip_dir in chip_dirs:
        by_band = {}
        for band in bands:
            band_path = _find_named_raster(chip_dir, band, image_extensions)
            if band_path is not None:
                by_band[band] = band_path
        if not by_band:
            continue
        if set(by_band) != set(bands):
            message = f"Skipping {chip_dir}: missing bands {sorted(set(bands) - set(by_band))}"
            if strict:
                raise ValueError(message)
            logger.warning(message)
            continue

        chip_id = chip_dir.relative_to(image_root).as_posix()
        if chip_id == ".":
            chip_id = image_root.name

        row = {"chip_id": chip_id}
        for band in bands:
            band_path = by_band[band]
            row[f"{band}_path"] = _path_to_csv_value(band_path, relative_base)
            used_paths.add(band_path)
        if include_labels and label_root is not None:
            label_path = _find_matching_label(
                label_root, chip_id, None, label_extensions, required=strict
            )
            if label_path is not None:
                row["label_path"] = _path_to_csv_value(label_path, relative_base)
            elif not strict:
                row["label_path"] = ""
        rows.append(row)

    grouped_band_files: dict[tuple[Path, str], dict[str, Path]] = {}
    for image_path in image_files:
        if image_path in used_paths:
            continue
        band_match = _strip_band_token(image_path.stem, bands)
        if band_match is None:
            continue
        chip_stem, band = band_match
        grouped_band_files.setdefault((image_path.parent, chip_stem), {})[band] = image_path

    for (parent_dir, chip_stem), by_band in sorted(grouped_band_files.items()):
        if set(by_band) != set(bands):
            message = (
                f"Skipping {parent_dir / chip_stem}: "
                f"missing bands {sorted(set(bands) - set(by_band))}"
            )
            if strict:
                raise ValueError(message)
            logger.warning(message)
            continue

        chip_id = (parent_dir.relative_to(image_root) / chip_stem).as_posix()
        row = {"chip_id": chip_id}
        for band in bands:
            band_path = by_band[band]
            row[f"{band}_path"] = _path_to_csv_value(band_path, relative_base)
            used_paths.add(band_path)
        if include_labels and label_root is not None:
            label_path = _find_matching_label(
                label_root,
                chip_id,
                Path(chip_id).with_suffix(by_band[bands[0]].suffix),
                label_extensions,
                required=strict,
            )
            if label_path is not None:
                row["label_path"] = _path_to_csv_value(label_path, relative_base)
            elif not strict:
                row["label_path"] = ""
        rows.append(row)

    if allow_multiband:
        for image_path in image_files:
            if image_path in used_paths:
                continue
            image_rel = image_path.relative_to(image_root)
            chip_id = image_rel.with_suffix("").as_posix()
            row = {"chip_id": chip_id}
            for band in bands:
                row[f"{band}_path"] = _path_to_csv_value(image_path, relative_base)
            if include_labels and label_root is not None:
                label_path = _find_matching_label(
                    label_root,
                    chip_id,
                    image_rel,
                    label_extensions,
                    required=strict,
                )
                if label_path is not None:
                    row["label_path"] = _path_to_csv_value(label_path, relative_base)
                elif not strict:
                    row["label_path"] = ""
            rows.append(row)

    if not rows:
        raise ValueError(f"No valid chips found in {image_root}")

    metadata = pd.DataFrame(rows).drop_duplicates("chip_id").sort_values("chip_id")
    metadata = metadata.reset_index(drop=True)

    required = required_chip_columns(bands)
    if include_labels:
        required.add("label_path")
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Generated metadata is missing columns: {sorted(missing)}")

    ordered_columns = ["chip_id", *(f"{band}_path" for band in bands)]
    if "label_path" in metadata.columns:
        ordered_columns.append("label_path")
    metadata = metadata[ordered_columns + [c for c in metadata.columns if c not in ordered_columns]]

    if add_label_distribution:
        if not include_labels or "label_path" not in metadata.columns:
            raise ValueError("add_label_distribution=True requires label_path metadata.")
        if not strict and metadata["label_path"].eq("").any():
            raise ValueError(
                "Cannot add label distribution when some labels are missing. "
                "Use strict=True or add_label_distribution=False."
            )

        distribution_metadata = metadata.copy()
        if relative_base is not None:
            distribution_metadata["label_path"] = distribution_metadata["label_path"].apply(
                lambda value: str((relative_base / value).resolve())
            )
        distribution_metadata = enrich_label_distribution_from_paths(
            distribution_metadata,
            num_classes=num_classes,
        )
        distribution_columns = label_distribution_columns(num_classes=num_classes)
        metadata = pd.concat(
            [
                metadata,
                distribution_metadata[distribution_columns],
            ],
            axis=1,
        )

    if output_csv is not None:
        save_chip_metadata(metadata, Path(output_csv))

    return metadata


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


def label_distribution_columns(num_classes: int = NUM_CLASSES) -> list[str]:
    """Return the standard label-distribution columns stored in metadata."""
    columns = ["dominant_label_class", "label_non_background_ratio"]
    for class_id in range(num_classes):
        columns.append(f"label_class_{class_id}_ratio")
    return columns


def enrich_label_distribution_from_paths(
    metadata: pd.DataFrame,
    label_column: str = "label_path",
    num_classes: int = NUM_CLASSES,
) -> pd.DataFrame:
    """Add per-class label ratios by reading each label GeoTIFF."""
    if label_column not in metadata.columns:
        raise ValueError(f"Label column '{label_column}' not found")

    import rasterio

    rows = []
    for _, row in metadata.iterrows():
        label_path = Path(row[label_column])
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        with rasterio.open(label_path) as src:
            label = src.read(1)
        rows.append(label_class_stats(label, num_classes=num_classes))

    distribution = pd.DataFrame(rows, index=metadata.index)
    return pd.concat(
        [
            metadata.drop(
                columns=[c for c in distribution.columns if c in metadata.columns],
                errors="ignore",
            ),
            distribution,
        ],
        axis=1,
    )


def summarize_label_distribution(
    metadata: pd.DataFrame,
    num_classes: int = NUM_CLASSES,
) -> pd.DataFrame:
    """Summarize mean chip-level class ratios from metadata."""
    required = [f"label_class_{class_id}_ratio" for class_id in range(num_classes)]
    missing = [col for col in required if col not in metadata.columns]
    if missing:
        raise ValueError(
            "Metadata is missing label distribution columns. "
            f"Missing: {missing}. Run enrich_label_distribution_from_paths() first."
        )

    rows = []
    for class_id in range(num_classes):
        ratio_col = f"label_class_{class_id}_ratio"
        rows.append({
            "class_id": class_id,
            "mean_chip_ratio": float(metadata[ratio_col].mean()) if len(metadata) else 0.0,
            "dominant_chip_count": int(
                (metadata.get("dominant_label_class") == class_id).sum()
            )
            if "dominant_label_class" in metadata.columns
            else 0,
        })
    return pd.DataFrame(rows)


def plan_label_distribution(
    metadata: pd.DataFrame,
    ratio_column: str = "label_non_background_ratio",
    bins: Sequence[float] = (-0.01, 0.0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0),
    group_columns: Iterable[str] = ("dominant_label_class",),
) -> pd.DataFrame:
    """
    Build a chip-count planning table by label-ratio bins and optional groups.

    This is useful before train/validation splitting: inspect which bins/classes
    are underrepresented and sample from each group deliberately.
    """
    if ratio_column not in metadata.columns:
        raise ValueError(
            f"Metadata is missing '{ratio_column}'. "
            "Run enrich_label_distribution_from_paths() first if needed."
        )

    planned = add_label_ratio_bins(metadata, ratio_column=ratio_column, bins=bins)
    group_columns = [col for col in group_columns if col in planned.columns]

    grouping = [*group_columns, "label_ratio_bin"]
    summary = (
        planned.groupby(grouping, observed=True)
        .size()
        .rename("chip_count")
        .reset_index()
    )
    summary["chip_ratio"] = summary["chip_count"] / max(len(planned), 1)
    return summary.sort_values(grouping).reset_index(drop=True)


def add_label_ratio_bins(
    metadata: pd.DataFrame,
    ratio_column: str = "label_non_background_ratio",
    bins: Sequence[float] = (-0.01, 0.0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0),
    bin_column: str = "label_ratio_bin",
) -> pd.DataFrame:
    """Return metadata with a categorical bin column for a label ratio."""
    if ratio_column not in metadata.columns:
        raise ValueError(
            f"Metadata is missing '{ratio_column}'. "
            "Run enrich_label_distribution_from_paths() first if needed."
        )

    planned = metadata.copy()
    planned[bin_column] = pd.cut(
        planned[ratio_column],
        bins=list(bins),
        include_lowest=True,
        duplicates="drop",
    )
    return planned


def sample_train_metadata_by_label_bins(
    metadata: pd.DataFrame,
    keep_ratio: float | Mapping[str, float] = 1.0,
    ratio_column: str = "label_non_background_ratio",
    bins: Sequence[float] = (-0.01, 0.0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0),
    group_columns: Iterable[str] = ("dominant_label_class",),
    random_state: int = 42,
    min_per_group: int = 0,
    drop_bin_column: bool = True,
) -> pd.DataFrame:
    """
    Sample chips from each label-ratio bin/group to build a new train metadata.

    Args:
        metadata: Chip metadata with label ratio columns.
        keep_ratio: Either one global ratio, or a mapping keyed by bin string.
            Use keys like ``"(-0.001, 0.01]"`` from plan_label_distribution().
            A ``"default"`` key is used when a bin-specific ratio is absent.
        ratio_column: Ratio column used for binning.
        bins: Bin edges.
        group_columns: Optional grouping columns in addition to the ratio bin.
        random_state: Random seed for reproducible sampling.
        min_per_group: Minimum chips to keep from each non-empty group.
        drop_bin_column: Remove the helper ``label_ratio_bin`` column before return.

    Returns:
        A sampled train metadata DataFrame.
    """
    planned = add_label_ratio_bins(metadata, ratio_column=ratio_column, bins=bins)
    group_columns = [col for col in group_columns if col in planned.columns]
    grouping = [*group_columns, "label_ratio_bin"]

    if not grouping:
        grouping = ["label_ratio_bin"]

    def ratio_for_group(group: pd.DataFrame) -> float:
        if isinstance(keep_ratio, Mapping):
            bin_key = str(group["label_ratio_bin"].iloc[0])
            ratio = keep_ratio.get(bin_key, keep_ratio.get("default", 1.0))
        else:
            ratio = keep_ratio
        if not 0 <= ratio <= 1:
            raise ValueError(f"keep_ratio must be in [0, 1], got {ratio}")
        return float(ratio)

    sampled_groups = []
    for _, group in planned.groupby(grouping, observed=True):
        ratio = ratio_for_group(group)
        sample_size = int(round(len(group) * ratio))
        if ratio > 0 and min_per_group > 0:
            sample_size = max(sample_size, min(min_per_group, len(group)))
        sample_size = min(sample_size, len(group))
        if sample_size == 0:
            continue
        sampled_groups.append(group.sample(n=sample_size, random_state=random_state))

    if sampled_groups:
        sampled = pd.concat(sampled_groups, ignore_index=True)
    else:
        sampled = planned.iloc[0:0].copy()

    if drop_bin_column and "label_ratio_bin" in sampled.columns:
        sampled = sampled.drop(columns=["label_ratio_bin"])
    return sampled.reset_index(drop=True)


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

    distribution_cols = [
        col for col in metadata.columns
        if (
            col.startswith("label_class_") and col.endswith("_ratio")
        ) or col == "label_non_background_ratio"
    ]
    if distribution_cols:
        stats["label_distribution_columns"] = distribution_cols

    return stats
