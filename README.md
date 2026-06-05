# MyCloudSenseNet

MyCloudSenseNet is a cloud segmentation benchmark for multispectral remote
sensing imagery. It provides a layered Python package for model configuration,
training, inference, GeoTIFF prediction output, evaluation metrics, and
notebook-based experiments.

The current model stack is built around PyTorch Lightning and
`segmentation-models-pytorch`, with optional bit-depth adaptive encoding,
compound losses, boundary-aware optimization, and test-time augmentation (TTA).

## Features

- Cloud segmentation with configurable `unet`, `segformer`, and `deeplabv3+`
  backbones.
- Preset model configs for fast experiments, balanced training, and higher
  accuracy runs.
- Multiband GeoTIFF loading from chip-level CSV metadata.
- PyTorch Lightning training workflow with checkpoints, early stopping, and
  TensorBoard logging.
- Batch inference API that writes georeferenced GeoTIFF masks.
- Optional probability export for downstream analysis.
- IoU and class coverage metrics for chip-level or geospatially aligned
  predictions.

## Project Layout

```text
MyCloudSenseNet/
  benchmark/
    configs/        Model presets and runtime defaults
    core/           GeoTIFF reader, writer, tiling, and raster helpers
    dataset/        Dataset and Lightning DataModule
    evaluation/     Metrics and prediction/label alignment helpers
    inference/      Prediction API and GeoTIFF output helpers
    models/         Model architecture, losses, optimizers, and schedulers
    notebooks/      Experiments, demos, and analysis notebooks
    training/       Training CLI and data-loading workflow
    utils/          Generic filesystem, tensor, and shape helpers
    visualization/  Plotting and display helpers
data/               Local datasets and generated predictions
requirements.txt    Python dependencies
```

The `benchmark` package is the implementation surface. Imports are intentionally
explicit so configuration, data, model, training, inference, and evaluation code
stay loosely coupled.

## Installation

Create a Python environment, install dependencies, and expose the package from
the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/MyCloudSenseNet"
```

For one-off commands, you can also prefix `PYTHONPATH=MyCloudSenseNet`.

## Dataset Metadata

Training and inference expect chip-level CSV files. Each row represents one
image chip and should include a `chip_id`, one path column per band, and
`label_path` for supervised training.

Example:

```csv
chip_id,B02_path,B03_path,B04_path,B08_path,label_path
chip_0001,data/chips/chip_0001_B02.tif,data/chips/chip_0001_B03.tif,data/chips/chip_0001_B04.tif,data/chips/chip_0001_B08.tif,data/labels/chip_0001.tif
```

Band columns are inferred from columns ending in `_path`, excluding
`label_path`. The default band order is `B02`, `B03`, `B04`, `B08`, but the
training CLI will override `config.bands` with the bands found in the training
CSV.

Generated chip metadata also includes label pixel ratio columns. In this
benchmark, class `0` is background:

- `label_class_0_ratio`
- `label_class_1_ratio`
- `label_class_2_ratio`
- `label_non_background_ratio`
- `dominant_label_class`

For existing metadata CSVs, backfill these columns from `label_path` and build
a planning table for train/validation distribution checks:

```python
import pandas as pd

from benchmark.core.metadata_io import (
    enrich_label_distribution_from_paths,
    plan_label_distribution,
    sample_train_metadata_by_label_bins,
    summarize_label_distribution,
)

metadata = pd.read_csv("dataset/train.csv")
metadata = enrich_label_distribution_from_paths(metadata)

class_summary = summarize_label_distribution(metadata)
training_plan = plan_label_distribution(metadata)

train_meta = sample_train_metadata_by_label_bins(
    metadata,
    keep_ratio={
        "(-0.001, 0.01]": 0.2,
        "(0.01, 0.1]": 0.5,
        "default": 1.0,
    },
    min_per_group=1,
)
train_meta.to_csv("dataset/train_sampled.csv", index=False)
```

## Training

Run training from the repository root:

```bash
PYTHONPATH=MyCloudSenseNet python3 -m benchmark.training.train \
  --config balanced \
  --epochs 100 \
  --batch_size 8 \
  --workers 4 \
  --train_csv dataset/train.csv \
  --val_csv dataset/val.csv \
  --output outputs \
  --name cloud_model
```

Useful arguments:

- `--config`: one of `fast`, `balanced`, or `high_accuracy`.
- `--epochs`: overrides the preset epoch count.
- `--batch_size`: overrides the preset batch size.
- `--lr`: overrides the preset learning rate.
- `--workers`: sets DataLoader workers.
- `--train_csv` and `--val_csv`: chip metadata CSV files.
- `--output`: checkpoint and TensorBoard output directory.
- `--seed`: random seed, default `42`.

Checkpoints are written to:

```text
outputs/checkpoints/
```

TensorBoard logs are written under the selected output directory:

```bash
tensorboard --logdir outputs
```

## Model Presets

```python
from benchmark.configs import Configs
from benchmark.models import CloudModel

config = Configs.balanced()
model = CloudModel(config=config)
```

Available presets:

- `fast`: short experiments and smoke tests.
- `balanced`: default research/training setup.
- `high_accuracy`: larger backbone and stronger validation-time settings.

You can mutate a preset before constructing a model:

```python
config = Configs.balanced()
config.model_name = "unet"
config.backbone = "timm-efficientnet-b0"
config.bands = ["B02", "B03", "B04", "B08"]
config.batch_size = 4
config.use_tta = False
```

## Inference

Use the high-level `simple` API to load a checkpoint and save prediction masks:

```python
from pathlib import Path

import pandas as pd

from benchmark.configs import Configs
from benchmark.inference import simple

metadata = pd.read_csv("data/test.csv")

config = Configs.balanced()
config.batch_size = 4
config.num_workers = 2
config.use_tta = False
config.bands = ["B02", "B03", "B04", "B08"]

count = simple(
    model_path=Path("outputs/checkpoints/last.ckpt"),
    input_paths=metadata,
    output_dir=Path("data/predictions"),
    config=config,
    device="auto",
    save_probs=False,
)

print(f"Saved {count} predictions")
```

Inference reads `bands`, `use_tta`, `tta_strategy`, `tta_threshold`,
`batch_size`, and `num_workers` from `ModelConfig`. Create or modify a config
object instead of passing those values through each function call.

Prediction masks are saved as one GeoTIFF per `chip_id`:

```text
data/predictions/<chip_id>.tif
```

When `save_probs=True`, class probabilities are also saved as NumPy arrays:

```text
data/predictions/probabilities/<chip_id>.npy
```

## Evaluation

Core metric helpers live in `benchmark.evaluation`:

```python
import rasterio

from benchmark.evaluation.metrics import intersection_over_union

with rasterio.open("data/predictions/chip_0001.tif") as src:
    pred = src.read(1)

with rasterio.open("data/labels/chip_0001.tif") as src:
    true = src.read(1)

miou, per_class_iou, true_coverage, pred_coverage = intersection_over_union(
    pred,
    true,
    n_classes=3,
)
```

For full-image labels and smaller georeferenced chip predictions, use
`read_prediction_and_aligned_true` before computing metrics.

## Notebooks

Example and exploratory notebooks are stored in:

```text
MyCloudSenseNet/benchmark/notebooks/
```

Keep reusable training, inference, model, and evaluation logic in the
`benchmark` package rather than inside notebooks.

## Dependency Direction

The package is organized so dependencies flow in one direction:

```text
configs -> core
configs -> dataset
configs -> models
configs + dataset + models -> training
configs + core + dataset + models + utils -> inference
configs + core -> evaluation
configs + core + evaluation -> visualization
```

This keeps runtime configuration central while avoiding circular imports between
model, data, and inference code.
