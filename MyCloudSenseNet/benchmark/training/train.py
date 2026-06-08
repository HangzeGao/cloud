"""Training CLI for the benchmark package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from ..configs import Configs, ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cloud segmentation model")
    parser.add_argument(
        "--config",
        choices=["fast", "balanced", "high_accuracy"],
        default="balanced",
        help="Configuration preset",
    )
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--train_csv", type=str, default="dataset/train.csv")
    parser.add_argument("--val_csv", type=str, default="dataset/val.csv")
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--run_test", action="store_true", help="Run trainer.test after training")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--name", type=str, default="cloud_model", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(
    train_csv: str,
    val_csv: str,
    test_csv: Optional[str] = None,
) -> Tuple[
    pd.DataFrame,
    Optional[pd.DataFrame],
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    list[str],
]:
    """Load chip metadata and infer spectral bands from ``*_path`` columns."""
    import pandas as pd

    train_df = pd.read_csv(train_csv).reset_index(drop=True)
    val_df = pd.read_csv(val_csv).reset_index(drop=True)
    test_df = pd.read_csv(test_csv).reset_index(drop=True) if test_csv else None

    band_cols = [
        col for col in train_df.columns if col.endswith("_path") and not col.startswith("label")
    ]
    bands = [col.removesuffix("_path") for col in band_cols]

    def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        x = df[band_cols]
        y = df[["label_path"]] if "label_path" in df.columns else None
        if "chip_id" in df.columns:
            x = pd.concat([df[["chip_id"]], x], axis=1)
        return x, y

    x_train, y_train = split_xy(train_df)
    x_val, y_val = split_xy(val_df)
    x_test, y_test = split_xy(test_df) if test_df is not None else (None, None)

    return x_train, y_train, x_val, y_val, x_test, y_test, bands


def apply_overrides(config: ModelConfig, args: argparse.Namespace) -> ModelConfig:
    """Apply CLI overrides to a preset config."""
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.workers is not None:
        config.num_workers = args.workers
    return config


def main() -> None:
    args = parse_args()

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from ..dataset import CloudDataModule
    from ..models.cloud_model import CloudModel

    pl.seed_everything(args.seed)

    config = apply_overrides(getattr(Configs, args.config)(), args)
    x_test = y_test = None

    try:
        x_train, y_train, x_val, y_val, x_test, y_test, bands = load_data(
            args.train_csv,
            args.val_csv,
            args.test_csv,
        )
        config.bands = bands
        config.validate()
        print(f"Training samples: {len(x_train)}, validation samples: {len(x_val)}")
        if x_test is not None:
            print(f"Test samples: {len(x_test)}")
        print(f"Bands: {config.bands}")
    except FileNotFoundError:
        print("Warning: training CSV files were not found; building the model only.")
        x_train = y_train = x_val = y_val = None

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    datamodule = CloudDataModule(
        config=config,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
    model = CloudModel(config=config)

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir / "checkpoints",
            filename="{epoch:02d}-{val/avg_iou:.4f}",
            monitor="val/avg_iou",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val/avg_iou", mode="max", patience=10),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        callbacks=callbacks,
        logger=TensorBoardLogger(out_dir, name=args.name),
        gradient_clip_val=1.0,
    )

    print(f"\nStarting training with config: {args.config}")
    print(f"Backbone: {config.backbone}, epochs: {config.max_epochs}")
    print(f"Bands: {config.bands}, channels: {config.in_channels}")
    print(f"Bit-depth adaptation: {config.bit_depth_enabled}\n")

    trainer.fit(model, datamodule=datamodule)

    if args.run_test:
        if x_test is None:
            raise ValueError("--run_test requires --test_csv.")
        if y_test is None:
            raise ValueError("--run_test requires test_csv to contain label_path.")
        ckpt_path = "best" if trainer.checkpoint_callback.best_model_path else None
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    print("\nTraining complete.")
    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
