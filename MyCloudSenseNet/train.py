"""
云检测模型训练脚本 v2.0

Usage:
    python train.py --config fast --epochs 20
    python train.py --config balanced --epochs 100
    python train.py --config high_accuracy --epochs 150
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Direct imports from submodules (avoid circular imports)
from benchmark.core.cloud_model import CloudModel
from benchmark.core.cloud_dataset import CloudDataset
from benchmark.models.architecture_config import ModelConfig, Configs


def parse_args():
    parser = argparse.ArgumentParser(description="Train cloud detection model")
    parser.add_argument("--config", choices=["fast", "balanced", "high_accuracy"],
                       default="balanced", help="Configuration preset")
    parser.add_argument("--config_path", type=str, help="Path to config JSON")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/val.csv")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--name", type=str, default="cloud_model", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(train_csv: str, val_csv: str):
    """加载数据集"""
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # 获取波段列
    band_cols = [c for c in train_df.columns if c.endswith("_path") and not c.startswith("label")]
    bands = [c.replace("_path", "") for c in band_cols]
    
    X_train = train_df[band_cols]
    y_train = train_df[["label_path"]] if "label_path" in train_df.columns else None
    
    X_val = val_df[band_cols]
    y_val = val_df[["label_path"]] if "label_path" in val_df.columns else None
    
    return X_train, y_train, X_val, y_val, bands


def main():
    args = parse_args()
    
    pl.seed_everything(args.seed)
    
    # 加载配置
    if args.config_path:
        config = ModelConfig.load(Path(args.config_path))
    else:
        config = getattr(Configs, args.config)()
    
    # 覆盖命令行参数
    if args.epochs:
        config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.workers:
        config.num_workers = args.workers
    
    # 保存配置
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save(out_dir / f"{args.name}_config.json")
    
    # 加载数据
    try:
        X_train, y_train, X_val, y_val, bands = load_data(args.train_csv, args.val_csv)
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Bands: {bands}")
    except FileNotFoundError:
        print("Warning: Data files not found, creating dummy data for testing")
        bands = ["B02", "B03", "B04", "B08"]
        X_train = y_train = X_val = y_val = None
    
    # 创建模型
    model = CloudModel(
        bands=bands,
        config=config,
        x_train=X_train,
        y_train=y_train,
        x_val=X_val,
        y_val=y_val,
    )
    
    # 回调
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
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        callbacks=callbacks,
        logger=TensorBoardLogger(out_dir, name=args.name),
        gradient_clip_val=1.0,
    )
    
    # 训练
    print(f"\nStarting training with config: {args.config}")
    print(f"Backbone: {config.backbone}, Epochs: {config.max_epochs}")
    print(f"Bit-depth adaptation: {config.bit_depth_enabled}, Multiscale: {config.bit_depth_multiscale}\n")
    
    trainer.fit(model)
    
    print(f"\nTraining complete!")
    print(f"Best model: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
