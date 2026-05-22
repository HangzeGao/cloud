"""
CloudSense-Net 训练脚本
========================

支持多种模型架构的训练，针对 Apple Silicon (MPS)、CUDA 和 CPU 优化。

主要功能:
- 多数据集混合训练
- 分层学习率优化
- 早停机制
- TensorBoard 日志记录
- 梯度累积
- MPS 显存管理

使用示例:
    # MPS 训练 (Apple Silicon)
    python train.py --config configs/cloudseg_mps.yaml --device mps

    # CUDA 训练
    python train.py --config configs/cloudseg_base.yaml --device cuda --gpu 0

    # 恢复训练
    python train.py --config configs/cloudseg_mps.yaml --resume experiments/best_model.pth

    # 开发模式（快速验证）
    python train.py --config configs/cloudseg_mps.yaml --dev-run
"""
import os
import sys
import argparse
import random
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model
from models.heads.losses import CombinedLoss
from utils import load_config, save_config, SegmentationMetrics, AverageMeter
from data import create_mixed_dataloaders


# =============================================================================
# 常量定义
# =============================================================================

DEFAULT_CONFIG = 'configs/cloudseg_base.yaml'
GRADIENT_CLIP_NORM = 1.0
TB_LOG_INTERVAL = 10


# =============================================================================
# 配置数据类
# =============================================================================

@dataclass
class TrainingState:
    """训练状态数据类"""
    epoch: int = 0
    best_miou: float = 0.0
    patience_counter: int = 0
    global_step: int = 0


@dataclass
class TrainingConfig:
    """训练配置数据类"""
    # 梯度累积
    use_grad_accum: bool = False
    accum_steps: int = 1

    # 显存管理
    empty_cache_every_n: int = 0
    monitor_interval: int = 0

    # 早停
    patience: int = 0

    # 混合精度训练 (AMP)
    use_amp: bool = False
    amp_dtype: str = "float16"

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TrainingConfig':
        """从配置字典创建"""
        grad_accum = config.get('training', {}).get('gradient_accumulation', {})
        memory = config.get('training', {}).get('memory', {})
        amp_config = config.get('training', {}).get('amp', {})

        return cls(
            use_grad_accum=grad_accum.get('enabled', False),
            accum_steps=grad_accum.get('steps', 1) if grad_accum.get('enabled', False) else 1,
            empty_cache_every_n=memory.get('empty_cache_every_n_batches', 0),
            monitor_interval=memory.get('monitor_interval', 0),
            patience=config['training'].get('patience', 0),
            use_amp=amp_config.get('enabled', False),
            amp_dtype=amp_config.get('dtype', 'float16')
        )


# =============================================================================
# 随机种子设置
# =============================================================================

class SeedManager:
    """随机种子管理器"""

    @staticmethod
    def set_seed(seed: int) -> None:
        """
        设置全局随机种子以确保实验可复现

        Args:
            seed: 随机种子值
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # CUDA 特定设置
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # MPS (Apple Silicon) 特定设置
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)


# =============================================================================
# 设备管理
# =============================================================================

class DeviceManager:
    """设备管理器"""

    @staticmethod
    def get_device(preference: str = 'auto', gpu_id: int = 0) -> torch.device:
        """
        获取最佳可用计算设备

        优先级: MPS (Apple Silicon) > CUDA > CPU

        Args:
            preference: 设备类型 ('auto', 'mps', 'cuda', 'cpu')
            gpu_id: CUDA GPU ID

        Returns:
            torch.device: 可用的计算设备
        """
        if preference == 'auto':
            preference = DeviceManager._detect_best_device()

        device = DeviceManager._try_create_device(preference, gpu_id)
        DeviceManager._print_device_info(device)
        return device

    @staticmethod
    def _detect_best_device() -> str:
        """自动检测最佳可用设备"""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    @staticmethod
    def _try_create_device(device_type: str, gpu_id: int) -> torch.device:
        """尝试创建设备，失败则回退"""
        if device_type == 'mps':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            print("⚠️  MPS not available, falling back to CUDA/CPU")
            return DeviceManager._try_create_device('cuda', gpu_id)

        elif device_type == 'cuda':
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{gpu_id}')
                return device
            print("⚠️  CUDA not available, falling back to CPU")

        return torch.device('cpu')

    @staticmethod
    def _print_device_info(device: torch.device) -> None:
        """打印设备信息"""
        print(f"\n{'=' * 60}")
        if device.type == 'mps':
            print("✅ Using Apple MPS (Metal Performance Shaders)")
            print("   Device: Apple Silicon (M1/M2/M3)")
        elif device.type == 'cuda':
            print("✅ Using CUDA")
            print(f"   Device: {torch.cuda.get_device_name(device)}")
        else:
            print("⚠️  Using CPU")
        print(f"{'=' * 60}\n")


# =============================================================================
# 优化器和调度器
# =============================================================================

class OptimizerFactory:
    """优化器工厂"""

    @staticmethod
    def create(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
        """
        创建优化器并配置分层学习率

        Backbone 使用较低学习率，其他层使用较高学习率，
        以稳定预训练特征提取器的微调过程。

        Args:
            model: 待优化的模型
            config: 训练配置字典

        Returns:
            optim.Optimizer: 配置好的优化器
        """
        opt_cfg = config['training']['optimizer']

        # 解析配置参数
        base_lr = float(str(opt_cfg['lr']).replace('e', 'E'))
        weight_decay = float(str(opt_cfg['weight_decay']).replace('e', 'E'))
        backbone_lr_mult = float(str(opt_cfg.get('backbone_lr_mult', 0.1)))

        print(f"[Optimizer] Configuration:")
        print(f"  Base LR: {base_lr}")
        print(f"  Weight Decay: {weight_decay}")
        print(f"  Backbone LR Multiplier: {backbone_lr_mult}")

        # 分层参数组
        backbone_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if 'encoder' in name or 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': base_lr * backbone_lr_mult,
             'weight_decay': weight_decay, 'name': 'backbone'},
            {'params': other_params, 'lr': base_lr,
             'weight_decay': weight_decay, 'name': 'head'}
        ]

        # 创建优化器
        optimizer_type = opt_cfg['type'].lower()
        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(param_groups)
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(param_groups)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(param_groups, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        return optimizer


class WarmupScheduler:
    """带 warmup 的学习率调度器包装器"""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        base_scheduler: optim.lr_scheduler._LRScheduler
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.current_step = 0

        # 保存初始学习率
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch=None):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # warmup 阶段：线性增加学习率
            warmup_factor = self.current_step / self.warmup_steps
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i] * warmup_factor
        else:
            # warmup 结束后：使用基础调度器
            if epoch is not None:
                self.base_scheduler.step(epoch)
            else:
                self.base_scheduler.step()

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'base_scheduler': self.base_scheduler.state_dict(),
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
        self.base_lrs = state_dict['base_lrs']


class SchedulerFactory:
    """学习率调度器工厂"""

    @staticmethod
    def create(
        optimizer: optim.Optimizer,
        config: Dict[str, Any],
        steps_per_epoch: int
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        创建学习率调度器

        支持多种调度策略: cosine_warmup, cosine, step, plateau
        支持 warmup_epochs 配置

        Args:
            optimizer: 优化器实例
            config: 训练配置字典
            steps_per_epoch: 每个epoch的步数

        Returns:
            Optional[LRScheduler]: 学习率调度器
        """
        scheduler_cfg = config['training']['scheduler']
        scheduler_type = scheduler_cfg['type'].lower()

        # 获取 warmup 配置
        warmup_epochs = scheduler_cfg.get('warmup_epochs', 0)
        warmup_steps = warmup_epochs * steps_per_epoch

        base_scheduler = None

        if scheduler_type == 'cosine_warmup':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            T_0 = scheduler_cfg.get('T_0', 10)
            T_mult = scheduler_cfg.get('T_mult', 2)
            base_scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_0 * steps_per_epoch,
                T_mult=T_mult
            )
        elif scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            # 如果有 warmup，调整总 epoch 数
            total_steps = config['training']['num_epochs'] * steps_per_epoch
            if warmup_epochs > 0:
                total_steps -= warmup_steps
            base_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
        elif scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            step_size = scheduler_cfg.get('step_size', 30)
            gamma = scheduler_cfg.get('gamma', 0.1)
            base_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            base_scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.1,
                patience=5,
                verbose=True
            )
        else:
            base_scheduler = None

        # 如果有 warmup 配置且是基础调度器，包装它
        if warmup_epochs > 0 and base_scheduler is not None:
            print(f"[Scheduler] Enabled warmup for {warmup_epochs} epochs "
                  f"({warmup_steps} steps)")
            return WarmupScheduler(optimizer, warmup_steps, base_scheduler)

        return base_scheduler


# =============================================================================
# 损失函数
# =============================================================================

class LossManager:
    """损失函数管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.criterion = CombinedLoss({
            'types': ['dice', 'bce'],
            'weights': [0.5, 0.5]
        })

    def compute(self, outputs: Any, masks: torch.Tensor) -> torch.Tensor:
        """计算损失"""
        if isinstance(outputs, dict):
            return self.criterion(outputs['logits'], masks)
        return self.criterion(outputs, masks)


# =============================================================================
# 训练循环
# =============================================================================

class Trainer:
    """训练器 - 支持混合精度训练 (AMP)"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        writer: Optional[SummaryWriter],
        train_cfg: TrainingConfig
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = writer
        self.train_cfg = train_cfg
        self.loss_manager = LossManager({})

        # 初始化 AMP (Automatic Mixed Precision)
        self.scaler = None
        if train_cfg.use_amp and device.type == 'cuda':
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print(f"[Trainer] AMP enabled with dtype={train_cfg.amp_dtype}")
        elif train_cfg.use_amp and device.type != 'cuda':
            print(f"[Trainer] Warning: AMP requested but device is {device.type}, "
                  "AMP only works on CUDA. Disabling AMP.")

    def _forward_pass(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> Any:
        """执行前向传播，支持 AMP autocast"""
        if self.scaler is not None:
            from torch.cuda.amp import autocast
            with autocast(dtype=torch.float16 if self.train_cfg.amp_dtype == 'float16' else torch.bfloat16):
                if self.model.use_semantic_enhancement and masks is not None:
                    return self.model(images, masks)
                else:
                    return self.model(images)
        else:
            if self.model.use_semantic_enhancement and masks is not None:
                return self.model(images, masks)
            else:
                return self.model(images)

    def train_epoch(
        self,
        train_loader,
        epoch: int,
        state: TrainingState
    ) -> Tuple[float, Dict[str, float]]:
        """
        执行一个训练 epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前 epoch 编号
            state: 训练状态

        Returns:
            Tuple[float, Dict]: (平均损失, 训练指标字典)
        """
        self.model.train()

        losses = AverageMeter()
        metrics = SegmentationMetrics(num_classes=2)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # 数据加载
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)

            # 前向传播 - 支持 AMP
            outputs = self._forward_pass(images, masks)

            # 损失计算（考虑梯度累积）
            loss = self.loss_manager.compute(outputs, masks) / self.train_cfg.accum_steps

            # 反向传播 - 支持 AMP
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 更新统计（使用 scaler 时需要在更新前获取原始 loss）
            if self.scaler is not None:
                actual_loss = loss.item() * self.train_cfg.accum_steps
            else:
                actual_loss = loss.item() * self.train_cfg.accum_steps
            losses.update(actual_loss, images.size(0))

            with torch.no_grad():
                pred = outputs['logits'] if isinstance(outputs, dict) else outputs
                metrics.update(pred, masks)

            # 梯度更新
            self._gradient_step(batch_idx, len(train_loader))

            # MPS 显存管理
            self._memory_management(batch_idx)

            # 更新进度条
            self._update_progress_bar(pbar, losses, metrics, epoch, batch_idx)

            # TensorBoard 记录
            self._log_to_tensorboard(epoch, batch_idx, len(train_loader), actual_loss)

            state.global_step += 1

        # 确保剩余梯度更新
        self._finalize_gradient_step(len(train_loader))

        return losses.avg, metrics.compute()

    def _gradient_step(self, batch_idx: int, num_batches: int) -> None:
        """执行梯度更新步骤 - 支持 AMP"""
        cfg = self.train_cfg
        if (batch_idx + 1) % cfg.accum_steps == 0 or (batch_idx + 1) == num_batches:
            # AMP: 先 unscale 再裁剪
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRADIENT_CLIP_NORM)

            # AMP: 使用 scaler.step 和 scaler.update
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            if self.scheduler and isinstance(self.scheduler, (
                optim.lr_scheduler.CosineAnnealingWarmRestarts,
                optim.lr_scheduler.CosineAnnealingLR,
                optim.lr_scheduler.StepLR
            )):
                self.scheduler.step()

    def _finalize_gradient_step(self, num_batches: int) -> None:
        """确保所有梯度都已更新 - 支持 AMP"""
        cfg = self.train_cfg
        if cfg.use_grad_accum and num_batches % cfg.accum_steps != 0:
            # AMP: 先 unscale 再裁剪
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=GRADIENT_CLIP_NORM)

            # AMP: 使用 scaler.step 和 scaler.update
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

    def _memory_management(self, batch_idx: int) -> None:
        """管理设备显存"""
        if self.device.type == 'mps' and self.train_cfg.empty_cache_every_n > 0:
            if (batch_idx + 1) % self.train_cfg.empty_cache_every_n == 0:
                torch.mps.empty_cache()

        if self.device.type == 'mps' and self.train_cfg.monitor_interval > 0:
            if batch_idx % self.train_cfg.monitor_interval == 0 and self.writer:
                mem_allocated = torch.mps.current_allocated_memory() / 1024 ** 2
                self.writer.add_scalar('system/mps_memory_mb', mem_allocated, self.state.global_step)

    def _update_progress_bar(
        self,
        pbar: tqdm,
        losses: AverageMeter,
        metrics: SegmentationMetrics,
        epoch: int,
        batch_idx: int
    ) -> None:
        """更新进度条显示"""
        postfix = {
            'loss': f'{losses.avg:.4f}',
            'mIoU': f'{metrics.compute()["mIoU"]:.4f}',
            'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
        }

        if self.train_cfg.use_grad_accum:
            postfix['accum'] = f'{(batch_idx % self.train_cfg.accum_steps) + 1}/{self.train_cfg.accum_steps}'

        pbar.set_postfix(postfix)

    def _log_to_tensorboard(
        self,
        epoch: int,
        batch_idx: int,
        num_batches: int,
        loss: float
    ) -> None:
        """记录到TensorBoard"""
        if self.writer and batch_idx % TB_LOG_INTERVAL == 0:
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('train/loss', loss, global_step)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], global_step)

    @property
    def state(self) -> TrainingState:
        """获取当前训练状态（用于属性访问）"""
        # 这里需要外部设置，简化处理
        return TrainingState()


class Validator:
    """验证器"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        writer: Optional[SummaryWriter]
    ):
        self.model = model
        self.device = device
        self.writer = writer
        self.loss_manager = LossManager({})

    @torch.no_grad()
    def validate(
        self,
        val_loader,
        epoch: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        执行验证

        Args:
            val_loader: 验证数据加载器
            epoch: 当前 epoch 编号

        Returns:
            Tuple[float, Dict]: (平均损失, 验证指标字典)
        """
        self.model.eval()

        losses = AverageMeter()
        metrics = SegmentationMetrics(num_classes=2)

        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)

            # 验证时也传入 masks 以支持语义增强模块
            if self.model.use_semantic_enhancement:
                outputs = self.model(images, masks)
            else:
                outputs = self.model(images)
            loss = self.loss_manager.compute(outputs, masks)

            pred = outputs['logits'] if isinstance(outputs, dict) else outputs

            losses.update(loss.item(), images.size(0))
            metrics.update(pred, masks)

        val_metrics = metrics.compute()

        # TensorBoard 记录
        if self.writer:
            self.writer.add_scalar('val/loss', losses.avg, epoch)
            self.writer.add_scalar('val/mIoU', val_metrics['mIoU'], epoch)
            self.writer.add_scalar('val/mDice', val_metrics['mDice'], epoch)
            self.writer.add_scalar('val/pixel_acc', val_metrics['pixel_acc'], epoch)

        return losses.avg, val_metrics


# =============================================================================
# 模型持久化
# =============================================================================

class CheckpointManager:
    """检查点管理器"""

    @staticmethod
    def save(
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        best_miou: float,
        config: Dict[str, Any],
        save_dir: str,
        is_best: bool = False
    ) -> str:
        """
        保存训练检查点

        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前 epoch
            best_miou: 最佳 mIoU
            config: 配置字典
            save_dir: 保存目录
            is_best: 是否为最佳模型

        Returns:
            str: 保存路径
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou': best_miou,
            'config': config
        }

        if is_best:
            save_path = os.path.join(save_dir, 'best_model.pth')
        else:
            save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')

        torch.save(checkpoint, save_path)
        return save_path

    @staticmethod
    def load(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        device: torch.device = None
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        加载训练检查点

        Args:
            checkpoint_path: 检查点路径
            model: 要加载权重的模型
            optimizer: 优化器（可选）
            device: 设备

        Returns:
            Tuple[int, float, Dict]: (epoch, best_miou, config)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return (
            checkpoint.get('epoch', 0),
            checkpoint.get('best_miou', 0.0),
            checkpoint.get('config', {})
        )


# =============================================================================
# 训练流程控制
# =============================================================================

class EarlyStoppingManager:
    """早停管理器"""

    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_miou = 0.0

    def check(self, miou: float) -> Tuple[bool, bool]:
        """
        检查早停状态

        Args:
            miou: 当前mIoU

        Returns:
            Tuple[is_best, should_stop]: (是否最佳, 是否应该停止)
        """
        if miou > self.best_miou:
            self.best_miou = miou
            self.counter = 0
            return True, False
        else:
            self.counter += 1
            should_stop = self.patience > 0 and self.counter >= self.patience
            return False, should_stop

    def get_status(self) -> str:
        """获取当前状态描述"""
        if self.patience == 0:
            return "Disabled"
        return f"{self.counter}/{self.patience}"


# =============================================================================
# 命令行参数
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='Train CloudSense-Net - Cloud Segmentation Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # MPS 训练 (Apple Silicon)
  python train.py --config configs/cloudseg_mps.yaml --device mps

  # CUDA 训练
  python train.py --config configs/cloudseg_base.yaml --device cuda --gpu 0

  # 恢复训练
  python train.py --config configs/cloudseg_mps.yaml --resume experiments/best_model.pth

  # 开发模式（小数据集快速验证）
  python train.py --config configs/cloudseg_mps.yaml --dev-run
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=DEFAULT_CONFIG,
        help=f'配置文件路径 (默认: {DEFAULT_CONFIG})'
    )
    parser.add_argument(
        '--resume', '-r',
        type=str,
        default=None,
        help='从检查点恢复训练'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'mps', 'cuda', 'cpu'],
        help='计算设备 (默认: auto，优先 MPS)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='CUDA GPU ID'
    )
    parser.add_argument(
        '--exp_name', '-n',
        type=str,
        default=None,
        help='实验名称（用于保存目录）'
    )
    parser.add_argument(
        '--dev-run',
        action='store_true',
        help='开发模式：使用少量数据快速验证'
    )

    return parser


# =============================================================================
# 环境设置
# =============================================================================

class TrainingEnvironment:
    """训练环境设置"""

    @staticmethod
    def setup(args: argparse.Namespace) -> Tuple[torch.device, Dict[str, Any], str]:
        """
        设置训练环境

        Args:
            args: 命令行参数

        Returns:
            Tuple[device, config, save_dir]: 设备、配置、保存目录
        """
        # 加载配置
        config = load_config(args.config)

        # 设置设备
        device = DeviceManager.get_device(args.device, args.gpu)

        # 设置随机种子
        SeedManager.set_seed(config['experiment']['seed'])

        # 创建保存目录
        exp_name = args.exp_name or config['model']['name']
        save_dir = os.path.join(config['experiment']['save_dir'], exp_name)
        os.makedirs(save_dir, exist_ok=True)

        # 保存配置
        save_config(config, os.path.join(save_dir, 'config.yaml'))

        # 打印信息
        TrainingEnvironment._print_setup_info(exp_name, save_dir, config)

        return device, config, save_dir

    @staticmethod
    def _print_setup_info(exp_name: str, save_dir: str, config: Dict[str, Any]) -> None:
        """打印设置信息"""
        print(f"\n{'=' * 60}")
        print(f"Experiment: {exp_name}")
        print(f"Save Dir:   {save_dir}")
        print(f"Epochs:     {config['training']['num_epochs']}")
        print(f"Batch Size: {config['training']['batch_size']}")
        print(f"{'=' * 60}\n")


# =============================================================================
# 主训练流程
# =============================================================================

def main():
    """主训练流程"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # 设置环境
    device, config, save_dir = TrainingEnvironment.setup(args)

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))

    # 数据加载
    print("Loading data...")
    train_loader, val_loader, _ = create_mixed_dataloaders(config, dev_run=args.dev_run)

    if train_loader is None:
        raise RuntimeError("Failed to create train loader")

    print(f"  Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"  Val samples:   {len(val_loader.dataset)}")

    # 模型创建
    print("\nCreating model...")
    model = build_model(config)
    model.print_architecture()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # 优化器和调度器
    optimizer = OptimizerFactory.create(model, config)
    scheduler = SchedulerFactory.create(optimizer, config, len(train_loader))

    # 训练配置
    train_cfg = TrainingConfig.from_dict(config)

    # 训练状态
    state = TrainingState()
    early_stopping = EarlyStoppingManager(train_cfg.patience)

    if train_cfg.patience > 0:
        print(f"\n[Early Stopping] Enabled (patience={train_cfg.patience})")

    # 恢复训练
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        state.epoch, state.best_miou, _ = CheckpointManager.load(
            args.resume, model, optimizer, device
        )
        early_stopping.best_miou = state.best_miou
        print(f"  Resumed from epoch {state.epoch}, best mIoU: {state.best_miou:.4f}")

    # 训练组件
    trainer = Trainer(model, optimizer, scheduler, device, writer, train_cfg)
    validator = Validator(model, device, writer)

    # 训练循环
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)

    num_epochs = config['training']['num_epochs']
    val_interval = config['experiment'].get('val_interval', 1)

    try:
        for epoch in range(state.epoch, num_epochs):
            state.epoch = epoch
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            print("-" * 40)

            # 训练阶段
            train_loss, train_metrics = trainer.train_epoch(train_loader, epoch, state)
            print(f"Train - Loss: {train_loss:.4f}, mIoU: {train_metrics['mIoU']:.4f}, "
                  f"Dice: {train_metrics['mDice']:.4f}")

            # 验证阶段
            if val_loader and epoch % val_interval == 0:
                val_loss, val_metrics = validator.validate(val_loader, epoch)
                print(f"Val   - Loss: {val_loss:.4f}, mIoU: {val_metrics['mIoU']:.4f}, "
                      f"Dice: {val_metrics['mDice']:.4f}")

                # ReduceLROnPlateau 调度（可能被 WarmupScheduler 包装）
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics['mIoU'])
                elif hasattr(scheduler, 'base_scheduler') and isinstance(scheduler.base_scheduler, ReduceLROnPlateau):
                    scheduler.base_scheduler.step(val_metrics['mIoU'])

                # 检查早停和保存最佳模型
                is_best, should_stop = early_stopping.check(val_metrics['mIoU'])

                if is_best:
                    save_path = CheckpointManager.save(
                        model, optimizer, epoch, val_metrics['mIoU'],
                        config, save_dir, is_best=True
                    )
                    print(f"✓ Saved best model (mIoU: {val_metrics['mIoU']:.4f}) to {save_path}")
                else:
                    print(f"  No improvement ({early_stopping.get_status()})")

                    if should_stop:
                        print(f"\n[Early Stopping] Triggered after {train_cfg.patience} epochs without improvement")
                        break

            # 定期保存检查点
            save_interval = config['experiment'].get('save_interval', 10)
            if epoch > 0 and epoch % save_interval == 0:
                save_path = CheckpointManager.save(
                    model, optimizer, epoch, early_stopping.best_miou,
                    config, save_dir, is_best=False
                )
                print(f"✓ Saved checkpoint: {save_path}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    except Exception as e:
        print(f"\n\nTraining error: {e}")
        raise

    finally:
        writer.close()
        print(f"\n{'=' * 60}")
        print("Training Completed!")
        print(f"  Best mIoU: {early_stopping.best_miou:.4f}")
        print(f"  Results:   {save_dir}")
        print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
