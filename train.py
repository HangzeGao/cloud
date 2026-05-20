"""
训练脚本
支持多种模型架构配置的训练
支持 Apple Silicon MPS (Metal Performance Shaders)
"""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CloudSenseNet, build_model
from utils import load_config, save_config, SegmentationMetrics, AverageMeter
from data import get_data_loaders, create_mixed_dataloaders


def set_seed(seed: int):
    """设置随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_optimizer(model: nn.Module, config: dict):
    """
    获取优化器，支持分层学习率
    """
    opt_cfg = config['training']['optimizer']
    
    # 确保数值类型正确（从YAML加载的可能是字符串）
    base_lr = float(str(opt_cfg['lr']).replace('e', 'E'))
    weight_decay = float(str(opt_cfg['weight_decay']).replace('e', 'E'))
    backbone_lr_mult = float(str(opt_cfg.get('backbone_lr_mult', 0.1)))
    
    print(f"[Optimizer] base_lr={base_lr}, weight_decay={weight_decay}, backbone_lr_mult={backbone_lr_mult}")
    
    # 分层参数组
    backbone_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 识别backbone参数
        if 'encoder' in name or 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    param_groups = [
        {'params': backbone_params, 'lr': base_lr * backbone_lr_mult, 'weight_decay': weight_decay},
        {'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay}
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


def get_scheduler(optimizer: optim.Optimizer, config: dict, steps_per_epoch: int):
    """
    获取学习率调度器
    """
    scheduler_cfg = config['training']['scheduler']
    scheduler_type = scheduler_cfg['type'].lower()
    
    if scheduler_type == 'cosine_warmup':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        T_0 = scheduler_cfg.get('T_0', 10)
        T_mult = scheduler_cfg.get('T_mult', 2)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0 * steps_per_epoch,
            T_mult=T_mult
        )
    elif scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        T_max = config['training']['num_epochs'] * steps_per_epoch
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = scheduler_cfg.get('step_size', 30)
        gamma = scheduler_cfg.get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=5,
            verbose=True
        )
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model: nn.Module, train_loader, optimizer, scheduler, epoch: int, device, writer=None):
    """训练一个epoch"""
    model.train()
    
    losses = AverageMeter()
    metrics = SegmentationMetrics(num_classes=2)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        if isinstance(outputs, dict):
            loss = model.get_loss(outputs, masks)
        else:
            # 如果没有自定义损失，使用默认损失
            from models.heads.losses import CombinedLoss
            criterion = CombinedLoss({'types': ['dice', 'bce'], 'weights': [0.5, 0.5]})
            loss = criterion(outputs, masks)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler is not None and isinstance(scheduler, (optim.lr_scheduler.CosineAnnealingWarmRestarts, 
                                                          optim.lr_scheduler.CosineAnnealingLR,
                                                          optim.lr_scheduler.StepLR)):
            scheduler.step()
        
        # 更新统计
        losses.update(loss.item(), images.size(0))
        
        with torch.no_grad():
            if isinstance(outputs, dict):
                pred = outputs['logits']
            else:
                pred = outputs
            metrics.update(pred, masks)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'mIoU': f'{metrics.compute()["mIoU"]:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # TensorBoard记录
        if writer is not None and batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
    
    # 计算epoch平均指标
    epoch_metrics = metrics.compute()
    
    return losses.avg, epoch_metrics


@torch.no_grad()
def validate(model: nn.Module, val_loader, epoch: int, device, writer=None):
    """验证"""
    model.eval()
    
    losses = AverageMeter()
    metrics = SegmentationMetrics(num_classes=2)
    
    for batch in tqdm(val_loader, desc=f"Validation"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        if isinstance(outputs, dict):
            from models.heads.losses import CombinedLoss
            criterion = CombinedLoss({'types': ['dice', 'bce'], 'weights': [0.5, 0.5]})
            loss = criterion(outputs['logits'], masks)
        else:
            from models.heads.losses import CombinedLoss
            criterion = CombinedLoss({'types': ['dice', 'bce'], 'weights': [0.5, 0.5]})
            loss = criterion(outputs, masks)
        
        losses.update(loss.item(), images.size(0))
        
        # 更新指标
        if isinstance(outputs, dict):
            pred = outputs['logits']
        else:
            pred = outputs
        metrics.update(pred, masks)
    
    # 计算指标
    val_metrics = metrics.compute()
    
    # TensorBoard记录
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/mIoU', val_metrics['mIoU'], epoch)
        writer.add_scalar('val/mDice', val_metrics['mDice'], epoch)
        writer.add_scalar('val/pixel_acc', val_metrics['pixel_acc'], epoch)
    
    return losses.avg, val_metrics


def get_device(device_arg: str = 'auto', gpu_id: int = 0):
    """
    获取最佳可用设备
    优先级：MPS (Apple Silicon) > CUDA > CPU
    """
    # 检查MPS (Apple Silicon)
    if device_arg == 'auto' or device_arg == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"✅ Using Apple MPS (Metal Performance Shaders)")
            print(f"   Device: Apple Silicon (M1/M2/M3)")
            return device
        elif device_arg == 'mps':
            print(f"⚠️  MPS not available, falling back to CPU")
    
    # 检查CUDA
    if device_arg == 'auto' or device_arg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            print(f"✅ Using CUDA")
            print(f"   Device: {torch.cuda.get_device_name(gpu_id)}")
            return device
        elif device_arg == 'cuda':
            print(f"⚠️  CUDA not available, falling back to CPU")
    
    # 默认CPU
    device = torch.device('cpu')
    print(f"⚠️  Using CPU")
    return device


def main():
    parser = argparse.ArgumentParser(description='Train CloudSense-Net')
    parser.add_argument('--config', type=str, default='configs/cloudseg_base.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'mps', 'cuda', 'cpu'],
                        help='Device to use (auto prefers MPS on Apple Silicon)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use (for CUDA)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--dev-run', action='store_true',
                        help='Quick dev run with small dataset subset (20 train, 10 val samples)')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = get_device(args.device, args.gpu)
    
    # 设置随机种子
    set_seed(config['experiment']['seed'])
    
    # 创建保存目录
    exp_name = args.exp_name or config['model']['name']
    save_dir = os.path.join(config['experiment']['save_dir'], exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置
    save_config(config, os.path.join(save_dir, 'config.yaml'))
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))
    
    # 获取数据加载器
    print("Loading data...")
    
    # 检查是否启用多数据集混合训练
    if config['data'].get('multi_dataset', False):
        print("[Train] Multi-dataset mixed training mode enabled")
        train_loader, val_loader, test_loader = create_mixed_dataloaders(config, dev_run=args.dev_run)
    else:
        train_loader, val_loader, test_loader = get_data_loaders(config, dev_run=args.dev_run)
    
    if train_loader is None:
        raise RuntimeError("Failed to create train_loader")
    if val_loader is None:
        print("Warning: No validation loader created")
    
    # 创建模型
    print("Creating model...")
    model = build_model(config)
    model.print_architecture()
    model = model.to(device)
    
    # 创建优化器和调度器
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    
    # 恢复训练
    start_epoch = 0
    best_miou = 0.0
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0.0)
    
    # 训练循环
    print("\nStarting training...")
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        # 训练
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, epoch, device, writer
        )
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train Metrics: {train_metrics}")
        
        # 验证
        if epoch % config['experiment']['val_interval'] == 0:
            val_loss, val_metrics = validate(model, val_loader, epoch, device, writer)
            
            print(f"\nVal Loss: {val_loss:.4f}")
            print(f"Val Metrics: {val_metrics}")
            
            # 学习率调度（对于ReduceLROnPlateau）
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['mIoU'])
            
            # 保存最佳模型
            if val_metrics['mIoU'] > best_miou:
                best_miou = val_metrics['mIoU']
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_miou': best_miou,
                    'config': config
                }
                
                save_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(checkpoint, save_path)
                print(f"Saved best model (mIoU: {best_miou:.4f}) to {save_path}")
        
        # 定期保存checkpoint
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'config': config
            }
            save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)
    
    print(f"\nTraining completed! Best mIoU: {best_miou:.4f}")
    writer.close()


if __name__ == '__main__':
    main()
