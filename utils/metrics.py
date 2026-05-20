"""
分割评估指标
"""
import torch
import numpy as np
from typing import Dict, List


class SegmentationMetrics:
    """
    语义分割评估指标
    支持：IoU, Dice, Pixel Accuracy, Precision, Recall
    """
    
    def __init__(self, num_classes: int = 2, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """重置所有统计量"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        更新混淆矩阵
        
        Args:
            pred: [B, H, W] 或 [B, num_classes, H, W]
            target: [B, H, W]
        """
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)
        
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        # 过滤忽略索引
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        # 计算混淆矩阵
        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            metrics: 包含各项指标的字典
        """
        cm = self.confusion_matrix
        
        # 每个类别的IoU
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou_per_class = intersection / (union + 1e-10)
        
        # 每个类别的Dice
        dice_per_class = 2 * intersection / (cm.sum(axis=1) + cm.sum(axis=0) + 1e-10)
        
        # 像素准确率
        pixel_acc = np.diag(cm).sum() / (cm.sum() + 1e-10)
        
        # 平均IoU (mIoU)
        miou = np.nanmean(iou_per_class)
        
        # 平均Dice
        mdice = np.nanmean(dice_per_class)
        
        # Precision 和 Recall
        precision = intersection / (cm.sum(axis=0) + 1e-10)
        recall = intersection / (cm.sum(axis=1) + 1e-10)
        
        # F1 Score
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return {
            'mIoU': float(miou),
            'mDice': float(mdice),
            'pixel_acc': float(pixel_acc),
            'mean_precision': float(np.nanmean(precision)),
            'mean_recall': float(np.nanmean(recall)),
            'mean_f1': float(np.nanmean(f1)),
            'iou_per_class': iou_per_class.tolist(),
            'dice_per_class': dice_per_class.tolist()
        }
    
    def get_cloud_iou(self) -> float:
        """获取云类别的IoU（假设类别1是云）"""
        metrics = self.compute()
        iou_per_class = metrics['iou_per_class']
        if len(iou_per_class) > 1:
            return float(iou_per_class[1])  # 云类别通常是1
        return 0.0
    
    def __str__(self):
        metrics = self.compute()
        return (f"mIoU: {metrics['mIoU']:.4f}, "
                f"mDice: {metrics['mDice']:.4f}, "
                f"Pixel Acc: {metrics['pixel_acc']:.4f}")


class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = 2) -> float:
    """
    计算IoU（numpy版本）
    
    Args:
        pred: 预测结果 [H, W]
        target: 目标标注 [H, W]
        num_classes: 类别数
        
    Returns:
        miou: 平均IoU
    """
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection[cls] = np.logical_and(pred_cls, target_cls).sum()
        union[cls] = np.logical_or(pred_cls, target_cls).sum()
    
    iou = intersection / (union + 1e-10)
    return float(np.mean(iou))
