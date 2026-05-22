"""
分割评估指标
============

提供语义分割任务的各种评估指标计算，包括IoU、Dice、像素准确率等。

主要功能:
- 混淆矩阵计算和更新
- 多类别IoU和Dice系数
- 像素级准确率
- 精确率和召回率
- 支持GPU张量和NumPy数组

使用示例:
    >>> from utils.metrics import SegmentationMetrics, AverageMeter
    >>> metrics = SegmentationMetrics(num_classes=2)
    >>> for pred, target in dataloader:
    ...     metrics.update(pred, target)
    >>> results = metrics.compute()
    >>> print(f"mIoU: {results['mIoU']:.4f}")
"""
import torch
import numpy as np
from typing import Dict, List, Union, Optional
from dataclasses import dataclass


@dataclass
class SegmentationMetricsResult:
    """分割指标结果数据类"""
    mIoU: float
    mDice: float
    pixel_acc: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    iou_per_class: List[float]
    dice_per_class: List[float]
    confusion_matrix: Optional[np.ndarray] = None


class SegmentationMetrics:
    """
    语义分割评估指标
    
    基于混淆矩阵计算各项评估指标，支持多类别分割任务。
    
    Attributes:
        num_classes: 类别数量
        ignore_index: 忽略的类别索引
        confusion_matrix: 混淆矩阵 [num_classes, num_classes]
    
    Example:
        >>> metrics = SegmentationMetrics(num_classes=2, ignore_index=255)
        >>> metrics.update(pred_batch, target_batch)
        >>> results = metrics.compute()
        >>> print(results.mIoU)
    """
    
    def __init__(self, num_classes: int = 2, ignore_index: int = 255):
        """
        初始化评估指标
        
        Args:
            num_classes: 类别数量（通常是2：背景和云）
            ignore_index: 忽略的像素标签（通常是255表示无效区域）
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self) -> None:
        """重置所有统计量，清空混淆矩阵"""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64
        )
    
    def update(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray]
    ) -> None:
        """
        更新混淆矩阵
        
        Args:
            pred: 预测结果 [B, H, W] 或 [B, num_classes, H, W]
                  如果是多通道，会自动取argmax
            target: 目标标注 [B, H, W]
        """
        # 转换为numpy并处理维度
        if isinstance(pred, torch.Tensor):
            if pred.dim() == 4:
                pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy()
        
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        # 展平为一维数组
        pred = pred.flatten()
        target = target.flatten()
        
        # 过滤忽略索引
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        # 更新混淆矩阵
        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1
    
    def compute(self) -> Dict[str, Union[float, List[float]]]:
        """
        计算所有评估指标
        
        Returns:
            包含以下指标的字典:
            - mIoU: 平均交并比
            - mDice: 平均Dice系数
            - pixel_acc: 像素准确率
            - mean_precision: 平均精确率
            - mean_recall: 平均召回率
            - mean_f1: 平均F1分数
            - iou_per_class: 每个类别的IoU
            - dice_per_class: 每个类别的Dice系数
        """
        cm = self.confusion_matrix
        
        # 计算每个类别的IoU
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou_per_class = intersection / (union + 1e-10)
        
        # 计算每个类别的Dice
        dice_per_class = 2 * intersection / (cm.sum(axis=1) + cm.sum(axis=0) + 1e-10)
        
        # 像素准确率
        pixel_acc = np.diag(cm).sum() / (cm.sum() + 1e-10)
        
        # 平均IoU (mIoU)
        miou = float(np.nanmean(iou_per_class))
        
        # 平均Dice
        mdice = float(np.nanmean(dice_per_class))
        
        # Precision 和 Recall
        precision = intersection / (cm.sum(axis=0) + 1e-10)
        recall = intersection / (cm.sum(axis=1) + 1e-10)
        
        # F1 Score
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return {
            'mIoU': miou,
            'mDice': mdice,
            'pixel_acc': float(pixel_acc),
            'mean_precision': float(np.nanmean(precision)),
            'mean_recall': float(np.nanmean(recall)),
            'mean_f1': float(np.nanmean(f1)),
            'iou_per_class': iou_per_class.tolist(),
            'dice_per_class': dice_per_class.tolist()
        }
    
    def compute_cloud_iou(self) -> float:
        """
        获取云类别的IoU（假设类别1是云）
        
        Returns:
            float: 云类别的IoU值
        """
        metrics = self.compute()
        iou_per_class = metrics['iou_per_class']
        if len(iou_per_class) > 1:
            return float(iou_per_class[1])
        return 0.0
    
    def compute_background_iou(self) -> float:
        """
        获取背景类别的IoU（假设类别0是背景）
        
        Returns:
            float: 背景类别的IoU值
        """
        metrics = self.compute()
        iou_per_class = metrics['iou_per_class']
        if len(iou_per_class) > 0:
            return float(iou_per_class[0])
        return 0.0
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        获取混淆矩阵
        
        Returns:
            np.ndarray: 混淆矩阵 [num_classes, num_classes]
        """
        return self.confusion_matrix.copy()
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        获取每个类别的像素分布
        
        Returns:
            Dict[int, int]: 类别到像素数的映射
        """
        return {
            i: int(self.confusion_matrix[i].sum())
            for i in range(self.num_classes)
        }
    
    def __str__(self) -> str:
        """字符串表示，显示主要指标"""
        metrics = self.compute()
        return (
            f"mIoU: {metrics['mIoU']:.4f}, "
            f"mDice: {metrics['mDice']:.4f}, "
            f"Pixel Acc: {metrics['pixel_acc']:.4f}"
        )
    
    def __repr__(self) -> str:
        """详细表示"""
        return (
            f"SegmentationMetrics(num_classes={self.num_classes}, "
            f"ignore_index={self.ignore_index}, "
            f"mIoU={self.compute()['mIoU']:.4f})"
        )


class AverageMeter:
    """
    计算并存储平均值和当前值
    
    用于训练过程中跟踪损失等指标。
    
    Attributes:
        val: 当前值
        avg: 平均值
        sum: 累加和
        count: 计数
    
    Example:
        >>> meter = AverageMeter()
        >>> for loss in losses:
        ...     meter.update(loss)
        >>> print(f"Average Loss: {meter.avg:.4f}")
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """重置所有统计量"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        更新统计量
        
        Args:
            val: 当前值
            n: 样本数量（用于批量更新）
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"val={self.val:.4f}, avg={self.avg:.4f}"
    
    def __repr__(self) -> str:
        """详细表示"""
        return f"AverageMeter(val={self.val:.4f}, avg={self.avg:.4f}, count={self.count})"


# ==================== 便捷函数 ====================

def compute_iou_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 2,
    ignore_index: int = 255
) -> float:
    """
    计算IoU（NumPy版本）
    
    Args:
        pred: 预测结果 [H, W]
        target: 目标标注 [H, W]
        num_classes: 类别数
        ignore_index: 忽略的索引
        
    Returns:
        float: 平均IoU
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


def compute_dice_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 2
) -> float:
    """
    计算Dice系数（NumPy版本）
    
    Args:
        pred: 预测结果 [H, W]
        target: 目标标注 [H, W]
        num_classes: 类别数
        
    Returns:
        float: 平均Dice系数
    """
    dice_scores = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        
        if union > 0:
            dice = 2 * intersection / union
            dice_scores.append(dice)
    
    return float(np.mean(dice_scores)) if dice_scores else 0.0


def compute_pixel_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    ignore_index: int = 255
) -> float:
    """
    计算像素准确率
    
    Args:
        pred: 预测结果 [H, W]
        target: 目标标注 [H, W]
        ignore_index: 忽略的像素值
        
    Returns:
        float: 像素准确率
    """
    mask = (target != ignore_index)
    correct = (pred[mask] == target[mask]).sum()
    total = mask.sum()
    return float(correct / (total + 1e-10))


def compute_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 2
) -> np.ndarray:
    """
    计算混淆矩阵
    
    Args:
        pred: 预测结果 [H, W]
        target: 目标标注 [H, W]
        num_classes: 类别数
        
    Returns:
        np.ndarray: 混淆矩阵 [num_classes, num_classes]
    """
    pred = pred.flatten()
    target = target.flatten()
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(pred, target):
        if 0 <= p < num_classes and 0 <= t < num_classes:
            cm[t, p] += 1
    
    return cm


class MetricsTracker:
    """
    多指标追踪器
    
    同时追踪多个评估指标，适用于训练过程中的综合评估。
    
    Example:
        >>> tracker = MetricsTracker(['loss', 'mIoU', 'accuracy'])
        >>> tracker.update('loss', 0.5)
        >>> tracker.update('mIoU', 0.75)
        >>> print(tracker.get_summary())
    """
    
    def __init__(self, metric_names: Optional[List[str]] = None):
        """
        Args:
            metric_names: 要追踪的指标名称列表
        """
        self.meters: Dict[str, AverageMeter] = {}
        if metric_names:
            for name in metric_names:
                self.add_metric(name)
    
    def add_metric(self, name: str) -> None:
        """添加新指标"""
        self.meters[name] = AverageMeter()
    
    def update(self, name: str, value: float, n: int = 1) -> None:
        """更新指定指标"""
        if name not in self.meters:
            self.add_metric(name)
        self.meters[name].update(value, n)
    
    def get(self, name: str) -> Optional[AverageMeter]:
        """获取指定指标的计量器"""
        return self.meters.get(name)
    
    def get_avg(self, name: str) -> float:
        """获取指定指标的平均值"""
        meter = self.meters.get(name)
        return meter.avg if meter else 0.0
    
    def get_all_avg(self) -> Dict[str, float]:
        """获取所有指标的平均值"""
        return {name: meter.avg for name, meter in self.meters.items()}
    
    def reset(self) -> None:
        """重置所有指标"""
        for meter in self.meters.values():
            meter.reset()
    
    def get_summary(self) -> str:
        """获取摘要字符串"""
        return ', '.join(
            f"{name}: {meter.avg:.4f}"
            for name, meter in self.meters.items()
        )
    
    def __str__(self) -> str:
        return self.get_summary()
