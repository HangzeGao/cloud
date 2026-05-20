"""
损失函数模块 - 支持多种损失函数组合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    组合损失函数
    支持多种损失函数的组合，并可通过配置调整权重
    
    支持的损失类型：
    - dice: Dice Loss（处理类别不平衡）
    - bce: 二元交叉熵
    - focal: Focal Loss（聚焦难分样本）
    - boundary: Boundary Loss（边缘精细化）
    - tversky: Tversky Loss（控制precision/recall平衡）
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.loss_types = config.get('types', ['dice', 'bce'])
        self.weights = config.get('weights', [0.5, 0.5])
        
        assert len(self.loss_types) == len(self.weights), \
            "Number of loss types must match number of weights"
        
        self.loss_modules = nn.ModuleDict()
        
        for loss_type in self.loss_types:
            if loss_type == 'dice':
                self.loss_modules[loss_type] = DiceLoss()
            elif loss_type == 'bce':
                self.loss_modules[loss_type] = BCEWithLogitsLoss()
            elif loss_type == 'focal':
                self.loss_modules[loss_type] = FocalLoss(
                    alpha=config.get('focal_alpha', 0.25),
                    gamma=config.get('focal_gamma', 2.0)
                )
            elif loss_type == 'boundary':
                self.loss_modules[loss_type] = BoundaryLoss()
            elif loss_type == 'tversky':
                self.loss_modules[loss_type] = TverskyLoss(
                    alpha=config.get('tversky_alpha', 0.3),
                    beta=config.get('tversky_beta', 0.7)
                )
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, num_classes, H, W] 预测logits
            target: [B, H, W] 或 [B, num_classes, H, W] 目标标注
        Returns:
            loss: 标量损失值
        """
        total_loss = 0
        
        for loss_type, weight in zip(self.loss_types, self.weights):
            loss_value = self.loss_modules[loss_type](pred, target)
            total_loss += weight * loss_value
        
        return total_loss


class DiceLoss(nn.Module):
    """Dice Loss - 特别适合类别不平衡的分割任务"""
    
    def __init__(self, smooth: float = 1e-5, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, num_classes, H, W] logits
            target: [B, H, W]
        """
        # 应用sigmoid获取概率
        pred = torch.sigmoid(pred)
        
        # 将target转换为one-hot
        num_classes = pred.shape[1]
        if target.dim() == 3:
            target_onehot = F.one_hot(target.long(), num_classes=num_classes)
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        else:
            target_onehot = target
        
        # 忽略特定类别
        mask = (target != self.ignore_index).float().unsqueeze(1)
        
        # 展平
        pred_flat = pred * mask
        target_flat = target_onehot * mask
        
        # 计算Dice系数
        intersection = (pred_flat * target_flat).sum(dim=(2, 3))
        union = pred_flat.sum(dim=(2, 3)) + target_flat.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class BCEWithLogitsLoss(nn.Module):
    """带logits的BCE损失"""
    
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, num_classes, H, W] logits
            target: [B, H, W]
        """
        num_classes = pred.shape[1]
        if target.dim() == 3:
            target_onehot = F.one_hot(target.long(), num_classes=num_classes)
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        else:
            target_onehot = target
        
        # 计算损失
        loss = self.bce(pred, target_onehot)
        
        # 忽略mask
        mask = (target != self.ignore_index).float().unsqueeze(1)
        loss = loss * mask
        
        return loss.sum() / (mask.sum() + 1e-8)


class FocalLoss(nn.Module):
    """
    Focal Loss - 让模型更关注难分样本
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, num_classes, H, W] logits
            target: [B, H, W]
        """
        # 应用sigmoid
        pred_prob = torch.sigmoid(pred)
        
        num_classes = pred.shape[1]
        if target.dim() == 3:
            target_onehot = F.one_hot(target.long(), num_classes=num_classes)
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        else:
            target_onehot = target
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target_onehot, reduction='none')
        
        # Focal weight
        p_t = (pred_prob * target_onehot) + ((1 - pred_prob) * (1 - target_onehot))
        focal_weight = (1 - p_t) ** self.gamma
        
        # 应用权重
        focal_loss = self.alpha * focal_weight * bce
        
        # 忽略mask
        mask = (target != self.ignore_index).float().unsqueeze(1)
        focal_loss = focal_loss * mask
        
        return focal_loss.sum() / (mask.sum() + 1e-8)


class BoundaryLoss(nn.Module):
    """
    Boundary Loss - 优化分割边缘
    基于到边界的距离变换
    """
    
    def __init__(self, theta0: int = 3, theta: int = 5, ignore_index: int = 255):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, num_classes, H, W]
            target: [B, H, W]
        """
        # 简化的边界损失（基于梯度）
        pred_prob = torch.sigmoid(pred)
        
        # Sobel算子计算边缘
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        
        # 对每个类别计算边界
        boundary_loss = 0
        num_classes = pred.shape[1]
        
        for c in range(num_classes):
            pred_c = pred_prob[:, c:c+1]
            
            # 计算预测的边缘
            grad_x = F.conv2d(pred_c, sobel_x, padding=1)
            grad_y = F.conv2d(pred_c, sobel_y, padding=1)
            pred_edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
            
            # 计算目标的边缘（简化版）
            target_c = (target == c).float().unsqueeze(1)
            target_grad_x = F.conv2d(target_c, sobel_x, padding=1)
            target_grad_y = F.conv2d(target_c, sobel_y, padding=1)
            target_edge = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-8)
            
            # MSE损失
            boundary_loss += F.mse_loss(pred_edge, target_edge)
        
        return boundary_loss / num_classes


class TverskyLoss(nn.Module):
    """
    Tversky Loss - 可调整的precision和recall平衡
    当alpha < beta时，更注重recall（减少漏检）
    当alpha > beta时，更注重precision（减少误检）
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5, ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, num_classes, H, W]
            target: [B, H, W]
        """
        pred = torch.sigmoid(pred)
        
        num_classes = pred.shape[1]
        if target.dim() == 3:
            target_onehot = F.one_hot(target.long(), num_classes=num_classes)
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        else:
            target_onehot = target
        
        # 忽略mask
        mask = (target != self.ignore_index).float().unsqueeze(1)
        pred = pred * mask
        target_onehot = target_onehot * mask
        
        # 计算TP, FP, FN
        tp = (pred * target_onehot).sum(dim=(2, 3))
        fp = (pred * (1 - target_onehot)).sum(dim=(2, 3))
        fn = ((1 - pred) * target_onehot).sum(dim=(2, 3))
        
        # Tversky指数
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky.mean()
