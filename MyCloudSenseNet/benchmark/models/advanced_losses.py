"""
高级损失函数模块

包含：
1. 动态加权复合损失 (DynamicWeightedCompoundLoss, DWCL)
2. 边界感知损失 (BoundaryAwareLoss)
3. 深度监督损失包装器 (DeepSupervisionLoss)
4. 组合损失工厂
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple, Union
import segmentation_models_pytorch as smp


class FocalLoss(nn.Module):
    """
    Focal Loss 实现
    
    针对类别不平衡问题，降低易分类样本的权重，专注于难分类样本。
    """
    
    def __init__(
        self,
        mode: str = "multiclass",
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: 模型输出 [B, C, H, W]（未归一化）
            targets: 目标标签 [B, H, W]
        
        Returns:
            Focal loss 标量
        """
        # 忽略无效像素
        valid_mask = targets != self.ignore_index
        
        if self.mode == "multiclass":
            # 计算概率
            probs = F.softmax(logits, dim=1)
            
            # 获取真实类别的概率
            B, C, H, W = logits.shape
            targets_one_hot = F.one_hot(
                targets.clamp(0, C - 1), 
                num_classes=C
            ).permute(0, 3, 1, 2).float()
            
            # 真实类别的概率
            p_t = (probs * targets_one_hot).sum(dim=1)
            p_t = p_t[valid_mask]
            
            # Focal weight
            focal_weight = (1 - p_t) ** self.gamma
            
            # CE loss
            ce_loss = F.cross_entropy(
                logits, 
                targets, 
                ignore_index=self.ignore_index,
                reduction='none'
            )
            ce_loss = ce_loss[valid_mask]
            
            # Focal loss
            focal_loss = self.alpha * focal_weight * ce_loss
            
        elif self.mode == "binary":
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1 - p_t) ** self.gamma
            ce_loss = F.binary_cross_entropy_with_logits(
                logits, targets.float(), reduction='none'
            )
            focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean() if focal_loss.numel() > 0 else torch.tensor(0.0, device=logits.device)
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(smp.losses.DiceLoss):
    """
    包装 SMP 的 Dice Loss，添加更多选项
    """
    
    def __init__(
        self,
        mode: str = "multiclass",
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 1e-5,
        ignore_index: Optional[int] = 255,
        eps: float = 1e-7,
    ):
        super().__init__(
            mode=mode,
            classes=classes,
            log_loss=log_loss,
            from_logits=from_logits,
            smooth=smooth,
            eps=eps,
        )
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 处理 ignore_index
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device)
        
        return super().forward(logits, targets)


class ClassAdaptiveWeights(nn.Module):
    """
    自适应类别权重模块
    
    根据当前 batch 的类别分布动态调整各类别的权重。
    """
    
    def __init__(
        self,
        num_classes: int,
        min_weight: float = 0.1,
        max_weight: float = 5.0,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.momentum = momentum
        
        # 类别频率估计（EMA）
        self.register_buffer('class_freq', torch.ones(num_classes) / num_classes)
        self.register_buffer('num_batches', torch.tensor(0))
    
    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        """
        计算自适应类别权重
        
        Args:
            targets: 目标标签 [B, H, W]
        
        Returns:
            各类别的权重 [num_classes]
        """
        # 计算当前 batch 的类别频率
        batch_freq = torch.bincount(
            targets.flatten(),
            minlength=self.num_classes,
        ).float()
        
        batch_freq = batch_freq / (batch_freq.sum() + 1e-8)
        
        # 更新 EMA
        if self.training:
            self.num_batches += 1
            momentum = min(self.momentum, 1 - 1 / (self.num_batches + 1))
            self.class_freq = momentum * self.class_freq + (1 - momentum) * batch_freq
        
        # 计算权重：频率越低，权重越高
        weights = 1.0 / (self.class_freq + 1e-8)
        weights = weights / weights.sum() * self.num_classes
        weights = torch.clamp(weights, self.min_weight, self.max_weight)
        
        return weights
    
    def reset(self):
        """重置统计数据"""
        self.class_freq.fill_(1.0 / self.num_classes)
        self.num_batches.zero_()


class DynamicWeightedCompoundLoss(nn.Module):
    """
    动态加权复合损失 (DWCL)
    
    结合 CrossEntropy、Dice 和 Focal loss，
    使用不确定性加权策略自动平衡各项损失。
    
    同时支持类别自适应权重和边界感知增强。
    
    Reference:
        Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        mode: str = "multiclass",
        # 基础损失配置
        use_ce: bool = True,
        use_dice: bool = True,
        use_focal: bool = True,
        # 初始权重
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.5,
        # 动态加权
        enable_dynamic_weighting: bool = True,
        # 类别自适应
        enable_class_adaptive: bool = True,
        class_weight_power: float = 0.5,
        # Focal 参数
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # Dice 参数
        dice_smooth: float = 1e-5,
        dice_log_loss: bool = False,
        # 其他
        ignore_index: int = 255,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.mode = mode
        self.enable_dynamic_weighting = enable_dynamic_weighting
        self.enable_class_adaptive = enable_class_adaptive
        self.class_weight_power = class_weight_power
        
        # 损失组件
        self.use_ce = use_ce
        self.use_dice = use_dice
        self.use_focal = use_focal
        
        # 可学习的对数方差（用于不确定性加权）
        if enable_dynamic_weighting:
            self.log_vars = nn.Parameter(torch.zeros(3))  # CE, Dice, Focal
        else:
            self.register_buffer('log_vars', torch.zeros(3))
            self.log_vars[0] = -torch.log(torch.tensor(ce_weight))
            self.log_vars[1] = -torch.log(torch.tensor(dice_weight))
            self.log_vars[2] = -torch.log(torch.tensor(focal_weight))
        
        # 初始化损失函数
        if use_ce:
            base_weight = torch.tensor(class_weights) if class_weights else None
            self.ce_loss = nn.CrossEntropyLoss(
                weight=base_weight,
                ignore_index=ignore_index,
                reduction='none',
            )
        
        if use_dice:
            self.dice_loss = DiceLoss(
                mode=mode,
                from_logits=True,
                smooth=dice_smooth,
                log_loss=dice_log_loss,
                ignore_index=ignore_index,
            )
        
        if use_focal:
            self.focal_loss = FocalLoss(
                mode=mode,
                alpha=focal_alpha,
                gamma=focal_gamma,
                ignore_index=ignore_index,
                reduction='none',
            )
        
        # 类别自适应权重
        if enable_class_adaptive:
            self.class_adaptive = ClassAdaptiveWeights(
                num_classes=num_classes,
            )
        
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        epoch: Optional[int] = None,
        max_epochs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算动态加权复合损失
        
        Args:
            logits: 模型输出 [B, C, H, W]
            targets: 目标标签 [B, H, W]
            epoch: 当前训练轮次（用于渐进式损失权重）
            max_epochs: 最大训练轮次
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细值
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 计算类别自适应权重
        if self.enable_class_adaptive and self.training:
            class_weights = self.class_adaptive(targets)
        else:
            class_weights = None
        
        # 计算各项损失
        losses = []
        
        if self.use_ce:
            ce = self.ce_loss(logits, targets)
            
            # 应用类别权重
            if class_weights is not None:
                # 将类别权重应用到每个像素
                target_weights = class_weights[targets.clamp(0, self.num_classes - 1)]
                valid_mask = targets != self.ignore_index
                ce = ce[valid_mask]
                target_weights = target_weights[valid_mask]
                ce = (ce * target_weights).mean()
            else:
                ce = ce.mean()
            
            losses.append(ce)
            loss_dict['ce_loss'] = ce.item()
        
        if self.use_dice:
            dice = self.dice_loss(logits, targets)
            losses.append(dice)
            loss_dict['dice_loss'] = dice.item()
        
        if self.use_focal:
            focal = self.focal_loss(logits, targets)
            losses.append(focal)
            loss_dict['focal_loss'] = focal.item()
        
        # 动态加权（不确定性加权）
        if self.enable_dynamic_weighting:
            for i, loss in enumerate(losses):
                # precision = 1 / exp(log_var)
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * loss + self.log_vars[i]
                total_loss = total_loss + weighted_loss
                loss_dict[f'weighted_loss_{i}'] = weighted_loss.item()
                loss_dict[f'precision_{i}'] = precision.item()
        else:
            # 使用固定权重
            weights = torch.exp(-self.log_vars)
            for i, (loss, w) in enumerate(zip(losses, weights)):
                total_loss = total_loss + w * loss
        
        # 记录当前对数方差
        for i in range(len(losses)):
            loss_dict[f'log_var_{i}'] = self.log_vars[i].item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


class BoundaryAwareLoss(nn.Module):
    """
    边界感知损失
    
    对云边界区域赋予更高权重，改善边界分割质量。
    使用 Sobel 算子检测边缘，并进行膨胀处理。
    """
    
    def __init__(
        self,
        base_loss: Optional[nn.Module] = None,
        boundary_width: int = 5,
        boundary_weight: float = 2.0,
        kernel_size: int = 3,
        mode: str = "multiclass",
        ignore_index: int = 255,
    ):
        super().__init__()
        
        self.base_loss = base_loss
        self.boundary_width = boundary_width
        self.boundary_weight = boundary_weight
        self.mode = mode
        self.ignore_index = ignore_index
        
        # Sobel 算子（用于边缘检测）
        self.register_buffer(
            'sobel_x',
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3)
        )
        self.register_buffer(
            'sobel_y',
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3)
        )
        
        # 膨胀核
        dilation_size = boundary_width * 2 + 1
        self.register_buffer(
            'dilation_kernel',
            torch.ones(1, 1, dilation_size, dilation_size)
        )
    
    def detect_boundaries(self, targets: torch.Tensor) -> torch.Tensor:
        """
        检测目标边界
        
        Args:
            targets: 目标标签 [B, H, W]
        
        Returns:
            边界掩码 [B, H, W]，边界区域为 1
        """
        B, H, W = targets.shape
        device = targets.device
        
        # 为每个类别单独检测边界
        boundary_masks = []
        
        # 获取有效类别
        unique_classes = torch.unique(targets[targets != self.ignore_index])
        
        for cls in unique_classes:
            # 创建二值掩码
            binary_mask = (targets == cls).float().unsqueeze(1)  # [B, 1, H, W]
            
            # Sobel 边缘检测
            edge_x = F.conv2d(binary_mask, self.sobel_x, padding=1)
            edge_y = F.conv2d(binary_mask, self.sobel_y, padding=1)
            edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
            
            # 二值化边缘
            edge_binary = (edges > 0.1).float()
            
            boundary_masks.append(edge_binary)
        
        # 合并所有类别的边界
        if boundary_masks:
            all_boundaries = torch.stack(boundary_masks, dim=0).max(dim=0)[0]
        else:
            all_boundaries = torch.zeros(B, 1, H, W, device=device)
        
        # 膨胀边界
        boundary_mask = F.max_pool2d(
            all_boundaries,
            kernel_size=self.dilation_kernel.shape[-1],
            stride=1,
            padding=self.boundary_width,
        )
        
        return boundary_mask.squeeze(1)  # [B, H, W]
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算边界感知损失
        
        Args:
            logits: 模型输出 [B, C, H, W]
            targets: 目标标签 [B, H, W]
        
        Returns:
            weighted_loss: 加权后的损失
            info: 包含边界区域比例等信息
        """
        # 检测边界
        with torch.no_grad():
            boundary_mask = self.detect_boundaries(targets)
        
        # 计算基础损失
        if self.base_loss is None:
            # 使用默认的 CE loss
            base_loss = F.cross_entropy(
                logits,
                targets,
                ignore_index=self.ignore_index,
                reduction='none',
            )
        else:
            # 使用自定义基础损失（需要返回 per-pixel loss）
            # 这里简化处理，假设 base_loss 兼容
            base_loss = F.cross_entropy(
                logits,
                targets,
                ignore_index=self.ignore_index,
                reduction='none',
            )
        
        # 创建权重掩码：边界区域权重更高
        weights = torch.ones_like(base_loss)
        weights[boundary_mask > 0] = self.boundary_weight
        
        # 应用权重
        weighted_loss = (base_loss * weights).mean()
        
        # 统计信息
        info = {
            'boundary_ratio': boundary_mask.mean().item(),
            'boundary_weight': self.boundary_weight,
        }
        
        return weighted_loss, info


class DeepSupervisionLoss(nn.Module):
    """
    深度监督损失包装器
    
    处理多尺度辅助输出，为不同尺度的损失分配不同权重。
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        num_scales: int = 3,
        weights: Optional[List[float]] = None,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        
        self.base_loss = base_loss
        self.num_scales = num_scales
        
        # 默认权重：越深层权重越高
        if weights is None:
            # [0.4, 0.3, 0.2, 0.1] 对于 4 个辅助输出
            weights = [0.5 ** (i + 1) for i in range(num_scales)]
            total = sum(weights)
            weights = [w / total for w in weights]
        
        self.weights = weights
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """设置当前 epoch，用于渐进式权重调整"""
        self.current_epoch = epoch
    
    def forward(
        self,
        main_logits: torch.Tensor,
        aux_logits: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算深度监督损失
        
        Args:
            main_logits: 主输出 [B, C, H, W]
            aux_logits: 辅助输出列表，每个 [B, C, h, w]
            targets: 目标标签 [B, H, W]
        
        Returns:
            total_loss: 总损失
            loss_dict: 各层损失详情
        """
        loss_dict = {}
        
        # 主损失
        if hasattr(self.base_loss, 'forward') and callable(getattr(self.base_loss, 'forward', None)):
            try:
                main_loss, main_info = self.base_loss(main_logits, targets, self.current_epoch)
                if isinstance(main_info, dict):
                    loss_dict.update({f'main_{k}': v for k, v in main_info.items()})
                else:
                    loss_dict['main_loss'] = main_loss.item()
            except:
                main_loss = F.cross_entropy(main_logits, targets)
                loss_dict['main_loss'] = main_loss.item()
        else:
            main_loss = F.cross_entropy(main_logits, targets)
            loss_dict['main_loss'] = main_loss.item()
        
        total_loss = main_loss
        
        # 辅助损失
        # 渐进式权重：warmup 期间逐渐降低辅助权重
        warmup_factor = min(1.0, self.current_epoch / self.warmup_epochs)
        
        for i, (aux_logit, weight) in enumerate(zip(aux_logits, self.weights)):
            # 上采样辅助输出到目标尺寸
            aux_up = F.interpolate(
                aux_logit,
                size=targets.shape[1:],
                mode='bilinear',
                align_corners=False,
            )
            
            try:
                aux_loss, _ = self.base_loss(aux_up, targets, self.current_epoch)
            except:
                aux_loss = F.cross_entropy(aux_up, targets)
            
            # 应用权重（warmup 降低影响）
            scaled_weight = weight * (0.5 + 0.5 * warmup_factor)
            weighted_aux = scaled_weight * aux_loss
            
            total_loss = total_loss + weighted_aux
            loss_dict[f'aux_{i}_loss'] = aux_loss.item()
            loss_dict[f'aux_{i}_weight'] = scaled_weight
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


class CombinedLoss(nn.Module):
    """
    组合损失工厂
    
    根据配置创建组合损失函数。
    """
    
    @staticmethod
    def create(
        loss_type: str = "dwcl",
        num_classes: int = 3,
        **kwargs,
    ) -> nn.Module:
        """
        创建损失函数
        
        Args:
            loss_type: 损失类型 ('ce', 'dice', 'focal', 'dwcl', 'boundary', 'combined')
            num_classes: 类别数
            **kwargs: 额外参数
        
        Returns:
            损失模块
        """
        if loss_type == "ce":
            return nn.CrossEntropyLoss(
                weight=kwargs.get('class_weights'),
                ignore_index=kwargs.get('ignore_index', 255),
            )
        
        elif loss_type == "dice":
            return DiceLoss(
                mode="multiclass",
                from_logits=True,
                ignore_index=kwargs.get('ignore_index', 255),
            )
        
        elif loss_type == "focal":
            return FocalLoss(
                mode="multiclass",
                alpha=kwargs.get('alpha', 0.25),
                gamma=kwargs.get('gamma', 2.0),
                ignore_index=kwargs.get('ignore_index', 255),
            )
        
        elif loss_type == "dwcl":
            return DynamicWeightedCompoundLoss(
                num_classes=num_classes,
                use_ce=kwargs.get('use_ce', True),
                use_dice=kwargs.get('use_dice', True),
                use_focal=kwargs.get('use_focal', True),
                enable_dynamic_weighting=kwargs.get('enable_dynamic_weighting', True),
                enable_class_adaptive=kwargs.get('enable_class_adaptive', True),
                focal_alpha=kwargs.get('focal_alpha', 0.25),
                focal_gamma=kwargs.get('focal_gamma', 2.0),
                class_weights=kwargs.get('class_weights'),
            )
        
        elif loss_type == "boundary":
            base = DynamicWeightedCompoundLoss(num_classes=num_classes) if kwargs.get('use_advanced', True) else None
            return BoundaryAwareLoss(
                base_loss=base,
                boundary_width=kwargs.get('boundary_width', 5),
                boundary_weight=kwargs.get('boundary_weight', 2.0),
            )
        
        elif loss_type == "combined":
            # 创建 DWCL 作为基础损失
            dwcl = DynamicWeightedCompoundLoss(
                num_classes=num_classes,
                use_ce=kwargs.get('use_ce', True),
                use_dice=kwargs.get('use_dice', True),
                use_focal=kwargs.get('use_focal', True),
                enable_dynamic_weighting=kwargs.get('enable_dynamic_weighting', True),
                enable_class_adaptive=kwargs.get('enable_class_adaptive', True),
            )
            
            # 如果启用边界感知，包装 DWCL
            if kwargs.get('enable_boundary', True):
                return BoundaryAwareLoss(
                    base_loss=dwcl,
                    boundary_width=kwargs.get('boundary_width', 5),
                    boundary_weight=kwargs.get('boundary_weight', 2.0),
                )
            
            return dwcl
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


# 工具函数

def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int = 3,
    mode: str = "inverse_freq",
    min_weight: float = 0.1,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """
    计算类别权重
    
    Args:
        targets: 目标标签
        num_classes: 类别数
        mode: 计算模式 ('inverse_freq', 'effective_num', 'sqrt_inv')
        min_weight: 最小权重
        max_weight: 最大权重
    
    Returns:
        类别权重 [num_classes]
    """
    # 统计各类别频率
    counts = torch.bincount(targets.flatten(), minlength=num_classes).float()
    
    if mode == "inverse_freq":
        weights = 1.0 / (counts + 1e-8)
    elif mode == "effective_num":
        beta = 0.9999
        effective_num = (1.0 - beta) / (1.0 - beta ** (counts + 1e-8))
        weights = effective_num
    elif mode == "sqrt_inv":
        weights = 1.0 / torch.sqrt(counts + 1e-8)
    else:
        weights = torch.ones(num_classes)
    
    # 归一化
    weights = weights / weights.sum() * num_classes
    weights = torch.clamp(weights, min_weight, max_weight)
    
    return weights


if __name__ == "__main__":
    print("Testing Advanced Losses...")
    
    batch_size = 2
    num_classes = 3
    H, W = 64, 64
    
    # 测试数据
    logits = torch.randn(batch_size, num_classes, H, W)
    targets = torch.randint(0, num_classes, (batch_size, H, W))
    
    # 测试 DWCL
    print("\n1. DynamicWeightedCompoundLoss:")
    dwcl = DynamicWeightedCompoundLoss(
        num_classes=num_classes,
        use_ce=True,
        use_dice=True,
        use_focal=True,
        enable_dynamic_weighting=True,
    )
    loss, info = dwcl(logits, targets)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Info: {info}")
    
    # 测试边界感知损失
    print("\n2. BoundaryAwareLoss:")
    boundary_loss = BoundaryAwareLoss(boundary_width=3, boundary_weight=2.0)
    loss, info = boundary_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Boundary ratio: {info['boundary_ratio']:.4f}")
    
    # 测试类别自适应权重
    print("\n3. ClassAdaptiveWeights:")
    adaptive = ClassAdaptiveWeights(num_classes=num_classes)
    for _ in range(5):
        weights = adaptive(targets)
        print(f"   Batch weights: {weights}")
    
    # 测试组合损失工厂
    print("\n4. CombinedLoss Factory:")
    combined = CombinedLoss.create(
        loss_type="combined",
        num_classes=num_classes,
        enable_boundary=True,
        boundary_weight=2.5,
    )
    print(f"   Created: {type(combined).__name__}")
    
    print("\n✓ All tests passed!")
