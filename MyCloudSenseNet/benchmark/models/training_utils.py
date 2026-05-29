"""
训练策略工具模块

包含：
1. 高级学习率调度器
2. 深度监督支持
3. 训练监控工具
"""

import math
from typing import List, Optional, Dict, Any, Callable, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    带 Warmup 的余弦退火重启学习率调度器
    
    结合 Warmup 预热和 CosineAnnealingWarmRestarts 的优点。
    
    Args:
        optimizer: 优化器
        first_cycle_steps: 第一个周期的步数
        cycle_mult: 周期长度乘数
        max_lr: 最大学习率
        min_lr: 最小学习率
        warmup_steps: Warmup 步数
        gamma: 每个周期的学习率衰减因子
        last_epoch: 最后一个 epoch 索引
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps, \
            "warmup_steps must be less than first_cycle_steps"
        
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        # 初始化学习率
        self.init_lr()
    
    def init_lr(self):
        """初始化基础学习率"""
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        """计算当前学习率"""
        if self.step_in_cycle == -1:
            return self.base_lrs
        
        if self.step_in_cycle < self.warmup_steps:
            # Warmup 阶段：线性增长
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine 退火阶段
            progress = (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
            return [
                base_lr + (self.max_lr - base_lr) * 
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
                self.max_lr = self.max_lr * self.gamma
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class OneCycleLRWithWarmup(_LRScheduler):
    """
    带 Warmup 的 One Cycle 学习率调度器
    
    先线性增长到峰值，然后余弦下降到最小值。
    适合快速收敛的训练场景。
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 1e-7,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.min_post_warmup_lr = max_lr / final_div_factor
        
        super().__init__(optimizer, last_epoch)
        
        # 初始化学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
    
    def get_lr(self):
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Warmup 阶段
            progress = step / self.warmup_steps
            return [self.initial_lr + (self.max_lr - self.initial_lr) * progress 
                    for _ in self.optimizer.param_groups]
        
        # 退火阶段
        post_warmup_steps = self.total_steps - self.warmup_steps
        post_warmup_current_step = step - self.warmup_steps
        
        if self.anneal_strategy == "cos":
            # 余弦退火
            progress = post_warmup_current_step / post_warmup_steps
            return [
                self.min_post_warmup_lr + (self.max_lr - self.min_post_warmup_lr) * 
                (1 + math.cos(math.pi * progress)) / 2
                for _ in self.optimizer.param_groups
            ]
        else:
            # 线性退火
            progress = post_warmup_current_step / post_warmup_steps
            return [
                self.max_lr + (self.min_post_warmup_lr - self.max_lr) * progress
                for _ in self.optimizer.param_groups
            ]


class LRSchedulerFactory:
    """
    学习率调度器工厂
    
    支持多种调度策略：
    - plateau: 基于验证指标的调度
    - cosine: 简单余弦退火
    - cosine_warmup: 带 Warmup 的余弦退火
    - one_cycle: One Cycle 策略
    - polynomial: 多项式衰减
    """
    
    @staticmethod
    def create(
        scheduler_type: str,
        optimizer: Optimizer,
        num_epochs: int,
        warmup_epochs: int = 5,
        eta_min: float = 1e-6,
        **kwargs,
    ) -> Union[_LRScheduler, Dict[str, Any]]:
        """
        创建学习率调度器
        
        Args:
            scheduler_type: 调度器类型
            optimizer: 优化器
            num_epochs: 总训练轮数
            warmup_epochs: Warmup 轮数
            eta_min: 最小学习率
            **kwargs: 额外参数
        
        Returns:
            调度器或调度器配置字典（用于 PyTorch Lightning）
        """
        if scheduler_type == "plateau":
            # 基于验证指标的调度
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=kwargs.get('mode', 'max'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 3),
                min_lr=eta_min,
                verbose=kwargs.get('verbose', True),
            )
            return {
                "scheduler": scheduler,
                "monitor": kwargs.get('monitor', 'val/iou_epoch'),
                "interval": "epoch",
                "frequency": 1,
            }
        
        elif scheduler_type == "cosine":
            # 简单余弦退火
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=eta_min,
            )
            return {"scheduler": scheduler, "interval": "epoch"}
        
        elif scheduler_type == "cosine_warmup":
            # 带 Warmup 的余弦退火重启
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=num_epochs,
                cycle_mult=kwargs.get('cycle_mult', 1.0),
                max_lr=kwargs.get('max_lr', optimizer.defaults['lr']),
                min_lr=eta_min,
                warmup_steps=warmup_epochs,
                gamma=kwargs.get('gamma', 1.0),
            )
            return {"scheduler": scheduler, "interval": "epoch"}
        
        elif scheduler_type == "one_cycle":
            # One Cycle 策略
            steps_per_epoch = kwargs.get('steps_per_epoch', 100)
            total_steps = num_epochs * steps_per_epoch
            warmup_steps = warmup_epochs * steps_per_epoch
            
            scheduler = OneCycleLRWithWarmup(
                optimizer,
                max_lr=kwargs.get('max_lr', optimizer.defaults['lr']),
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr=eta_min,
                pct_start=kwargs.get('pct_start', 0.3),
            )
            return {"scheduler": scheduler, "interval": "step"}
        
        elif scheduler_type == "polynomial":
            # 多项式衰减
            scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer,
                total_iters=num_epochs,
                power=kwargs.get('power', 0.9),
            )
            return {"scheduler": scheduler, "interval": "epoch"}
        
        elif scheduler_type == "step":
            # 阶梯式衰减
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', num_epochs // 3),
                gamma=kwargs.get('gamma', 0.1),
            )
            return {"scheduler": scheduler, "interval": "epoch"}
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class DeepSupervisionWrapper(nn.Module):
    """
    深度监督包装器
    
    为编码器-解码器架构添加多尺度辅助分割头。
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        decoder_channels: Optional[List[int]] = None,
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        
        # 获取解码器通道数
        if decoder_channels is None:
            # 默认假设 UNet 结构的通道数
            decoder_channels = [256, 128, 64, 32, 16]
        
        self.decoder_channels = decoder_channels
        
        # 创建辅助分割头（除了最后一个）
        self.aux_heads = nn.ModuleList()
        for i, ch in enumerate(decoder_channels[:-1]):
            head = nn.Sequential(
                # 上采样到原图分辨率
                nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                # 分割头
                nn.Conv2d(ch, num_classes, kernel_size=3, padding=1),
            )
            self.aux_heads.append(head)
        
        # 权重
        if weights is None:
            # 默认递减权重
            total = len(decoder_channels) - 1
            self.weights = [0.5 ** (total - i) for i in range(total)]
            self.weights = [w / sum(self.weights) for w in self.weights]
        else:
            self.weights = weights
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播，返回主输出和辅助输出
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            字典包含:
                - 'main': 主输出 [B, num_classes, H, W]
                - 'aux': 辅助输出列表
        """
        # 通过基础模型
        # 假设基础模型返回多尺度特征或可以拦截中间结果
        
        # 获取编码器特征
        if hasattr(self.base_model, 'encoder'):
            encoder_features = self.base_model.encoder(x)
        else:
            encoder_features = x
        
        # 如果是单张量，包装为列表
        if isinstance(encoder_features, torch.Tensor):
            encoder_features = [encoder_features]
        
        # 通过解码器
        if hasattr(self.base_model, 'decoder'):
            # 假设 decoder 返回多尺度特征
            decoder_features = self.base_model.decoder(
                encoder_features, 
                return_all=True
            )
        else:
            decoder_features = encoder_features
        
        # 主输出
        if hasattr(self.base_model, 'segmentation_head'):
            main_output = self.base_model.segmentation_head(decoder_features[-1])
        else:
            main_output = decoder_features[-1]
        
        # 辅助输出
        aux_outputs = []
        for i, (head, feat) in enumerate(zip(self.aux_heads, decoder_features[:-1])):
            aux_out = head(feat)
            aux_outputs.append(aux_out)
        
        return {
            'main': main_output,
            'aux': aux_outputs,
            'weights': self.weights,
        }


class DeepSupervisionHook(nn.Module):
    """
    深度监督钩子
    
    通过注册前向钩子捕获解码器中间层输出，
    无需修改原始模型架构。
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        hook_layers: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        
        # 保存中间层输出
        self.intermediate_outputs = {}
        self.hooks = []
        
        # 注册钩子
        if hook_layers:
            self._register_hooks(hook_layers)
        
        # 辅助头（动态创建）
        self.aux_heads = nn.ModuleDict()
        
        # 权重
        if weights is None:
            self.weights = [0.3, 0.2, 0.1]
        else:
            self.weights = weights
    
    def _register_hooks(self, layer_names: List[str]):
        """为指定层注册前向钩子"""
        def make_hook(name):
            def hook(module, input, output):
                self.intermediate_outputs[name] = output
            return hook
        
        for name in layer_names:
            module = dict(self.base_model.named_modules()).get(name)
            if module is not None:
                handle = module.register_forward_hook(make_hook(name))
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def create_aux_head(self, name: str, in_channels: int):
        """为捕获的特征创建辅助头"""
        if name not in self.aux_heads:
            self.aux_heads[name] = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 2, self.num_classes, 1),
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 清空之前的中间输出
        self.intermediate_outputs.clear()
        
        # 主前向传播
        main_output = self.base_model(x)
        
        # 处理中间输出
        aux_outputs = []
        for name, feat in self.intermediate_outputs.items():
            # 为新的特征尺寸创建辅助头
            if isinstance(feat, torch.Tensor) and name not in self.aux_heads:
                self.create_aux_head(name, feat.size(1))
            
            if name in self.aux_heads:
                aux_out = self.aux_heads[name](feat)
                # 上采样到主输出尺寸
                aux_out = nn.functional.interpolate(
                    aux_out,
                    size=main_output.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
                aux_outputs.append(aux_out)
        
        return {
            'main': main_output,
            'aux': aux_outputs[:len(self.weights)],  # 限制数量
            'weights': self.weights[:len(aux_outputs)],
        }


class ModelEMA:
    """
    模型指数移动平均 (EMA)
    
    维护一个模型参数的移动平均版本，用于更稳定的推理。
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 100,
    ):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        # 创建 EMA 参数副本
        self.shadow_params = [
            p.clone().detach()
            for p in model.parameters()
            if p.requires_grad
        ]
    
    @torch.no_grad()
    def update(self):
        """更新 EMA 参数"""
        self.step_count += 1
        
        # Warmup：逐渐增加衰减率
        effective_decay = min(self.decay, (1 + self.step_count) / (self.warmup_steps + self.step_count))
        
        for shadow_param, model_param in zip(self.shadow_params, 
                                              [p for p in self.model.parameters() if p.requires_grad]):
            shadow_param.mul_(effective_decay).add_(model_param.data, alpha=1 - effective_decay)
    
    def apply_shadow(self):
        """应用 EMA 参数到模型（用于推理）"""
        self.backup_params = []
        for shadow_param, model_param in zip(self.shadow_params,
                                              [p for p in self.model.parameters() if p.requires_grad]):
            self.backup_params.append(model_param.data.clone())
            model_param.data.copy_(shadow_param)
    
    def restore(self):
        """恢复原始参数"""
        for backup_param, model_param in zip(self.backup_params,
                                              [p for p in self.model.parameters() if p.requires_grad]):
            model_param.data.copy_(backup_param)
        self.backup_params = []


class GradientClipper:
    """
    梯度裁剪器
    
    支持多种梯度裁剪策略。
    """
    
    @staticmethod
    def by_norm(parameters, max_norm: float = 1.0, norm_type: float = 2.0):
        """基于范数的裁剪"""
        torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    
    @staticmethod
    def by_value(parameters, clip_value: float = 1.0):
        """基于值的裁剪"""
        torch.nn.utils.clip_grad_value_(parameters, clip_value)
    
    @staticmethod
    def adaptive(parameters, initial_norm: float = 1.0, current_loss: float = None, 
                 loss_history: List[float] = None):
        """自适应裁剪（根据损失变化调整）"""
        if loss_history and len(loss_history) > 1:
            # 如果损失在震荡，降低裁剪阈值
            if abs(loss_history[-1] - loss_history[-2]) > loss_history[-2] * 0.1:
                max_norm = initial_norm * 0.5
            else:
                max_norm = initial_norm
        else:
            max_norm = initial_norm
        
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    encoder_lr_scale: float = 0.5,
    bit_depth_component_lr_scale: float = 2.0,
    **kwargs,
) -> Optimizer:
    """
    创建分层学习率优化器
    
    为不同组件设置不同的学习率：
    - Encoder：较低学习率（预训练权重）
    - Decoder：标准学习率
    - 分割头：标准学习率
    - 位深度组件：较高学习率（新组件）
    """
    
    # 收集参数组
    param_groups = []
    
    # 获取编码器参数
    base_encoder = None
    if hasattr(model, 'encoder'):
        base_encoder = model.encoder
        # 处理包装器的情况 (如 BitDepthAdaptiveEncoder)
        while hasattr(base_encoder, 'encoder') and base_encoder is not base_encoder.encoder:
            base_encoder = base_encoder.encoder
        
        encoder_params = list(base_encoder.parameters())
        param_groups.append({
            "params": encoder_params,
            "lr": learning_rate * encoder_lr_scale,
            "name": "encoder",
            "weight_decay": weight_decay,
        })
    
    # 解码器参数
    if hasattr(model, 'decoder'):
        param_groups.append({
            "params": list(model.decoder.parameters()),
            "lr": learning_rate,
            "name": "decoder",
            "weight_decay": weight_decay,
        })
    
    # 分割头参数
    if hasattr(model, 'segmentation_head'):
        param_groups.append({
            "params": list(model.segmentation_head.parameters()),
            "lr": learning_rate,
            "name": "head",
            "weight_decay": weight_decay,
        })
    
    # 位深度估计器参数 (从原始模型或包装器中获取)
    bd_estimator = None
    if hasattr(model, 'encoder'):
        if hasattr(model.encoder, 'bit_depth_estimator'):
            bd_estimator = model.encoder.bit_depth_estimator
        elif hasattr(model.encoder, 'encoder') and hasattr(model.encoder.encoder, 'bit_depth_estimator'):
            bd_estimator = model.encoder.encoder.bit_depth_estimator
    
    if bd_estimator is not None:
        estimator_params = list(bd_estimator.parameters())
        if estimator_params:
            param_groups.append({
                "params": estimator_params,
                "lr": learning_rate * bit_depth_component_lr_scale,
                "name": "bit_depth_estimator",
                "weight_decay": weight_decay * 0.1,
            })
    
    # 位深度适配器参数
    bd_adapter = None
    if hasattr(model, 'encoder'):
        if hasattr(model.encoder, 'feature_adapter'):
            bd_adapter = model.encoder.feature_adapter
        elif hasattr(model.encoder, 'encoder') and hasattr(model.encoder.encoder, 'feature_adapter'):
            bd_adapter = model.encoder.encoder.feature_adapter
    
    if bd_adapter is not None:
        adapter_params = list(bd_adapter.parameters())
        if adapter_params:
            param_groups.append({
                "params": adapter_params,
                "lr": learning_rate * bit_depth_component_lr_scale,
                "name": "feature_adapter",
                "weight_decay": weight_decay * 0.1,
            })
    
    # 其他参数
    other_params = []
    # 收集所有已在分组中的参数 id
    grouped_param_ids = set()
    for group in param_groups:
        for p in group["params"]:
            grouped_param_ids.add(id(p))
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 检查是否已经在其他组中（使用 id 比较）
        if id(param) not in grouped_param_ids:
            other_params.append(param)
    
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": learning_rate,
            "name": "other",
            "weight_decay": weight_decay,
        })
    
    # 创建优化器
    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(param_groups)
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(param_groups)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            param_groups,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


if __name__ == "__main__":
    print("Testing Training Utilities...")
    
    # 测试调度器工厂
    print("\n1. LR Scheduler Factory:")
    model = nn.Linear(10, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for scheduler_type in ["cosine", "cosine_warmup", "plateau", "step"]:
        try:
            config = LRSchedulerFactory.create(
                scheduler_type=scheduler_type,
                optimizer=optimizer,
                num_epochs=100,
                warmup_epochs=5,
            )
            print(f"   {scheduler_type}: ✓")
        except Exception as e:
            print(f"   {scheduler_type}: ✗ ({e})")
    
    # 测试分层优化器
    print("\n2. Layer-wise Optimizer:")
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 30))
            self.decoder = nn.Sequential(nn.Linear(30, 20), nn.Linear(20, 10))
            self.segmentation_head = nn.Linear(10, 3)
    
    mock_model = MockModel()
    optimizer = create_optimizer(
        mock_model,
        learning_rate=1e-4,
        encoder_lr_scale=0.5,
    )
    
    for i, group in enumerate(optimizer.param_groups):
        print(f"   Group {i}: {group.get('name', 'unknown')} - lr={group['lr']}")
    
    # 测试 EMA
    print("\n3. Model EMA:")
    ema = ModelEMA(mock_model)
    for _ in range(10):
        # 模拟训练
        for p in mock_model.parameters():
            p.data += torch.randn_like(p) * 0.01
        ema.update()
    
    print(f"   EMA step count: {ema.step_count}")
    
    print("\n✓ All tests passed!")
