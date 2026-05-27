"""
位深度估计器模块

包含三种位深度估计器实现。
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义位深度范围 [MIN_BIT_DEPTH, MAX_BIT_DEPTH]（包含边界）
MIN_BIT_DEPTH = 8
MAX_BIT_DEPTH = 12


def get_bit_depth_range() -> Tuple[int, int]:
    """获取位深度范围"""
    return MIN_BIT_DEPTH, MAX_BIT_DEPTH


def get_num_bit_depths() -> int:
    """获取位深度类别数"""
    return MAX_BIT_DEPTH - MIN_BIT_DEPTH + 1


def get_bit_depth_values() -> List[int]:
    """获取所有位深度值列表"""
    return list(range(MIN_BIT_DEPTH, MAX_BIT_DEPTH + 1))


def get_bit_depth_tensor():
    """获取位深度tensor（用于PyTorch）"""
    import torch
    return torch.tensor(get_bit_depth_values()).float()


def pretty_print_dict(
        data: Dict[str, Any],
        title: Optional[str] = None,
        indent: int = 2,
        sort_keys: bool = False,
        max_width: int = 80,
) -> str:
    """
    优雅打印字典，支持嵌套、中文对齐、标题。

    Args:
        data: 要打印的字典
        title: 可选标题（会加边框高亮）
        indent: 缩进空格数
        sort_keys: 是否按键排序
        max_width: 行最大宽度（用于截断提示）

    Returns:
        格式化后的字符串

    Example:
        >>> print(pretty_print_dict({'a': 1, 'b': {'c': 2}}, title="Config"))
    """
    lines = []

    # 标题
    if title:
        lines.append(f"╔{'═' * (len(title) + 4)}╗")
        lines.append(f"║  {title}  ║")
        lines.append(f"╚{'═' * (len(title) + 4)}╝")
        lines.append("")

    # 主体：JSON 风格，但保留中文可读性
    formatted = json.dumps(
        data,
        ensure_ascii=False,
        indent=indent,
        sort_keys=sort_keys,
        default=lambda o: repr(o) if not isinstance(o, (int, float, str, bool, type(None), list, dict)) else o,
    )

    # 为类引用添加友好提示（如 <class 'Foo'> → class Foo）
    import re
    formatted = re.sub(r'"<<class \'([^\']+)\'>"', r'class \1', formatted)

    lines.append(formatted)

    # 底部统计
    lines.append("")
    lines.append(f"  └─ 共 {len(data)} 项")

    return "\n".join(lines)


class MinimalBitDepthEstimator(nn.Module):
    """
    极简版位深度估计器（最快，仅用于监控）
    
    只做前向传播，不参与梯度计算，用于监控和日志记录
    训练时可以完全关闭，推理时启用
    """
    
    def __init__(self, in_channels: int = 4, num_bit_depths: int = None):
        super().__init__()
        # 从配置中心获取默认值
        self.num_bit_depths = num_bit_depths or get_num_bit_depths()
        min_bd, max_bd = get_bit_depth_range()
        
        # 动态生成位深度tensor
        self.register_buffer('bit_depths', get_bit_depth_tensor())
        self.register_buffer('min_bit_depth', torch.tensor(min_bd).float())
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """不计算梯度，只返回估计值"""
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        
        # 只计算一个简单指标：动态范围
        dynamic_range = (x_flat.max(dim=2)[0] - x_flat.min(dim=2)[0]).mean(dim=1)  # [B]
        
        # 启发式规则：动态范围小的更可能是低比特
        min_bd = float(self.min_bit_depth)
        max_bd = float(self.bit_depths.max())
        scale_factor = (max_bd - min_bd) / 0.2  # 假设0.8-1.0映射到整个范围
        estimated = torch.clamp(min_bd + (dynamic_range - 0.8) * scale_factor, min_bd, max_bd)
        
        # 创建伪logits（用于兼容性）
        logits = torch.zeros(B, self.num_bit_depths, device=x.device)
        # 在估计值附近设置高概率
        idx = (estimated - min_bd).long().clamp(0, self.num_bit_depths - 1)
        logits.scatter_(1, idx.unsqueeze(1), 1.0)
        
        return logits, estimated


class ConvBitDepthEstimator(nn.Module):
    """
    轻量卷积位深度估计器（速度与精度的平衡）
    
    使用极轻量卷积网络
    复杂度：O(H*W)，但并行度高，实际很快
    """
    
    def __init__(self, in_channels: int = 4, num_bit_depths: int = None):
        super().__init__()
        # 从配置中心获取默认值
        num_bit_depths = num_bit_depths or get_num_bit_depths()
        
        # 极轻量编码器：只下采样4次
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_bit_depths),
        )
        
        # 动态生成位深度tensor
        self.register_buffer('bit_depths', get_bit_depth_tensor())
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        logits = self.classifier(features)
        
        probs = F.softmax(logits, dim=1)
        bit_depths = self.bit_depths.to(x.device)
        estimated = (probs * bit_depths).sum(dim=1)
        
        return logits, estimated


class StatisticalBitDepthEstimator(nn.Module):
    """
    B1: 基于统计特征的位深度估计器（优化版）
    
    性能优化：
    - 移除昂贵的torch.quantile操作
    - 只使用O(1)复杂度的基本统计量
    - 增加直方图近似百分位数（向量化，无排序）
    """
    
    def __init__(self, in_channels: int = 4, hidden_dim: int = 32, num_bit_depths: int = None):
        super().__init__()
        self.in_channels = in_channels
        # 从配置中心获取默认值
        self.num_bit_depths = num_bit_depths or get_num_bit_depths()

        stat_dim = in_channels * 10

        self.mlp = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_bit_depths),
        )

        # 动态生成位深度tensor
        self.register_buffer('bit_depths', get_bit_depth_tensor())
        self.register_buffer('hist_bins', torch.linspace(0, 1, 65))
        
    def extract_statistics(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        
        mean = x_flat.mean(dim=2)
        std = x_flat.std(dim=2)
        min_val = x_flat.min(dim=2)[0]
        max_val = x_flat.max(dim=2)[0]
        range_val = max_val - min_val
        
        low_density = (x_flat < 0.05).float().mean(dim=2)
        high_density = (x_flat > 0.95).float().mean(dim=2)
        mean_neighborhood = (torch.abs(x_flat - mean.unsqueeze(2)) < 0.1).float().mean(dim=2)
        
        x_expanded = x_flat.unsqueeze(-1)
        bins_expanded = self.hist_bins.view(1, 1, 1, -1).to(x.device)
        
        bin_width = 1.0 / (len(self.hist_bins) - 1)
        dist_to_bins = torch.abs(x_expanded - bins_expanded) / bin_width
        weights = torch.clamp(1 - dist_to_bins, 0, 1)
        
        hist = weights.sum(dim=2)
        hist = hist / (hist.sum(dim=2, keepdim=True) + 1e-8)
        
        entropy = -(hist * torch.log(hist + 1e-8)).sum(dim=2)
        
        hist_diff = hist[:, :, 1:-1] - hist[:, :, :-2]
        hist_diff_next = hist[:, :, 2:] - hist[:, :, 1:-1]
        peak_count = ((hist_diff > 0) & (hist_diff_next < 0)).float().sum(dim=2)
        
        stats = torch.cat([
            mean, std, min_val, max_val, range_val,
            low_density, high_density, mean_neighborhood,
            entropy, peak_count
        ], dim=1)
        
        return stats
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stats = self.extract_statistics(x)
        logits = self.mlp(stats)
        
        probs = F.softmax(logits, dim=1)
        bit_depths = self.bit_depths.to(x.device)
        estimated_bit_depth = (probs * bit_depths).sum(dim=1)
        
        return logits, estimated_bit_depth


class BitDepthEstimatorFactory:
    """Factory for creating bit-depth estimators."""

    _ESTIMATOR_CONFIGS = {
        'minimal': {
            'class': MinimalBitDepthEstimator,
            'description': '极简版，无训练开销，仅用于监控',
            'speed': '⚡⚡⚡ 最快',
            'trainable': False,
        },
        'conv': {
            'class': ConvBitDepthEstimator,
            'description': '轻量卷积，平衡速度与精度',
            'speed': '⚡⚡ 快',
            'trainable': True,
        },
        'statistical': {
            'class': StatisticalBitDepthEstimator,
            'description': '统计特征MLP，精度最高但较慢',
            'speed': '⚡ 中等',
            'trainable': True,
        },
    }

    @staticmethod
    def create(
            estimator_type: str,
            in_channels: int = 4,
            num_bit_depths: int = None,
    ):
        """
        静态工厂方法：创建指定位深度估计器

        Args:
            estimator_type: 'minimal', 'conv', 'statistical'
            in_channels: 输入通道数
            num_bit_depths: 位深度类别数（None则使用配置中心默认值）

        Returns:
            BitDepthEstimator 实例

        Example:
            >>> estimator = BitDepthEstimatorFactory.create('conv', in_channels=4)
            >>> logits, estimated = estimator(x)
        """
        if estimator_type not in BitDepthEstimatorFactory._ESTIMATOR_CONFIGS:
            available = list(BitDepthEstimatorFactory._ESTIMATOR_CONFIGS.keys())
            raise ValueError(
                f"Unknown estimator_type: {estimator_type}. "
                f"Choose from {available}"
            )

        num_bit_depths = num_bit_depths or get_num_bit_depths()

        estimator_class = BitDepthEstimatorFactory._ESTIMATOR_CONFIGS[estimator_type]['class']
        return estimator_class(in_channels, num_bit_depths)

    @classmethod
    def register(
            cls,
            name: str,
            estimator_class: type,
            description: str = '',
            speed: str = '',
            trainable: bool = True,
    ):
        """动态注册新的估计器类型"""
        cls._ESTIMATOR_CONFIGS[name] = {
            'class': estimator_class,
            'description': description,
            'speed': speed,
            'trainable': trainable,
        }

    @classmethod
    def list_configs(cls):
        """返回可用的估计器配置（不含类引用）"""
        return {
            k: {
                'description': v['description'],
                'speed': v['speed'],
                'trainable': v['trainable'],
            }
            for k, v in cls._ESTIMATOR_CONFIGS.items()
        }


if __name__ == '__main__':
    print(pretty_print_dict(BitDepthEstimatorFactory.list_configs()))
