"""
位深度配置中心

修改此文件中的 DEFAULT_BIT_DEPTH_RANGE 即可改变所有组件的位深度范围。
"""

from typing import List, Tuple

# ========================= 唯一需要修改的地方 =========================
# 定义位深度范围 [MIN_BIT_DEPTH, MAX_BIT_DEPTH]（包含边界）
MIN_BIT_DEPTH = 8
MAX_BIT_DEPTH = 12
# =====================================================================


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
