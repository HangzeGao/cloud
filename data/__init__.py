"""
云分割数据加载模块

直接使用上层 Data 目录的 UnifiedCloudDataset
"""
import sys
from pathlib import Path

# 添加上层 Data 目录到路径
data_dir = Path(__file__).parent.parent.parent / "Data"
if str(data_dir) not in sys.path:
    sys.path.insert(0, str(data_dir))

# 从上层的 cloud_dataset_loader 导入
from cloud_dataset_loader import UnifiedCloudDataset

# 保留本项目的辅助功能
from .cloud_dataset import (
    CloudAugmentation,
    create_mixed_dataloaders,
)

__all__ = [
    'UnifiedCloudDataset',  # 上层的数据集加载器
    'CloudAugmentation',
    'create_mixed_dataloaders',
]
