"""
配置管理工具
============

提供YAML配置的加载、保存和合并功能，支持默认配置生成。

主要功能:
- 加载YAML配置文件
- 保存配置到YAML文件
- 递归合并配置字典
- 提供默认配置模板

使用示例:
    >>> from utils.config import load_config, save_config, merge_config
    >>> config = load_config('configs/cloudseg.yaml')
    >>> custom = {'training': {'batch_size': 16}}
    >>> merged = merge_config(config, custom)
    >>> save_config(merged, 'configs/custom.yaml')
"""
import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 配置字典
        
    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    # 确保目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def merge_config(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    递归合并两个配置字典
    
    override_config 中的值会覆盖 base_config 中的值。
    支持嵌套字典的递归合并。
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        Dict[str, Any]: 合并后的配置
        
    Example:
        >>> base = {'model': {'encoder': 'resnet', 'lr': 0.01}}
        >>> override = {'model': {'lr': 0.001}, 'batch_size': 8}
        >>> merged = merge_config(base, override)
        >>> # 结果: {'model': {'encoder': 'resnet', 'lr': 0.001}, 'batch_size': 8}
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if (
            key in merged 
            and isinstance(merged[key], dict) 
            and isinstance(value, dict)
        ):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def deep_update(
    base_dict: Dict[str, Any],
    update_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    深度更新字典
    
    与 merge_config 类似，但会直接修改 base_dict。
    
    Args:
        base_dict: 基础字典
        update_dict: 更新字典
        
    Returns:
        Dict[str, Any]: 更新后的字典（修改后的 base_dict）
    """
    for key, value in update_dict.items():
        if (
            key in base_dict 
            and isinstance(base_dict[key], dict) 
            and isinstance(value, dict)
        ):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def get_config_value(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None,
    separator: str = '.'
) -> Any:
    """
    通过路径获取配置值
    
    Args:
        config: 配置字典
        key_path: 键路径 (e.g., 'model.encoder.type')
        default: 默认值
        separator: 路径分隔符
        
    Returns:
        Any: 配置值或默认值
        
    Example:
        >>> config = {'model': {'encoder': {'type': 'resnet'}}}
        >>> value = get_config_value(config, 'model.encoder.type')
        >>> # 结果: 'resnet'
    """
    keys = key_path.split(separator)
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def set_config_value(
    config: Dict[str, Any],
    key_path: str,
    value: Any,
    separator: str = '.'
) -> None:
    """
    通过路径设置配置值
    
    Args:
        config: 配置字典
        key_path: 键路径 (e.g., 'model.encoder.type')
        value: 要设置的值
        separator: 路径分隔符
        
    Example:
        >>> config = {'model': {'encoder': {}}}
        >>> set_config_value(config, 'model.encoder.type', 'resnet')
        >>> # 结果: {'model': {'encoder': {'type': 'resnet'}}}
    """
    keys = key_path.split(separator)
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置模板
    
    返回一个完整的默认配置字典，可作为自定义配置的基础。
    
    Returns:
        Dict[str, Any]: 默认配置字典
    """
    return {
        'model': {
            'name': 'CloudSenseNet',
            'num_classes': 2,
            'encoder': {
                'type': 'convnext',
                'enabled': True,
                'convnext': {
                    'model_name': 'convnext_tiny',
                    'pretrained': True,
                    'drop_path_rate': 0.1
                }
            },
            'semantic_enhancement': {
                'enabled': False,
                'vocabulary_size': 2,
                'patch_size': 16,
                'embed_dim': 768,
                'merge_stage': 0
            },
            'fusion': {
                'enabled': True,
                'type': 'fpn',
                'fpn': {
                    'out_channels': 256,
                    'num_levels': 4
                }
            },
            'decoder': {
                'type': 'segformer',
                'enabled': True,
                'segformer': {
                    'embed_dim': 256,
                    'num_heads': 8,
                    'mlp_ratio': 4,
                    'drop_rate': 0.1
                }
            },
            'auxiliary_head': {
                'enabled': False,
                'loss_weight': 0.4
            }
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 100,
            'num_workers': 4,
            'optimizer': {
                'type': 'AdamW',
                'lr': 1e-4,
                'weight_decay': 1e-4,
                'backbone_lr_mult': 0.1
            },
            'scheduler': {
                'type': 'cosine_warmup',
                'warmup_epochs': 5,
                'T_0': 10,
                'T_mult': 2
            },
            'loss': {
                'types': ['dice', 'bce'],
                'weights': [0.5, 0.5]
            },
            'augmentation': {
                'enabled': True,
                'random_crop_size': [512, 512],
                'horizontal_flip': 0.5,
                'vertical_flip': 0.5,
                'brightness': 0.2,
                'contrast': 0.2,
                'gaussian_noise': 0.01
            }
        },
        'data': {
            'train_data_path': '../Data/RICE2',
            'val_data_path': '../Data/HRC_WHU',
            'test_data_path': '../Data/HRC_WHU',
            'inference': {
                'mode': 'sliding_window',
                'window_size': 512,
                'stride': 256,
                'scales': [0.5, 1.0, 1.5]
            }
        },
        'experiment': {
            'seed': 42,
            'save_dir': './experiments',
            'log_interval': 10,
            'val_interval': 1,
            'save_best_only': True
        }
    }


def validate_config(config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    验证配置完整性
    
    Args:
        config: 配置字典
        
    Returns:
        Tuple[bool, List[str]]: (是否有效, 错误信息列表)
    """
    errors = []
    
    # 检查必需字段
    required_sections = ['model', 'training', 'experiment']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: '{section}'")
    
    # 检查模型配置
    if 'model' in config:
        model_config = config['model']
        if 'num_classes' not in model_config:
            errors.append("Missing 'model.num_classes'")
        if 'encoder' not in model_config:
            errors.append("Missing 'model.encoder'")
    
    # 检查训练配置
    if 'training' in config:
        training_config = config['training']
        if 'batch_size' not in training_config:
            errors.append("Missing 'training.batch_size'")
        if 'num_epochs' not in training_config:
            errors.append("Missing 'training.num_epochs'")
    
    return len(errors) == 0, errors


class ConfigManager:
    """
    配置管理器类
    
    提供配置的全局管理和访问接口。
    
    Example:
        >>> manager = ConfigManager()
        >>> manager.load('config.yaml')
        >>> batch_size = manager.get('training.batch_size', default=8)
        >>> manager.set('training.lr', 0.001)
        >>> manager.save('output.yaml')
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 初始配置字典（可选）
        """
        self._config = config or {}
    
    def load(self, config_path: str) -> 'ConfigManager':
        """加载配置文件"""
        self._config = load_config(config_path)
        return self
    
    def save(self, save_path: str) -> None:
        """保存配置到文件"""
        save_config(self._config, save_path)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值"""
        return get_config_value(self._config, key_path, default)
    
    def set(self, key_path: str, value: Any) -> 'ConfigManager':
        """设置配置值"""
        set_config_value(self._config, key_path, value)
        return self
    
    def merge(self, other: Dict[str, Any]) -> 'ConfigManager':
        """合并其他配置"""
        self._config = merge_config(self._config, other)
        return self
    
    def validate(self) -> tuple[bool, list[str]]:
        """验证配置"""
        return validate_config(self._config)
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取配置字典"""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式设置"""
        self._config[key] = value
