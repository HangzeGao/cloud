"""
配置管理工具
"""
import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """保存配置到YAML文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_config(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并两个配置字典，override_config会覆盖base_config中的值
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
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
