"""
快速演示脚本 - 展示CloudSense-Net的灵活配置能力
"""
import torch
from models import CloudSenseNet, create_model_from_preset


def demo_preset_models():
    """演示预设模型"""
    print("="*60)
    print("Demo 1: 使用预设快速创建模型")
    print("="*60)
    
    presets = ['high_accuracy', 'fast', 'balanced', 'minimal']
    
    for preset in presets:
        print(f"\n--- {preset.upper()} ---")
        model = create_model_from_preset(preset, num_classes=2)
        
        # 打印架构信息
        info = model.get_model_info()
        print(f"Encoder: {info['encoder_type']}")
        print(f"Fusion: {'✓ ' + info['fusion_type'] if info['use_fusion'] else '✗'}")
        print(f"Decoder: {info['decoder_type']}")
        print(f"Total params: {info['total_params']:,}")


def demo_custom_config():
    """演示自定义配置"""
    print("\n" + "="*60)
    print("Demo 2: 自定义配置")
    print("="*60)
    
    # 自定义配置
    custom_config = {
        'model': {
            'name': 'MyCustomModel',
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
                'enabled': True,  # 启用语义增强
                'vocabulary_size': 2,
                'patch_size': 16,
                'embed_dim': 768,
                'merge_stage': 0
            },
            'fusion': {
                'enabled': True,
                'type': 'bifpn',  # 使用BiFPN融合
                'bifpn': {
                    'out_channels': 256,
                    'num_iterations': 2
                }
            },
            'decoder': {
                'type': 'segformer',  # 使用SegFormer解码器
                'enabled': True,
                'segformer': {
                    'embed_dim': 256
                }
            },
            'auxiliary_head': {
                'enabled': False
            }
        },
        'training': {
            'loss': {
                'types': ['dice', 'bce'],
                'weights': [0.5, 0.5]
            }
        }
    }
    
    # 创建模型
    model = CloudSenseNet(custom_config)
    model.print_architecture()
    
    return model


def demo_inference_modes(model):
    """演示不同的推理模式"""
    print("\n" + "="*60)
    print("Demo 3: 不同推理模式")
    print("="*60)
    
    # 创建测试图像（任意尺寸）
    test_sizes = [
        (3, 512, 512),    # 标准尺寸
        (3, 1024, 1024),  # 大图像
        (3, 2048, 1536),  # 非标准比例
    ]
    
    model.eval()
    
    for size in test_sizes:
        print(f"\n输入尺寸: {size}")
        dummy_input = torch.randn(1, *size)
        
        with torch.no_grad():
            # 直接推理
            output = model(dummy_input)
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output
            
            print(f"  输出尺寸: {logits.shape}")
            print(f"  云分割结果: 批大小={logits.shape[0]}, 类别数={logits.shape[1]}, "
                  f"高={logits.shape[2]}, 宽={logits.shape[3]}")


def demo_architecture_switches():
    """演示架构开关效果"""
    print("\n" + "="*60)
    print("Demo 4: 架构开关对比")
    print("="*60)
    
    # 基础配置
    base_config = {
        'model': {
            'name': 'SwitchDemo',
            'num_classes': 2,
            'encoder': {
                'type': 'convnext',
                'enabled': True,
                'convnext': {'model_name': 'convnext_tiny', 'pretrained': False}
            },
            'semantic_enhancement': {'enabled': False},
            'fusion': {'enabled': False},
            'decoder': {
                'type': 'segformer',
                'enabled': True,
                'segformer': {'embed_dim': 256}
            },
            'auxiliary_head': {'enabled': False}
        },
        'training': {'loss': {'types': ['dice'], 'weights': [1.0]}}
    }
    
    # 对比不同开关组合
    switch_combinations = [
        {'semantic': False, 'fusion': False, 'desc': 'Baseline (无增强)'},
        {'semantic': True, 'fusion': False, 'desc': '仅语义增强'},
        {'semantic': False, 'fusion': True, 'desc': '仅特征融合'},
        {'semantic': True, 'fusion': True, 'desc': '语义增强+特征融合'},
    ]
    
    for combo in switch_combinations:
        config = base_config.copy()
        config['model']['semantic_enhancement']['enabled'] = combo['semantic']
        config['model']['fusion']['enabled'] = combo['fusion']
        if combo['fusion']:
            config['model']['fusion']['type'] = 'fpn'
            config['model']['fusion']['fpn'] = {'out_channels': 256, 'num_levels': 4}
        
        model = CloudSenseNet(config)
        info = model.get_model_info()
        
        print(f"\n{combo['desc']}:")
        print(f"  总参数量: {info['total_params']:,}")
        print(f"  语义增强: {'✓' if info['use_semantic_enhancement'] else '✗'}")
        print(f"  特征融合: {'✓' if info['use_fusion'] else '✗'}")


if __name__ == '__main__':
    print("CloudSense-Net 演示脚本")
    print("="*60)
    
    # 运行演示
    demo_preset_models()
    model = demo_custom_config()
    demo_inference_modes(model)
    demo_architecture_switches()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print("\n提示:")
    print("1. 使用预设快速创建: create_model_from_preset('balanced')")
    print("2. 自定义配置: CloudSenseNet(your_config)")
    print("3. 查看架构: model.print_architecture()")
    print("4. 获取信息: model.get_model_info()")
