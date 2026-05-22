"""
CloudSense-Net 演示脚本
展示模型的灵活配置能力和新特性
"""
import torch
from models import CloudSenseNet, create_model_from_preset


def demo_preset_models():
    """演示预设模型 - 展示 timm 真实模型加载"""
    print("=" * 70)
    print("Demo 1: 使用预设快速创建模型（基于 timm 真实预训练模型）")
    print("=" * 70)

    presets = ['high_accuracy', 'fast', 'balanced', 'minimal']

    for preset in presets:
        print(f"\n--- {preset.upper()} ---")
        try:
            model = create_model_from_preset(preset, num_classes=2)

            # 打印架构信息
            info = model.get_model_info()
            print(f"Encoder: {info['encoder_type']}")
            print(f"Encoder Channels: {info['encoder_channels']}")
            print(f"Fusion: {'✓ ' + info['fusion_type'] if info['use_fusion'] else '✗'}")
            print(f"Decoder: {info['decoder_type']}")
            print(f"Total params: {info['total_params']:,}")
            print(f"Channel Adaptive: {'✓' if info['use_channel_adaptive'] else '✗'}")
        except Exception as e:
            print(f"  ⚠️  {preset} 需要安装 timm: pip install timm")
            print(f"     Error: {e}")


def demo_backbone_variants():
    """演示不同 Backbone 变体选择"""
    print("\n" + "=" * 70)
    print("Demo 2: Backbone 变体选择（EfficientNet B0-B4）")
    print("=" * 70)

    variants = [
        ('efficientnet_b0', '移动端/边缘设备'),
        ('efficientnet_b1', '低资源环境'),
        ('efficientnet_b3', '平衡配置（推荐）'),
        ('efficientnet_b4', '高精度需求'),
    ]

    for model_name, use_case in variants:
        print(f"\n--- {model_name} ({use_case}) ---")
        try:
            config = {
                'model': {
                    'name': f'Demo_{model_name}',
                    'num_classes': 2,
                    'encoder': {
                        'type': 'efficientnet',
                        'enabled': True,
                        'efficientnet': {
                            'model_name': model_name,
                            'pretrained': False  # 演示时关闭以加快加载
                        }
                    },
                    'channel_adaptive': {'enabled': True, 'method': 'conv', 'out_channels': 4},
                    'semantic_enhancement': {'enabled': False},
                    'fusion': {'enabled': False},
                    'decoder': {
                        'type': 'simple',
                        'enabled': True
                    },
                    'auxiliary_head': {'enabled': False}
                },
                'training': {'loss': {'types': ['dice'], 'weights': [1.0]}},
                'data': {'use_nir': True, 'normalize': 'percentile'}
            }

            model = CloudSenseNet(config)
            info = model.get_model_info()
            print(f"  参数量: {info['total_params']:,}")
            print(f"  特征通道: {info['encoder_channels']}")
        except Exception as e:
            print(f"  ⚠️  需要安装 timm: pip install timm")


def demo_resnext_encoder():
    """演示 ResNeXt 编码器"""
    print("\n" + "=" * 70)
    print("Demo 3: ResNeXt 分组卷积编码器")
    print("=" * 70)

    resnext_configs = [
        ('resnext50_32x4d', '标准配置'),
        ('resnext101_32x8d', '高精度配置'),
    ]

    for model_name, desc in resnext_configs:
        print(f"\n--- ResNeXt {model_name} ({desc}) ---")
        try:
            config = {
                'model': {
                    'name': f'ResNeXt_Demo',
                    'num_classes': 2,
                    'encoder': {
                        'type': 'resnext',
                        'enabled': True,
                        'resnext': {
                            'model_name': model_name,
                            'pretrained': False
                        }
                    },
                    'channel_adaptive': {'enabled': True, 'method': 'conv', 'out_channels': 4},
                    'semantic_enhancement': {'enabled': False},
                    'fusion': {'enabled': True, 'type': 'fpn', 'fpn': {'out_channels': 256, 'num_levels': 4}},
                    'decoder': {'type': 'segformer', 'enabled': True, 'segformer': {'embed_dim': 256}},
                    'auxiliary_head': {'enabled': False}
                },
                'training': {'loss': {'types': ['dice'], 'weights': [1.0]}},
                'data': {'use_nir': True, 'normalize': 'percentile'}
            }

            model = CloudSenseNet(config)
            info = model.get_model_info()
            print(f"  参数量: {info['total_params']:,}")
            print(f"  特征通道: {info['encoder_channels']}")
        except Exception as e:
            print(f"  ⚠️  错误: {e}")


def demo_custom_config():
    """演示自定义配置 - 展示完整配置能力"""
    print("\n" + "=" * 70)
    print("Demo 4: 完整自定义配置（含所有新特性）")
    print("=" * 70)

    # 自定义配置 - 使用 ConvNeXt + BiFPN + warmup + AMP 示例
    custom_config = {
        'model': {
            'name': 'MyCustomModel',
            'num_classes': 2,
            'encoder': {
                'type': 'convnext',
                'enabled': True,
                'convnext': {
                    'model_name': 'convnext_tiny',  # 可选: tiny/small/base/large
                    'pretrained': False,  # 演示时关闭
                    'drop_path_rate': 0.1
                }
            },
            'channel_adaptive': {
                'enabled': True,
                'method': 'hybrid',  # 使用 hybrid 方法
                'use_attention': True,
                'out_channels': 4
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
                'type': 'bifpn',  # 使用 BiFPN
                'bifpn': {
                    'out_channels': 256,
                    'num_iterations': 2
                }
            },
            'decoder': {
                'type': 'segformer',
                'enabled': True,
                'segformer': {
                    'embed_dim': 256
                }
            },
            'auxiliary_head': {
                'enabled': True,  # 启用辅助头
                'loss_weight': 0.4
            }
        },
        'training': {
            'loss': {
                'types': ['dice', 'bce', 'focal'],  # 多损失组合
                'weights': [0.4, 0.3, 0.3],
                'focal_alpha': 0.25,
                'focal_gamma': 2.0
            },
            'scheduler': {
                'type': 'cosine',
                'warmup_epochs': 5  # warmup 配置
            },
            'amp': {
                'enabled': False  # 演示时关闭 AMP
            }
        },
        'data': {
            'use_nir': True,
            'normalize': 'percentile'
        }
    }

    try:
        model = CloudSenseNet(custom_config)
        model.print_architecture()
        return model
    except Exception as e:
        print(f"⚠️ 需要安装 timm: pip install timm")
        print(f"   Error: {e}")
        return None


def demo_fusion_variants():
    """演示不同的 Fusion 变体"""
    print("\n" + "=" * 70)
    print("Demo 5: Fusion 变体对比（8种可选）")
    print("=" * 70)

    fusion_variants = [
        ('fpn', '标准特征金字塔'),
        ('fpnv2', '改进版 FPN'),
        ('bifpn', '双向特征金字塔'),
        ('fastbifpn', '快速 BiFPN'),
        ('aspp', '空洞空间金字塔'),
        ('asppv2', '改进版 ASPP'),
        ('lightaspp', '轻量级 ASPP'),
    ]

    for fusion_type, desc in fusion_variants:
        print(f"\n--- Fusion: {fusion_type} ({desc}) ---")
        try:
            config = {
                'model': {
                    'name': f'Fusion_Demo',
                    'num_classes': 2,
                    'encoder': {
                        'type': 'resnet',  # 使用稳定的 ResNet
                        'enabled': True,
                        'resnet': {'model_name': 'resnet50', 'pretrained': False}
                    },
                    'channel_adaptive': {'enabled': False},
                    'semantic_enhancement': {'enabled': False},
                    'fusion': {
                        'enabled': True,
                        'type': fusion_type
                    },
                    'decoder': {'type': 'simple', 'enabled': True},
                    'auxiliary_head': {'enabled': False}
                },
                'training': {'loss': {'types': ['dice'], 'weights': [1.0]}},
                'data': {'use_nir': False, 'normalize': 'percentile'}
            }

            # 添加 fusion 特定配置
            if fusion_type.startswith('fpn'):
                config['model']['fusion'][fusion_type] = {'out_channels': 256, 'num_levels': 4}
            elif fusion_type.startswith('bifpn'):
                config['model']['fusion'][fusion_type] = {'out_channels': 256, 'num_iterations': 2}
            elif fusion_type.startswith('aspp'):
                config['model']['fusion'][fusion_type] = {'out_channels': 256}

            model = CloudSenseNet(config)
            info = model.get_model_info()
            print(f"  参数量: {info['total_params']:,}")
            print(f"  融合类型: {info['fusion_type']}")
        except Exception as e:
            print(f"  ⚠️  错误: {e}")


def demo_freeze_encoder():
    """演示分层冻结功能"""
    print("\n" + "=" * 70)
    print("Demo 6: 分层冻结 Encoder（微调场景）")
    print("=" * 70)

    # 创建模型
    config = {
        'model': {
            'name': 'FreezeDemo',
            'num_classes': 2,
            'encoder': {
                'type': 'resnet',
                'enabled': True,
                'resnet': {'model_name': 'resnet50', 'pretrained': False}
            },
            'channel_adaptive': {'enabled': False},
            'semantic_enhancement': {'enabled': False},
            'fusion': {'enabled': False},
            'decoder': {'type': 'simple', 'enabled': True},
            'auxiliary_head': {'enabled': False}
        },
        'training': {'loss': {'types': ['dice'], 'weights': [1.0]}},
        'data': {'use_nir': False, 'normalize': 'percentile'}
    }

    model = CloudSenseNet(config)

    freeze_options = [
        (0, '不冻结（全部训练）'),
        (1, '冻结 stem'),
        (2, '冻结 stem + layer1'),
        (-1, '冻结全部 backbone'),
    ]

    for freeze_blocks, desc in freeze_options:
        print(f"\n--- freeze_blocks={freeze_blocks} ({desc}) ---")
        model_copy = CloudSenseNet(config)  # 重新创建模型
        model_copy.freeze_encoder(freeze_blocks=freeze_blocks)

        # 统计可训练参数
        info = model_copy.get_model_info()
        total = info['total_params']
        trainable = info['trainable_params']
        frozen = total - trainable
        print(f"  总参数: {total:,}")
        print(f"  可训练: {trainable:,} ({100 * trainable / total:.1f}%)")
        print(f"  已冻结: {frozen:,} ({100 * frozen / total:.1f}%)")


def demo_inference_modes(model):
    """演示不同的推理模式"""
    if model is None:
        print("\n" + "=" * 70)
        print("Demo 7: 推理模式演示（跳过 - 模型创建失败）")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print("Demo 7: 不同推理模式")
    print("=" * 70)

    # 创建测试图像（任意尺寸）
    test_sizes = [
        (4, 512, 512),    # 4通道标准尺寸
        (3, 1024, 1024),  # 3通道大图像
        (4, 2048, 1536),  # 4通道非标准比例
    ]

    model.eval()

    for size in test_sizes:
        print(f"\n输入尺寸: {size} ({size[0]}通道)")
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


def demo_channel_adaptive():
    """演示通道自适应的 3/4 通道切换"""
    print("\n" + "=" * 70)
    print("Demo 8: 通道自适应 - 3/4通道自动切换")
    print("=" * 70)

    # 创建支持通道自适应的模型
    config = {
        'model': {
            'name': 'ChannelAdaptiveDemo',
            'num_classes': 2,
            'encoder': {
                'type': 'resnet',
                'enabled': True,
                'resnet': {'model_name': 'resnet50', 'pretrained': False}
            },
            'channel_adaptive': {
                'enabled': True,
                'method': 'conv',
                'use_attention': True,
                'out_channels': 4
            },
            'semantic_enhancement': {'enabled': False},
            'fusion': {'enabled': False},
            'decoder': {'type': 'simple', 'enabled': True},
            'auxiliary_head': {'enabled': False}
        },
        'training': {'loss': {'types': ['dice'], 'weights': [1.0]}},
        'data': {'use_nir': True, 'normalize': 'percentile'}
    }

    model = CloudSenseNet(config)
    model.eval()

    # 测试不同通道输入
    test_inputs = [
        (3, 512, 512, "RGB 3通道输入"),
        (4, 512, 512, "RGB+NIR 4通道输入"),
    ]

    for channels, h, w, desc in test_inputs:
        print(f"\n--- {desc} ---")
        dummy_input = torch.randn(1, channels, h, w)

        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, dict):
                logits = output['logits']
            else:
                logits = output

            print(f"  输入: {dummy_input.shape}")
            print(f"  输出: {logits.shape}")
            print(f"  ✓ 通道自适应成功")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("CloudSense-Net 演示脚本")
    print("展示 v2.0 新特性：timm 模型、AMP、warmup、分层冻结等")
    print("=" * 70)

    # 运行演示
    demo_preset_models()
    demo_backbone_variants()
    demo_resnext_encoder()
    model = demo_custom_config()
    demo_fusion_variants()
    demo_freeze_encoder()
    demo_channel_adaptive()
    demo_inference_modes(model)

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\n💡 提示:")
    print("1. 使用预设快速创建: create_model_from_preset('balanced')")
    print("2. EfficientNet 变体: B0(轻量) -> B4(高精度) -> B7(顶级)")
    print("3. ResNeXt: 分组卷积改进的 ResNet")
    print("4. 启用 AMP (CUDA): config['training']['amp']['enabled'] = True")
    print("5. 学习率 warmup: config['training']['scheduler']['warmup_epochs'] = 5")
    print("6. 分层冻结: model.freeze_encoder(freeze_blocks=2)")
    print("7. 更多 Fusion 变体: fpnv2, fastbifpn, asppv2, lightaspp")
    print("\n📦 需要安装 timm 以使用完整功能:")
    print("   pip install timm")
