"""
测试 RGB+NIR 四通道功能
验证 NIR 生成算法和模型 4 通道输入
"""
import torch
import numpy as np
from PIL import Image

from data import NIRGenerator, CloudSegmentationDataset
from models import CloudSenseNet, build_model
from utils import load_config


def test_nir_generator():
    """测试 NIR 生成器"""
    print("\n" + "="*60)
    print("测试 1: NIR 生成器算法")
    print("="*60)
    
    # 创建模拟 RGB 图像 (模拟有云的场景)
    # 云通常是白色/灰色，高 R、G、B 值
    h, w = 256, 256
    
    # 创建模拟图像: 上半部分是云 (高亮)，下半部分是地面 (植被)
    rgb_image = torch.zeros(3, h, w)
    
    # 云区域 (上半部分): 高 R, G, B
    rgb_image[:, :h//2, :] = torch.tensor([0.9, 0.9, 0.9]).view(3, 1, 1)
    
    # 地面区域 (下半部分): 植被特征 - 低 R, 高 G, 低 B
    rgb_image[0, h//2:, :] = 0.2  # R
    rgb_image[1, h//2:, :] = 0.6  # G
    rgb_image[2, h//2:, :] = 0.1  # B
    
    methods = ['physical', 'vegetation', 'enhanced', 'weighted']
    
    for method in methods:
        generator = NIRGenerator(method=method, gain=1.1, offset=0.05)
        nir = generator(rgb_image)
        
        # 验证 NIR 特性: 云区域 NIR 应该高于地面区域
        cloud_nir = nir[0, :h//2, :].mean().item()
        ground_nir = nir[0, h//2:, :].mean().item()
        
        print(f"\n方法: {method}")
        print(f"  云区域平均 NIR: {cloud_nir:.3f}")
        print(f"  地面区域平均 NIR: {ground_nir:.3f}")
        print(f"  对比度 (云/地面): {cloud_nir/ground_nir:.2f}x")
        
        # 云在 NIR 应该有更高反射率
        assert cloud_nir > ground_nir, f"错误: {method} 方法生成的 NIR 没有正确区分云和地面!"
    
    print("\n✅ 所有 NIR 生成算法测试通过!")
    print("   云区域 NIR 值均高于地面区域，符合物理特性")


def test_model_4channel():
    """测试模型支持 4 通道输入"""
    print("\n" + "="*60)
    print("测试 2: 模型 4 通道输入")
    print("="*60)
    
    # 加载带 NIR 的配置
    config = load_config('configs/cloudseg_rgb_nir.yaml')
    
    print("\n配置信息:")
    print(f"  use_nir: {config['data'].get('use_nir', True)}")
    print(f"  nir_method: {config['data'].get('nir_method', 'physical')}")
    print(f"  nir_gain: {config['data'].get('nir_gain', 1.1)}")
    
    # 创建模型
    print("\n构建模型...")
    model = build_model(config)
    model.eval()
    
    # 测试 4 通道输入
    batch_size = 2
    h, w = 512, 512
    
    # 模拟 4 通道输入 (RGB+NIR)
    x_4ch = torch.randn(batch_size, 4, h, w)
    
    print(f"\n输入形状: {x_4ch.shape} (4通道)")
    
    with torch.no_grad():
        output = model(x_4ch)
        logits = output['logits']
    
    print(f"输出形状: {logits.shape}")
    print(f"输出尺寸与输入匹配: {logits.shape[-2:] == (h, w)}")
    
    assert logits.shape == (batch_size, 2, h, w), "输出形状不匹配!"
    
    print("\n✅ 模型 4 通道输入测试通过!")


def test_3channel_fallback():
    """测试 3 通道回退模式"""
    print("\n" + "="*60)
    print("测试 3: 3 通道回退模式")
    print("="*60)
    
    # 加载配置并禁用 NIR
    config = load_config('configs/cloudseg_base.yaml')
    config['data']['use_nir'] = False
    
    print("\n配置信息:")
    print(f"  use_nir: {config['data'].get('use_nir', False)}")
    
    # 创建模型
    print("\n构建模型 (3通道模式)...")
    model = build_model(config)
    model.eval()
    
    # 测试 3 通道输入
    batch_size = 2
    h, w = 512, 512
    x_3ch = torch.randn(batch_size, 3, h, w)
    
    print(f"\n输入形状: {x_3ch.shape} (3通道)")
    
    with torch.no_grad():
        output = model(x_3ch)
        logits = output['logits']
    
    print(f"输出形状: {logits.shape}")
    
    assert logits.shape == (batch_size, 2, h, w), "输出形状不匹配!"
    
    print("\n✅ 模型 3 通道回退测试通过!")


def test_nir_visualization():
    """可视化 NIR 生成效果"""
    print("\n" + "="*60)
    print("测试 4: NIR 可视化对比")
    print("="*60)
    
    # 创建模拟 RGB 图像 (有云和地面)
    h, w = 256, 256
    rgb_image = torch.zeros(3, h, w)
    
    # 左半部分: 厚云 (高 R,G,B)
    rgb_image[:, :, :w//2] = torch.tensor([0.95, 0.95, 0.95]).view(3, 1, 1)
    
    # 右半部分: 薄云 (中 R,G,B)
    rgb_image[:, :, w//2:] = torch.tensor([0.7, 0.7, 0.75]).view(3, 1, 1)
    
    generator = NIRGenerator(method='physical', gain=1.1)
    nir = generator(rgb_image)
    
    thick_cloud_nir = nir[0, :, :w//2].mean().item()
    thin_cloud_nir = nir[0, :, w//2:].mean().item()
    
    print(f"\n厚云区域平均 NIR: {thick_cloud_nir:.3f}")
    print(f"薄云区域平均 NIR: {thin_cloud_nir:.3f}")
    print(f"差异: {thick_cloud_nir - thin_cloud_nir:.3f}")
    
    # 厚云应该有更高的 NIR
    assert thick_cloud_nir > thin_cloud_nir, "厚云 NIR 应该高于薄云!"
    
    print("\n✅ NIR 可视化对比测试通过!")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("CloudSense-Net RGB+NIR 四通道功能测试")
    print("="*60)
    
    try:
        test_nir_generator()
        test_model_4channel()
        test_3channel_fallback()
        test_nir_visualization()
        
        print("\n" + "="*60)
        print("🎉 所有测试通过! RGB+NIR 功能正常工作")
        print("="*60)
        print("\n使用说明:")
        print("  1. 在配置文件中设置 use_nir: true 启用 NIR")
        print("  2. 选择 nir_method: physical/vegetation/enhanced/weighted")
        print("  3. 调整 nir_gain 控制 NIR 增益 (推荐 1.0-1.3)")
        print("  4. 使用 configs/cloudseg_rgb_nir.yaml 作为参考配置")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
