"""
位深度模拟变换使用示例

此文件展示如何在不同场景下使用 BitDepthSimulation 类
"""

import numpy as np
import albumentations as A

# 导入位深度模拟变换
from MyCloudSenseNet.benchmark.bit_depth_transform import (
    BitDepthSimulation,
    BitDepthSimulationWithLabel,
    RandomBitDepthBatch,
    create_bit_depth_transform,
    BIT_DEPTH_8_TO_11_RANDOM,
)


def example_1_basic_usage():
    """示例1: 基础用法 - 单图像变换"""
    print("=" * 60)
    print("示例1: 基础用法 - 单图像变换")
    print("=" * 60)
    
    # 创建模拟的归一化图像 [0, 1]
    image = np.random.rand(512, 512, 4).astype(np.float32)  # 4通道图像
    
    # 创建变换：50%概率模拟8-11位深度
    transform = BitDepthSimulation(bit_depth_range=(8, 11), p=0.5)
    
    # 应用变换
    result = transform(image=image)
    simulated_image = result['image']
    
    # 获取使用的位深度（如果应用了变换）
    bit_depth = result.get('simulated_bit_depth', None)
    
    print(f"输入图像范围: [{image.min():.4f}, {image.max():.4f}]")
    print(f"输出图像范围: [{simulated_image.min():.4f}, {simulated_image.max():.4f}]")
    print(f"模拟位深度: {bit_depth}")
    


def example_2_compose_with_other_transforms():
    """示例2: 与其他Albumentations变换组合使用"""
    print("\n" + "=" * 60)
    print("示例2: 与其他变换组合使用")
    print("=" * 60)
    
    # 创建图像
    image = np.random.rand(512, 512, 4).astype(np.float32)
    mask = np.random.randint(0, 3, (512, 512), dtype=np.uint8)  # 3类语义分割标签
    
    # 组合变换管道
    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.0625, rotate_limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        BitDepthSimulation(bit_depth_range=(8, 11), p=0.3),  # 30%概率模拟低位深度
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])
    
    # 同时应用图像和标签（位深度模拟只影响图像）
    result = transform(image=image, mask=mask)
    
    print(f"图像已应用变换")
    print(f"标签形状: {result['mask'].shape}")
    

def example_3_segmentation_task():
    """示例3: 语义分割任务 - 使用BitDepthSimulationWithLabel"""
    print("\n" + "=" * 60)
    print("示例3: 语义分割任务")
    print("=" * 60)
    
    image = np.random.rand(512, 512, 4).astype(np.float32)
    mask = np.random.randint(0, 3, (512, 512), dtype=np.uint8)
    
    # 使用支持标签的变换（标签保持不变）
    transform = A.Compose([
        BitDepthSimulationWithLabel(bit_depth_range=(8, 11), p=0.5),
        A.RandomCrop(384, 384),
    ])
    
    result = transform(image=image, mask=mask)
    
    print(f"图像变换后范围: [{result['image'].min():.4f}, {result['image'].max():.4f}]")
    print(f"标签唯一值: {np.unique(result['mask'])}")
    

def example_4_different_input_ranges():
    """示例4: 处理不同输入范围"""
    print("\n" + "=" * 60)
    print("示例4: 处理不同输入范围")
    print("=" * 60)
    
    # 场景1: 已归一化到[0, 1]的图像（默认）
    norm_image = np.random.rand(256, 256, 4).astype(np.float32)
    transform1 = BitDepthSimulation(
        bit_depth_range=(8, 10),
        input_max_value=1.0,  # 默认值
        p=1.0
    )
    result1 = transform1(image=norm_image)
    print(f"归一化图像 [0,1] -> 模拟8-10位: 范围 [{result1['image'].min():.4f}, {result1['image'].max():.4f}]")
    
    # 场景2: 原始10-bit图像 [0, 1023]
    raw_10bit = np.random.randint(0, 1024, (256, 256, 4)).astype(np.uint16)
    transform2 = BitDepthSimulation(
        bit_depth_range=(8, 9),
        input_max_value=1023.0,  # 10-bit最大值
        p=1.0
    )
    result2 = transform2(image=raw_10bit)
    print(f"10-bit原始图像 -> 模拟8-9位: dtype={result2['image'].dtype}")
    
    # 场景3: 8-bit图像 [0, 255]
    uint8_image = np.random.randint(0, 256, (256, 256, 4)).astype(np.uint8)
    transform3 = BitDepthSimulation(
        bit_depth_range=(8, 8),  # 保持8-bit（无实际变换）
        input_max_value=255.0,
        p=1.0
    )
    result3 = transform3(image=uint8_image)
    print(f"8-bit图像: dtype={result3['image'].dtype}, 范围 [{result3['image'].min()}, {result3['image'].max()}]")
    

def example_5_batch_processing():
    """示例5: 批次处理（用于PyTorch DataLoader）"""
    print("\n" + "=" * 60)
    print("示例5: 批次处理（PyTorch）")
    print("=" * 60)
    
    import torch
    
    # 模拟batch数据 [B, C, H, W]
    batch_images = torch.rand(4, 4, 256, 256)  # batch=4, 4通道, 256x256
    
    # 创建批次级变换
    batch_transform = RandomBitDepthBatch(
        bit_depth_range=(8, 11),
        probability=0.5
    )
    
    # 应用变换
    simulated_batch = batch_transform(batch_images)
    
    print(f"输入batch形状: {batch_images.shape}")
    print(f"输出batch形状: {simulated_batch.shape}")
    print(f"输入范围: [{batch_images.min():.4f}, {batch_images.max():.4f}]")
    print(f"输出范围: [{simulated_batch.min():.4f}, {simulated_batch.max():.4f}]")
    

def example_6_helper_functions():
    """示例6: 使用辅助函数快速创建变换"""
    print("\n" + "=" * 60)
    print("示例6: 使用辅助函数")
    print("=" * 60)
    
    # 快速创建随机位深度变换
    transform1 = create_bit_depth_transform(
        mode="random",
        probability=0.5,
        input_range="normalized"
    )
    
    # 快速创建固定位深度变换
    transform2 = create_bit_depth_transform(
        mode="fixed",
        fixed_bit_depth=10,
        input_range="10bit"
    )
    
    # 使用预定义配置
    transform3 = BIT_DEPTH_8_TO_11_RANDOM(p=0.3)
    
    image = np.random.rand(256, 256, 4).astype(np.float32)
    
    print(f"随机变换: {transform1}")
    print(f"固定10位: {transform2}")
    print(f"预定义配置: {transform3}")
    

def example_7_integration_with_dataset():
    """示例7: 与CloudDataset集成使用"""
    print("\n" + "=" * 60)
    print("示例7: 与CloudDataset集成")
    print("=" * 60)
    
    from MyCloudSenseNet.benchmark.cloud_dataset import CloudDataset
    
    # 假设有DataFrame
    # df = pd.read_csv("metadata.csv")
    
    # 创建包含位深度模拟的变换管道
    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.0625, rotate_limit=15, p=0.5),
        BitDepthSimulation(
            bit_depth_range=(8, 11),
            p=0.5,  # 50%概率应用
            input_max_value=1.0,  # 假设输入已归一化
        ),
        A.RandomCrop(384, 384, p=1.0),
        A.HorizontalFlip(p=0.5),
    ])
    
    print("变换管道已创建:")
    for t in transform.transforms:
        print(f"  - {t.__class__.__name__}")
    
    # 使用方式
    # dataset = CloudDataset(
    #     x_paths=train_df,
    #     bands=["B02", "B03", "B04", "B08"],
    #     y_paths=train_df,
    #     transforms=transform,  # 包含位深度模拟的变换
    # )


def example_8_visualize_effect():
    """示例8: 可视化位深度模拟效果"""
    print("\n" + "=" * 60)
    print("示例8: 可视化效果")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    
    # 创建渐变图像
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    xx, yy = np.meshgrid(x, y)
    image = (xx + yy) / 2  # 渐变
    image = np.stack([image] * 4, axis=-1).astype(np.float32)  # 4通道
    
    # 不同位深度的模拟
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    axes[0].imshow(image[:, :, :3])
    axes[0].set_title("Original (Float)")
    axes[0].axis('off')
    
    bit_depths = [8, 9, 10, 11, 12]
    for i, bd in enumerate(bit_depths):
        transform = BitDepthSimulation(bit_depth_range=(bd, bd), p=1.0)
        result = transform(image=image)
        
        axes[i+1].imshow(result['image'][:, :, :3])
        axes[i+1].set_title(f"Simulated {bd}-bit")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig("bit_depth_visualization.png")
    print("可视化图已保存: bit_depth_visualization.png")


def example_9_advanced_configuration():
    """示例9: 高级配置"""
    print("\n" + "=" * 60)
    print("示例9: 高级配置")
    print("=" * 60)
    
    # 配置1: 不加噪声（纯量化）
    transform1 = BitDepthSimulation(
        bit_depth_range=(8, 8),
        add_noise=False,
        p=1.0
    )
    
    # 配置2: 强噪声
    transform2 = BitDepthSimulation(
        bit_depth_range=(8, 8),
        add_noise=True,
        noise_factor=1.0,  # 满强度噪声
        p=1.0
    )
    
    # 配置3: 弱噪声
    transform3 = BitDepthSimulation(
        bit_depth_range=(8, 8),
        add_noise=True,
        noise_factor=0.2,  # 20%噪声
        p=1.0
    )
    
    image = np.random.rand(100, 100, 4).astype(np.float32)
    
    result1 = transform1(image=image)
    result2 = transform2(image=image)
    result3 = transform3(image=image)
    
    print(f"无噪声 - 标准差: {result1['image'].std():.6f}")
    print(f"强噪声 - 标准差: {result2['image'].std():.6f}")
    print(f"弱噪声 - 标准差: {result3['image'].std():.6f}")


if __name__ == "__main__":
    # 运行所有示例
    example_1_basic_usage()
    example_2_compose_with_other_transforms()
    example_3_segmentation_task()
    example_4_different_input_ranges()
    example_5_batch_processing()
    example_6_helper_functions()
    example_7_integration_with_dataset()
    # example_8_visualize_effect()  # 需要matplotlib，可选运行
    example_9_advanced_configuration()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
