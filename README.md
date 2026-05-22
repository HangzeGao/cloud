# CloudSense-Net 🌥️

一个高度可配置的遥感云分割模型，支持灵活的组件组合和通道自适应输入，适用于多种硬件环境（CUDA/MPS/CPU）。

## ✨ 特性

- **多可选架构**：编码器、语义增强、特征融合、解码器均支持多种方案切换
- **真实预训练模型**：基于 timm/torchvision 的 EfficientNet/Swin/ConvNeXt/ResNet/ResNeXt
- **通道自适应输入**：支持3通道RGB和4通道RGB+NIR无缝切换，5种自适应策略
- **统一数据集支持**：自动处理多数据集混合训练（HRC_WHU, RICE2, ONCLOUDN等）
- **任意尺寸推理**：支持滑动窗口、整图、多尺度等多种推理模式
- **跨平台优化**：专为CUDA（NVIDIA）、MPS（Apple Silicon）优化的配置
- **模块化设计**：每个组件都有独立开关，可灵活组合
- **训练优化**：支持混合精度训练、学习率 warmup、分层冻结

## 🏗️ 架构层次

### 0. 通道自适应 (Channel Adaptive) - 5种可选 ⭐NEW

| 方法 | 描述 | 特点 | 适用场景 |
|------|------|------|----------|
| `conv` | 卷积投影 + 通道注意力 | 轻量高效，推荐 | 通用/部署 |
| `attention` | 空间+通道注意力 | 特征提取强 | 精度优先 |
| `physics` | 物理模型 + 可学习调整 | 可解释性强 | 物理约束场景 |
| `hybrid` | conv + physics 门控融合 | 综合性能最佳 | 平衡需求 |
| `transformer` | 轻量ViT全局建模 | 全局关系建模 | 复杂场景 |

### 1. 编码器 (Encoder) - 5种可选

| 类型 | 描述 | 可用变体 | 特点 |
|------|------|----------|------|
| `swin` | Swin Transformer | Tiny/Small/Base/Large | 高精度，全局建模能力强 |
| `convnext` | ConvNeXt | Tiny/Small/Base/Large/XLarge | 平衡推荐，结合CNN效率与Transformer设计 |
| `efficientnet` | EfficientNet | B0-B7, B8, L2 | 轻量高效，适合部署 |
| `resnet` | ResNet | 18/34/50/101 | 经典可靠，快速实验 |
| `resnext` | ResNeXt | 50_32x4d/101_32x8d | 分组卷积改进 |

**EfficientNet 变体选择指南：**
```yaml
# 移动端/边缘设备
model_name: "efficientnet_b0"  # ~5M 参数

# 平衡配置 (推荐)
model_name: "efficientnet_b3"  # ~12M 参数

# 高精度
model_name: "efficientnet_b4"  # ~19M 参数

# 顶级精度 (需要更多显存)
model_name: "efficientnet_b7"  # ~66M 参数
```

### 2. 语义增强模块 (Semantic Enhancement) - 可选开关

- 参考 SkySense++ 的 MSL (Masked Semantic Learning) 设计
- 将像素级语义信息嵌入到特征中
- 支持云分割任务的空间先验学习
- **注意**：需要在训练配置中 `enabled: true` 并在训练时传入 masks

### 3. 特征融合层 (Fusion) - 8种可选

| 类型 | 描述 | 适用场景 |
|------|------|----------|
| `fpn` | 特征金字塔网络 | 多尺度目标检测 |
| `fpnv2` | 改进版FPN | 增强特征表达 |
| `bifpn` | 双向特征金字塔 | 高效多尺度融合 |
| `fastbifpn` | 快速BiFPN | 速度优先 |
| `aspp` | 空洞空间金字塔池化 | 捕获上下文信息 |
| `asppv2` | 改进版ASPP | 增强上下文 |
| `lightaspp` | 轻量级ASPP | 边缘设备 |
| `none` | 不使用融合 | 快速实验 |

### 4. 解码器 (Decoder) - 5种可选

| 类型 | 描述 | 特点 |
|------|------|------|
| `unetpp` | UNet++ | 高精度，嵌套跳跃连接 |
| `deeplabv3plus` | DeepLabV3+ | 经典语义分割架构 |
| `segformer` | SegFormer | 轻量高效，推荐方案 |
| `upernet` | UperNet | 金字塔池化融合 |
| `simple` | 简单解码器 | 快速baseline |

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt

# 如果需要使用 EfficientNet/Swin/ConvNeXt 等 timm 模型
pip install timm
```

### 准备数据

使用上层 `Data/` 目录的统一数据集：

```
../Data/
├── Unified_Cloud_Dataset/    # 统一数据集目录
│   ├── HRC_WHU/
│   ├── RICE2/
│   ├── ONCLOUDN/
│   ├── CLOUD38/
│   └── CLOUD95/
└── cloud_dataset_loader.py   # 统一数据集加载器
```

### 环境配置

选择适合你硬件的配置：

```bash
# Apple Silicon (M1/M2/M3)
python train.py --config configs/cloudseg_mps.yaml --device mps

# NVIDIA GPU
python train.py --config configs/cloudseg_cuda.yaml --device cuda

# 通用配置
python train.py --config configs/cloudseg_base.yaml
```

### 配置文件示例

```yaml
model:
  # 通道自适应（支持3/4通道输入）
  channel_adaptive:
    enabled: true
    method: conv           # conv/attention/physics/hybrid/transformer
    use_attention: true
    out_channels: 4
  
  encoder:
    type: "efficientnet"
    enabled: true
    efficientnet:
      model_name: "efficientnet_b4"  # B0-B7 可选
      pretrained: true
  
  semantic_enhancement:
    enabled: true
  
  fusion:
    enabled: true
    type: "fpn"
  
  decoder:
    type: "segformer"
    enabled: true

training:
  # 学习率 warmup
  scheduler:
    type: "cosine"
    warmup_epochs: 5      # 前5个epoch线性增加学习率
  
  # 混合精度训练 (CUDA)
  amp:
    enabled: true
    dtype: "float16"
  
  # 分层学习率
  optimizer:
    lr: 1e-4
    backbone_lr_mult: 0.1  # backbone使用0.1倍学习率

data:
  # 是否使用NIR通道（true=4通道, false=3通道）
  use_nir: true
  
  # 归一化方法
  normalize: "percentile"  # percentile/minmax/standard/reflectance/none
  
  # 多数据集混合
  datasets:
    - name: "hrcwhu"
    - name: "rice2"
    - name: "oncloudn"
```

### 训练模型

```bash
# 快速开发测试（小数据集）
python train.py --config configs/cloudseg_mps.yaml --dev-run

# 完整训练
python train.py --config configs/cloudseg_mps.yaml --device mps

# 使用 AMP 混合精度训练 (CUDA推荐)
python train.py --config configs/cloudseg_cuda.yaml --device cuda

# 恢复训练
python train.py --config configs/cloudseg_mps.yaml --resume experiments/best_model.pth
```

### 模型评估

```bash
python eval.py \
    --checkpoint experiments_mps/CloudSenseNet_MPS/best_model.pth \
    --image_dir ../Data/test/images \
    --mask_dir ../Data/test/masks \
    --mode sliding_window \
    --device mps
```

### 推理测试

```bash
# 单张图像推理（自动处理3/4通道）
python inference.py \
    --checkpoint experiments_mps/CloudSenseNet_MPS/best_model.pth \
    --input image.jpg \
    --output results/ \
    --mode sliding_window \
    --device mps

# 批量推理
python inference.py \
    --checkpoint experiments_mps/CloudSenseNet_MPS/best_model.pth \
    --input ../Data/test/images/ \
    --output results/ \
    --mode multi_scale
```

## 📊 预定义配置方案

### 配置文件说明

| 配置 | 适用环境 | 特点 |
|------|----------|------|
| `cloudseg_mps.yaml` | Apple Silicon | batch_size=4, num_workers=2, AMP关闭 |
| `cloudseg_cuda.yaml` | NVIDIA GPU | batch_size=16, AMP开启, 支持更大模型 |
| `cloudseg_base.yaml` | 通用环境 | 平衡配置，8GB显存可运行 |

### 推荐配置组合

**高精度方案：**
```yaml
encoder:
  type: swin
  swin:
    model_name: "swin_base_patch4_window7_224"
    pretrained: true
channel_adaptive: hybrid
semantic_enhancement: enabled
fusion: fpnv2
decoder: unetpp
```

**速度优先方案：**
```yaml
encoder:
  type: efficientnet
  efficientnet:
    model_name: "efficientnet_b0"
    pretrained: true
channel_adaptive: conv
semantic_enhancement: disabled
fusion: lightaspp
decoder: segformer
```

**平衡推荐方案 ⭐：**
```yaml
encoder:
  type: convnext
  convnext:
    model_name: "convnext_tiny"
    pretrained: true
    drop_path_rate: 0.1
channel_adaptive: conv
semantic_enhancement: enabled
fusion: bifpn
decoder: segformer
```

## 🔧 高级功能

### 学习率 Warmup

在训练开始时线性增加学习率，有助于稳定训练：

```yaml
training:
  scheduler:
    type: "cosine"  # 或 "cosine_warmup"
    warmup_epochs: 5  # 前5个epoch warmup
    T_0: 10
    T_mult: 2
```

### 混合精度训练 (AMP)

CUDA 环境下可显著加速训练并节省显存：

```yaml
training:
  amp:
    enabled: true
    dtype: "float16"  # float16 或 bfloat16 (Ampere+)
```

### 分层冻结

微调时冻结 backbone 部分层：

```python
from models import CloudSenseNet

model = CloudSenseNet(config)

# 冻结全部 backbone
model.freeze_encoder(freeze_blocks=-1)

# 冻结前2个stage (适用于ResNet/EfficientNet等)
model.freeze_encoder(freeze_blocks=2)
```

支持的冻结粒度：
- **ResNet/ResNeXt**: stem + layer1-4 (5个部分)
- **EfficientNet**: stage 0-3 (通过 blocks 索引)
- **Swin**: layers 0-3
- **ConvNeXt**: stages 0-3

### 通道自适应详解

#### 为什么需要通道自适应？

- **训练时**：使用真实的4通道数据（RGB+NIR）
- **推理时**：用户可能只有3通道RGB图像
- **解决方案**：自动适应输入通道数，无需修改代码

#### 使用方法

```yaml
model:
  channel_adaptive:
    enabled: true
    method: hybrid      # 根据需求选择
```

无需修改推理代码，模型自动处理：
```python
# 3通道输入
output = model.predict(torch.randn(1, 3, 512, 512))

# 4通道输入
output = model.predict(torch.randn(1, 4, 512, 512))
```

#### 方法选择建议

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 部署/移动端 | `conv` | 计算量最小 |
| 精度要求高 | `hybrid` | 综合利用多种策略 |
| 需要可解释性 | `physics` | 基于物理模型 |
| 复杂遥感场景 | `transformer` | 全局建模能力强 |

## 📝 代码示例

### 快速创建模型

```python
from models import create_model_from_preset

# 使用预设快速创建（支持 timm 模型）
model = create_model_from_preset('balanced', num_classes=2)
model.print_architecture()

# 查看是否使用了预训练权重
info = model.get_model_info()
print(f"Encoder: {info['encoder_type']}")
print(f"Total params: {info['total_params']:,}")
```

### 自定义架构与 Backbone 变体

```python
from models import CloudSenseNet
from utils import load_config

# 加载配置
config = load_config('configs/cloudseg_mps.yaml')

# 切换为 EfficientNet-B4（更多参数，更高精度）
config['model']['encoder']['type'] = 'efficientnet'
config['model']['encoder']['efficientnet'] = {
    'model_name': 'efficientnet_b4',  # B0, B1, B2, B3, B4, B5, B6, B7
    'pretrained': True
}

# 或切换为 ResNeXt
config['model']['encoder']['type'] = 'resnext'
config['model']['encoder']['resnext'] = {
    'model_name': 'resnext50_32x4d',
    'pretrained': True
}

# 创建模型
model = CloudSenseNet(config)
model.print_architecture()

# 启用 warmup 和 AMP (CUDA)
config['training']['scheduler']['warmup_epochs'] = 5
config['training']['amp']['enabled'] = True
```

### 推理示例

```python
import torch
from models import CloudSenseNet, build_model
from utils import load_config

# 加载配置和模型
config = load_config('configs/cloudseg_mps.yaml')
model = build_model(config)

# 加载权重
checkpoint = torch.load('experiments/best_model.pth', map_location='mps')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to('mps')
model.eval()

# 推理（自动适应3/4通道）
with torch.no_grad():
    # 可以是3通道或4通道
    image = torch.randn(1, 3, 1024, 1024).to('mps')  # 或 4通道
    pred = model.predict(image)
```

## 📈 支持的推理模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| `sliding_window` | 滑动窗口推理 | 超大图像，显存有限 |
| `whole` | 整图推理 | 小图像，速度优先 |
| `multi_scale` | 多尺度TTA | 精度优先 |
| `simple` | 直接推理 | 快速测试 |

## 📚 目录结构

```
MyCloudSense/
├── configs/                      # 配置文件
│   ├── cloudseg_base.yaml        # 通用配置
│   ├── cloudseg_mps.yaml         # Apple Silicon优化
│   └── cloudseg_cuda.yaml        # NVIDIA GPU优化
├── models/                       # 模型定义
│   ├── backbones/                # 编码器
│   │   ├── efficientnet_backbone.py  # ⭐ timm 真实模型
│   │   ├── swin_backbone.py          # ⭐ timm 真实模型
│   │   ├── convnext_backbone.py      # ⭐ timm 真实模型
│   │   ├── resnet_backbone.py        # torchvision 真实模型
│   │   └── base_backbone.py
│   ├── decoders/                 # 解码器
│   ├── necks/                    # 特征融合与通道自适应
│   │   ├── channel_adaptive.py   # ⭐ 5种自适应方法
│   │   ├── fpn_fusion.py
│   │   ├── bifpn_fusion.py       # 含 FastBiFPN
│   │   └── aspp_fusion.py       # 含 ASPPv2, LightASPP
│   ├── heads/                    # 损失函数
│   ├── cloudseg_model.py         # 主模型
│   └── builder.py                # 模型构建器
├── data/                         # 数据加载
├── utils/                        # 工具函数
│   ├── inference_utils.py        # 推理工具
│   └── metrics.py                # 评估指标
├── train.py                      # 训练脚本 (支持 AMP/warmup)
├── eval.py                       # 评估脚本
├── inference.py                  # 推理脚本
└── demo.py                       # 演示脚本
```

## 🎯 最新更新

### v2.0 主要更新

- ✅ **真实预训练模型**：EfficientNet/Swin/ConvNeXt 基于 timm，自动加载 ImageNet 权重
- ✅ **Encoder 变体支持**：EfficientNet B0-B7, Swin Tiny/Base/Large, ConvNeXt 全系列
- ✅ **新增 ResNeXt**：分组卷积改进的 ResNet
- ✅ **语义增强训练**：修复训练时不传入 masks 的问题
- ✅ **学习率 Warmup**：支持线性 warmup 稳定训练初期
- ✅ **AMP 混合精度**：CUDA 环境下自动混合精度训练
- ✅ **分层冻结**：支持按 stage 冻结 backbone
- ✅ **扩展 Fusion 变体**：FPNv2, FastBiFPN, ASPPv2, LightASPP
- ✅ **Builder 注册**：所有变体可通过配置选择

## 🤝 参考

- SkySense++: [kang-wu/SkySensePlusPlus](https://github.com/kang-wu/SkySensePlusPlus)
- timm: [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- RICE2 Dataset: 云检测经典数据集
- HRC_WHU Dataset: 高分辨率云分割数据集
- Sentinel-2: 多光谱卫星数据源

## 📄 License

MIT License
