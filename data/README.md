# 遥感图像云语义分割数据集 - 统一管理与分析

## 数据集概览

本目录包含5个遥感图像云语义分割数据集，经过统一分析后，现已支持转换为统一格式进行管理。

| 数据集 | 样本数 | 图像尺寸 | 波段数 | 数据源 | 格式 |
|--------|--------|---------|--------|--------|------|
| **OnCloudN** | 11,748 | 512×512 | 4 | Sentinel-2 | GeoTIFF |
| **HRC_WHU** | 150 | 1280×720 | 3 (RGB) | Landsat 8 | TIFF |
| **RICE2** | 736 | 512×512 | 3 (RGB) | 多源 | PNG |
| **38-Cloud** | 8,400 (训练) + 9,201 (测试) | 384×384 | 4 | Landsat 8 | TIFF |
| **95-Cloud** | 26,301 | 384×384 | 4 | Landsat 8 | TIFF |
| **合并后总计** | **~46,535** (不含测试集) | - | - | - | - |

---

## 目录结构

### 原始数据集结构

```
Data/
├── OnCloudN/                 # 参考格式
│   ├── images/{sample_id}/   # 子目录包含4个波段TIF
│   └── labels/{sample_id}.tif
├── HRC_WHU/
│   ├── images/{scene}_{n}.tif  # RGB图像
│   └── masks/{scene}_{n}.tif
├── RICE2/
│   ├── images/{n}.png          # RGB图像
│   └── masks/{n}.png
└── kagglehub/
    ├── 38cloud-cloud-segmentation-in-satellite-images/
    │   └── 38-Cloud_training/
    │       ├── train_{band}/   # 各波段分目录存放
    │       └── train_gt/
    └── 95cloud-cloud-segmentation-on-satellite-images/
        └── 95-cloud_training_only_additional_to38-cloud/
            └── train_{band}_additional_to38cloud/
```

---

## 统一转换工具

### 1. 数据集转换脚本: `unify_datasets.py`

将所有数据集转换为统一的 OnCloudN 格式。

#### 使用方式

```bash
# 转换所有数据集
python unify_datasets.py

# 指定输出目录
python unify_datasets.py --output ./Unified_Cloud_Dataset

# 跳过特定数据集
python unify_datasets.py --skip hrcwhu rice2

# 指定自定义路径
python unify_datasets.py \
    --output ./Unified_Cloud_Dataset \
    --oncloudn ./OnCloudN \
    --hrcwhu ./HRC_WHU \
    --rice2 ./RICE2 \
    --cloud38 ./kagglehub/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images/versions/4 \
    --cloud95 ./kagglehub/datasets/sorour/95cloud-cloud-segmentation-on-satellite-images/versions/3
```

#### 输出结构

```
Unified_Cloud_Dataset/
├── images/
│   ├── oncloudn_{id}/          # OnCloudN样本
│   │   ├── B02.tif
│   │   ├── B03.tif
│   │   ├── B04.tif
│   │   └── B08.tif
│   ├── hrcwhu_{scene}_{n}/     # HRC_WHU样本 (4波段)
│   │   ├── B02.tif  # 原Blue
│   │   ├── B03.tif  # 原Green
│   │   ├── B04.tif  # 原Red
│   │   └── B08.tif  # 伪NIR (基于云特征生成)
│   ├── rice2_{n}/              # RICE2样本 (4波段)
│   ├── cloud38_{id}/           # 38-Cloud样本
│   └── cloud95_{id}/           # 95-Cloud样本
├── labels/
│   ├── oncloudn_{id}.tif       # 标签文件 (0=非云, 1=云)
│   ├── hrcwhu_{scene}_{n}.tif
│   ├── rice2_{n}.tif
│   ├── cloud38_{id}.tif
│   └── cloud95_{id}.tif
└── metadata.json               # 统一元数据文件
```

#### 波段映射规则

| 数据源 | 原波段 | 统一命名 | 说明 |
|--------|--------|---------|------|
| OnCloudN | B02, B03, B04, B08 | B02.tif, B03.tif, B04.tif, B08.tif | 保持原样 |
| HRC_WHU | R, G, B | B04.tif, B03.tif, B02.tif | RGB映射 |
| HRC_WHU | **伪NIR** | B08.tif | [基于云特征生成](#伪nir通道生成算法) |
| RICE2 | R, G, B | B04.tif, B03.tif, B02.tif | RGB映射 |
| RICE2 | **伪NIR** | B08.tif | [基于云特征生成](#伪nir通道生成算法) |
| 38-Cloud/95-Cloud | red, green, blue, nir | B04.tif, B03.tif, B02.tif, B08.tif | 标准映射 |

#### 伪NIR通道生成算法

对于RGB数据集（HRC_WHU, RICE2），脚本会自动生成伪NIR (B08) 波段，采用基于云特征突出的算法：

**算法策略:**
1. **云特征识别**: 云具有高亮度、高白度特征，在可见光波段已较亮
2. **亮度增强**: 基于像素亮度进行非线性增强，亮区（云）得到更多增强
3. **白度加权**: 使用RGB标准差评估"白度"，越白的像素（云）增强越多
4. **标签引导**: 利用云标签进行精细调整，云区域进一步增强
5. **植被抑制**: 使用蓝红差异识别植被区域，适度降低NIR值以突出云-植被对比

**公式概览:**
```
NIR = Red × 1.5 × Enhancement × Vegetation_Factor

其中:
  Enhancement = 1 + Brightness^0.7 × 0.5 + Whiteness^2 × 0.8
  Vegetation_Factor = 1 - |Blue - Red| / 255

如果标签可用:
  Enhancement = Enhancement × (Cloud? 1.3 : 0.9)
```

结果归一化到 `uint16` 范围 `[0, 65535]`，与真实Sentinel-2/Landsat NIR波段一致。

#### 标签标准化

所有数据集的标签统一映射为：
- `0` = 非云 (non-cloud)
- `1` = 云 (cloud)

原始数据集中的 `255` 会自动重新映射为 `1`。

---

### 2. 数据集可视化工具: `visualize_dataset.py`

提供数据集统计分析和可视化功能。

#### 功能特性

- **数据集概览**: 样本数量、类别分布、波段可用性
- **类别分布分析**: 各数据集云覆盖率对比
- **样本可视化**: 显示RGB合成图、标签、各波段
- **交互式浏览**: 键盘/按钮导航浏览样本
- **跨数据集对比**: 并排对比不同数据集的样本

#### 使用方式

```bash
# 1. 数据集概览（默认）
python visualize_dataset.py ./Unified_Cloud_Dataset

# 2. 保存概览图
python visualize_dataset.py ./Unified_Cloud_Dataset --overview --save overview.png

# 3. 显示类别分布对比
python visualize_dataset.py ./Unified_Cloud_Dataset --class-dist

# 4. 可视化指定样本
python visualize_dataset.py ./Unified_Cloud_Dataset --visualize oncloudn_adwp

# 5. 交互式浏览（支持键盘左右键切换）
python visualize_dataset.py ./Unified_Cloud_Dataset --browse

# 6. 仅浏览指定数据集
python visualize_dataset.py ./Unified_Cloud_Dataset --browse --dataset oncloudn

# 7. 跨数据集对比
python visualize_dataset.py ./Unified_Cloud_Dataset --compare --save compare.png
```

#### 可视化示例

**概览图包含:**
- 各数据集样本数量柱状图
- 整体类别分布饼图
- 各波段可用性统计
- 数据集详细摘要

**样本可视化包含:**
- RGB合成图像
- 云标签遮罩（蓝色=非云，红色=云）
- 各波段独立显示（带数值范围）

---

### 3. 数据集加载器: `cloud_dataset_loader.py`

用于加载统一格式的数据集，支持 PyTorch DataLoader。

#### 使用方式

```python
from cloud_dataset_loader import UnifiedCloudDataset, get_dataloader

# 示例1: 加载所有数据集的所有波段
dataset = UnifiedCloudDataset(
    "./Unified_Cloud_Dataset",
    split="train",
    bands=["B02", "B03", "B04", "B08"]
)
dataset.print_statistics()

# 示例2: 只加载多光谱数据集 (4波段)
dataset_ms = UnifiedCloudDataset(
    "./Unified_Cloud_Dataset",
    split="train",
    bands=["B02", "B03", "B04", "B08"],
    datasets=["oncloudn", "cloud38", "cloud95"]
)

# 示例3: 只加载RGB数据集 (3波段)
dataset_rgb = UnifiedCloudDataset(
    "./Unified_Cloud_Dataset",
    split="train",
    bands=["B02", "B03", "B04"],
    datasets=["hrcwhu", "rice2"]
)

# 示例4: 快速创建DataLoader
train_loader = get_dataloader(
    "./Unified_Cloud_Dataset",
    split="train",
    batch_size=16,
    bands=["B02", "B03", "B04", "B08"],
    datasets=["oncloudn", "cloud38", "cloud95"],
    augmentation=True
)

# 迭代数据
for images, labels, info in train_loader:
    print(f"Batch shape: {images.shape}")  # (B, C, H, W)
    print(f"Labels shape: {labels.shape}")  # (B, H, W)
    break
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_dir` | str | - | 统一数据集根目录 |
| `split` | str | "train" | train/val/test |
| `bands` | List[str] | ["B02","B03","B04","B08"] | 加载的波段 |
| `datasets` | List[str] | None | 筛选数据集，如 ["oncloudn", "cloud38"] |
| `normalize` | str | "auto" | 归一化策略: auto/reflectance/minmax/standard/percentile/none |
| `norm_params` | Dict | None | 自定义归一化参数 |
| `train_ratio` | float | 0.7 | 训练集比例 |
| `val_ratio` | float | 0.15 | 验证集比例 |
| `random_seed` | int | 42 | 随机种子 |
| `compute_stats` | bool | False | 预计算数据集统计（用于bandwise_standard） |

#### 归一化策略详解

**警告：归一化不能简单粗暴地除以位宽！** 遥感图像具有特定的物理意义和数值范围。

| 归一化模式 | 适用场景 | 原理说明 |
|-----------|---------|---------|
| `"auto"` | **默认推荐** | 自动根据数据源选择：多光谱数据使用TOA反射率归一化（/10000），RGB数据使用DN→反射率映射（/255×0.3） |
| `"reflectance"` | 物理分析 | 强制使用TOA反射率归一化（/10000）。结果为真实物理反射率 [0,1] |
| `"minmax"` | 快速可视化 | 全局Min-Max到[0,1]。丢失物理意义，不推荐训练使用 |
| `"bandwise_minmax"` | 波段对比 | 各波段独立Min-Max归一化 |
| `"standard"` | 神经网络 | 标准分数 (x-mean)/std，使用预定义典型波段统计值 |
| `"bandwise_standard"` | 精确标准化 | 各波段独立标准分数，基于数据集预计算统计。需`compute_stats=True` |
| `"percentile"` | 鲁棒处理 | 1-99百分位数归一化，对异常值（如云边缘、阴影）鲁棒 |
| `"none"` | 原始分析 | 返回原始DN值，用于需要物理量的分析 |

**推荐用法:**

```python
# 1. 通用训练（最常用）
dataset = UnifiedCloudDataset("./Unified_Cloud_Dataset", normalize="auto")

# 2. 物理一致性分析
dataset = UnifiedCloudDataset(
    "./Unified_Cloud_Dataset",
    normalize="reflectance",
    datasets=["oncloudn", "cloud38", "cloud95"]  # 只使用多光谱
)

# 3. 神经网络训练（零均值单位方差）
dataset = UnifiedCloudDataset(
    "./Unified_Cloud_Dataset",
    normalize="bandwise_standard",
    compute_stats=True,  # 预计算统计
    max_samples=100
)

# 4. 数据质量不一（含异常值）
dataset = UnifiedCloudDataset("./Unified_Cloud_Dataset", normalize="percentile")
```

**为什么不能除以65535？**

- uint16遥感数据通常存储的是TOA反射率×10000，有效范围0-10000（代表反射率0-1）
- 除以65535会导致最大有效值仅0.15，严重压缩动态范围，丢失信息
- 正确做法：除以10000，结果直接对应物理反射率

**反归一化:**

```python
dataset = UnifiedCloudDataset("./Unified_Cloud_Dataset", normalize="reflectance")
image_norm, label, info = dataset[0]
image_dn = dataset.denormalize(image_norm, info["dataset"])
# image_dn 恢复为原始DN值范围
```

#### 内置数据增强

```python
from cloud_dataset_loader import RandomFlip, RandomRotate, Compose, ToTensor

# 组合变换
transform = Compose([
    RandomFlip(horizontal=True, vertical=True, p=0.5),
    RandomRotate(angles=[0, 90, 180, 270]),
    ToTensor()
])

dataset = UnifiedCloudDataset(
    "./Unified_Cloud_Dataset",
    split="train",
    transform=transform
)
```

---

## 数据集详细对比

### OnCloudN

- **来源**: Sentinel-2 卫星影像
- **波段**: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
- **空间分辨率**: 10m
- **数据类型**: uint16
- **坐标系统**: EPSG:32736 (UTM Zone 36S)
- **标签**: 0=非云, 1=云
- **特点**: 已是最优格式，具有地理参考信息

### HRC_WHU

- **来源**: Landsat 8 云检测数据集 (HRC_WHU)
- **场景类别**: barren(荒地), snow(雪地), urban(城市), vegetation(植被), water(水体)
- **每个场景**: 30 样本
- **数据类型**: uint8 RGB
- **特点**: 多场景覆盖，适合场景泛化测试

### RICE2

- **来源**: 多源遥感图像 (RICE: Remote Sensing Image Cloud Detection)
- **数据类型**: uint8 PNG
- **特点**: 图像清晰，云边界标注精确

### 38-Cloud + 95-Cloud

- **来源**: Landsat 8 影像
- **波段**: red, green, blue, nir
- **数据类型**: uint16
- **特点**: 
  - 38-Cloud: 基础训练集 (8,400 样本)
  - 95-Cloud: 38-Cloud的扩展集 (26,301 样本)
  - 两数据集无重复样本，可安全合并
  - 包含测试集 9,201 样本（无标签，用于在线评估）

---

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install rasterio numpy pillow torch tqdm
```

### 2. 统一转换数据集

```bash
python unify_datasets.py --output ./Unified_Cloud_Dataset
```

### 3. 验证加载

```bash
python cloud_dataset_loader.py ./Unified_Cloud_Dataset
```

---

## 注意事项

1. **存储空间**: 统一转换后数据集约 50-60GB，请确保有足够磁盘空间
2. **38-Cloud测试集**: 测试集 (9,201 样本) 无标签，转换脚本会跳过
3. **RGB映射**: HRC_WHU 和 RICE2 为 RGB 图像，映射为多波段后缺少 NIR (B08) 波段
4. **归一化**: 加载器会自动根据数据源类型进行归一化 (uint16/10000, uint8/255)

---

## 引用

如果您使用这些数据集，请引用原始论文：

- **OnCloudN**: (根据具体论文添加)
- **HRC_WHU**: Li et al., "HRC_WHU: A Benchmark Dataset for Cloud Detection"
- **RICE2**: (根据具体论文添加)
- **38-Cloud & 95-Cloud**: Mohajerani et al., "Cloud Detection Algorithm for Remote Sensing Images"

---

## 许可

各数据集遵循其原始发布者的许可协议，请在使用前确认相关许可条款。
