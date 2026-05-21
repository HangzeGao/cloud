#!/usr/bin/env python3
"""
遥感云分割数据集可视化与管理脚本
提供数据集统计、样本可视化、交互式浏览等功能
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import rasterio
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class DatasetVisualizer:
    """数据集可视化器"""

    # 配色方案
    COLORS = {
        'oncloudn': '#1f77b4',      # 蓝色
        'hrcwhu': '#ff7f0e',        # 橙色
        'rice2': '#2ca02c',         # 绿色
        'cloud38': '#d62728',       # 红色
        'cloud95': '#9467bd'        # 紫色
    }

    # 数据集显示名称
    DATASET_NAMES = {
        'oncloudn': 'OnCloudN (Sentinel-2)',
        'hrcwhu': 'HRC_WHU (Landsat 8)',
        'rice2': 'RICE2 (Multi-source)',
        'cloud38': '38-Cloud (Landsat 8)',
        'cloud95': '95-Cloud (Landsat 8)'
    }

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.metadata_path = self.data_dir / "metadata.json"

        # 加载元数据
        self.metadata = self._load_metadata()

        # 扫描数据集
        self.samples = self._scan_samples()

        # 统计信息缓存
        self._stats_cache = None

    def _load_metadata(self) -> Dict:
        """加载元数据"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def _scan_samples(self) -> List[Dict]:
        """扫描所有样本"""
        samples = []

        if not self.images_dir.exists():
            print(f"警告: 图像目录不存在: {self.images_dir}")
            return samples

        for sample_dir in self.images_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            sample_id = sample_dir.name
            dataset_name = sample_id.split('_')[0]

            # 检查标签
            label_path = self.labels_dir / f"{sample_id}.tif"
            if not label_path.exists():
                continue

            # 统计波段
            bands = [f.stem for f in sample_dir.glob("*.tif")]

            samples.append({
                'id': sample_id,
                'dataset': dataset_name,
                'image_dir': sample_dir,
                'label_path': label_path,
                'bands': bands
            })

        return samples

    def compute_statistics(self) -> Dict:
        """计算数据集统计信息"""
        if self._stats_cache is not None:
            return self._stats_cache

        stats = {
            'total_samples': len(self.samples),
            'datasets': defaultdict(lambda: {'count': 0, 'bands': set()}),
            'class_distribution': {0: 0, 1: 0},
            'size_distribution': defaultdict(int),
            'samples_by_dataset': defaultdict(list)
        }

        print("正在计算数据集统计信息...")

        for sample in self.samples:
            dataset = sample['dataset']
            stats['datasets'][dataset]['count'] += 1
            stats['datasets'][dataset]['bands'].update(sample['bands'])
            stats['samples_by_dataset'][dataset].append(sample['id'])

            # 读取标签统计类别分布
            try:
                with rasterio.open(sample['label_path']) as src:
                    label = src.read(1)
                    unique, counts = np.unique(label, return_counts=True)
                    for val, count in zip(unique, counts):
                        if val in stats['class_distribution']:
                            stats['class_distribution'][val] += count
            except Exception as e:
                pass

        # 转换set为list以便JSON序列化
        for dataset in stats['datasets']:
            stats['datasets'][dataset]['bands'] = sorted(list(stats['datasets'][dataset]['bands']))

        self._stats_cache = stats
        return stats

    def plot_dataset_overview(self, save_path: Optional[str] = None):
        """绘制数据集概览图"""
        stats = self.compute_statistics()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('遥感云分割数据集概览', fontsize=16, fontweight='bold')

        # 1. 样本数量分布 (左上)
        ax1 = axes[0, 0]
        datasets = list(stats['datasets'].keys())
        counts = [stats['datasets'][d]['count'] for d in datasets]
        colors = [self.COLORS.get(d, '#333333') for d in datasets]
        display_names = [self.DATASET_NAMES.get(d, d) for d in datasets]

        bars = ax1.barh(display_names, counts, color=colors, alpha=0.8)
        ax1.set_xlabel('样本数量')
        ax1.set_title('各数据集样本数量分布')

        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                    f'{count}', va='center', fontsize=10)

        # 2. 类别分布饼图 (右上)
        ax2 = axes[0, 1]
        class_dist = stats['class_distribution']
        total_pixels = sum(class_dist.values())
        if total_pixels > 0:
            sizes = [class_dist[0], class_dist[1]]
            labels = [f'非云 (Non-cloud)\n{class_dist[0]/total_pixels*100:.1f}%',
                     f'云 (Cloud)\n{class_dist[1]/total_pixels*100:.1f}%']
            colors_pie = ['#87CEEB', '#FFB6C1']
            explode = (0.02, 0.05)

            ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax2.set_title('整体类别分布 (像素级别)')

        # 3. 波段分布 (左下)
        ax3 = axes[1, 0]
        band_counts = defaultdict(int)
        for dataset, info in stats['datasets'].items():
            for band in info['bands']:
                band_counts[band] += info['count']

        bands = sorted(band_counts.keys())
        band_counts_list = [band_counts[b] for b in bands]
        band_colors = ['#4169E1', '#32CD32', '#DC143C', '#FF8C00'][:len(bands)]

        bars3 = ax3.bar(bands, band_counts_list, color=band_colors, alpha=0.8)
        ax3.set_xlabel('波段')
        ax3.set_ylabel('样本数量')
        ax3.set_title('各波段可用性统计')

        # 添加数值标签
        for bar, count in zip(bars3, band_counts_list):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontsize=9)

        # 4. 统计摘要 (右下)
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = f"""
        数据集统计摘要
        =================

        总样本数: {stats['total_samples']:,}

        各数据集详情:
        """
        for dataset in datasets:
            count = stats['datasets'][dataset]['count']
            band_list = ', '.join(stats['datasets'][dataset]['bands'])
            summary_text += f"\n  • {self.DATASET_NAMES.get(dataset, dataset)}:"
            summary_text += f"\n    - 样本数: {count}"
            summary_text += f"\n    - 可用波段: {band_list}"

        summary_text += f"\n\n类别分布 (像素):"
        summary_text += f"\n  • 非云: {class_dist[0]:,} ({class_dist[0]/total_pixels*100:.2f}%)"
        summary_text += f"\n  • 云: {class_dist[1]:,} ({class_dist[1]/total_pixels*100:.2f}%)"

        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"概览图已保存: {save_path}")

        plt.show()

    def plot_class_distribution_by_dataset(self, save_path: Optional[str] = None):
        """绘制各数据集的类别分布对比"""
        stats = self.compute_statistics()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('各数据集类别分布对比', fontsize=14, fontweight='bold')

        datasets = list(stats['datasets'].keys())
        display_names = [self.DATASET_NAMES.get(d, d) for d in datasets]

        # 计算每个数据集的类别分布
        class_ratios = {d: {0: 0, 1: 0} for d in datasets}

        print("正在计算各数据集的类别分布...")
        for sample in self.samples:
            dataset = sample['dataset']
            try:
                with rasterio.open(sample['label_path']) as src:
                    label = src.read(1)
                    unique, counts = np.unique(label, return_counts=True)
                    for val, count in zip(unique, counts):
                        if val in [0, 1]:
                            class_ratios[dataset][val] += count
            except:
                pass

        # 转换为百分比
        cloud_percentages = []
        noncloud_percentages = []

        for dataset in datasets:
            total = class_ratios[dataset][0] + class_ratios[dataset][1]
            if total > 0:
                noncloud_pct = class_ratios[dataset][0] / total * 100
                cloud_pct = class_ratios[dataset][1] / total * 100
            else:
                noncloud_pct = 0
                cloud_pct = 0
            noncloud_percentages.append(noncloud_pct)
            cloud_percentages.append(cloud_pct)

        # 堆叠柱状图
        ax1 = axes[0]
        x = np.arange(len(datasets))
        width = 0.6

        bars1 = ax1.bar(x, noncloud_percentages, width, label='非云 (Non-cloud)', color='#87CEEB', alpha=0.8)
        bars2 = ax1.bar(x, cloud_percentages, width, bottom=noncloud_percentages,
                       label='云 (Cloud)', color='#FFB6C1', alpha=0.8)

        ax1.set_ylabel('百分比 (%)')
        ax1.set_title('各类别占比 (堆叠)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_names, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim([0, 100])

        # 添加数值标签
        for i, (noncloud, cloud) in enumerate(zip(noncloud_percentages, cloud_percentages)):
            ax1.text(i, noncloud/2, f'{noncloud:.1f}%', ha='center', va='center', fontsize=9)
            ax1.text(i, noncloud + cloud/2, f'{cloud:.1f}%', ha='center', va='center', fontsize=9)

        # 分组柱状图 (云的百分比)
        ax2 = axes[1]
        colors = [self.COLORS.get(d, '#333333') for d in datasets]
        bars = ax2.bar(display_names, cloud_percentages, color=colors, alpha=0.8)

        ax2.set_ylabel('百分比 (%)')
        ax2.set_title('云覆盖率对比')
        ax2.set_xticklabels(display_names, rotation=45, ha='right')

        # 添加数值标签和平均线
        mean_cloud = np.mean(cloud_percentages)
        ax2.axhline(y=mean_cloud, color='r', linestyle='--', alpha=0.5, label=f'平均值: {mean_cloud:.1f}%')
        ax2.legend()

        for bar, pct in zip(bars, cloud_percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"类别分布图已保存: {save_path}")

        plt.show()

    def visualize_sample(self, sample_id: str, save_path: Optional[str] = None):
        """可视化单个样本"""
        # 找到样本
        sample = None
        for s in self.samples:
            if s['id'] == sample_id:
                sample = s
                break

        if sample is None:
            print(f"未找到样本: {sample_id}")
            return

        # 读取波段
        bands_data = {}
        available_bands = ['B02', 'B03', 'B04', 'B08']

        for band in available_bands:
            band_path = sample['image_dir'] / f"{band}.tif"
            if band_path.exists():
                with rasterio.open(band_path) as src:
                    bands_data[band] = src.read(1)

        # 读取标签
        with rasterio.open(sample['label_path']) as src:
            label = src.read(1)

        # 创建可视化
        n_bands = len(bands_data)
        fig = plt.figure(figsize=(16, 4 + n_bands * 0.5))
        gs = fig.add_gridspec(2, max(4, n_bands), height_ratios=[1, 1])

        fig.suptitle(f"样本: {sample_id} (数据集: {self.DATASET_NAMES.get(sample['dataset'], sample['dataset'])})",
                    fontsize=14, fontweight='bold')

        # 绘制RGB合成图
        if all(b in bands_data for b in ['B04', 'B03', 'B02']):
            ax_rgb = fig.add_subplot(gs[0, :2])

            # 归一化到0-1
            r = bands_data['B04'].astype(np.float32)
            g = bands_data['B03'].astype(np.float32)
            b = bands_data['B02'].astype(np.float32)

            # 根据数据类型选择合适的归一化方式
            if r.max() > 255:
                r = r / 10000.0
                g = g / 10000.0
                b = b / 10000.0
            else:
                r = r / 255.0
                g = g / 255.0
                b = b / 255.0

            rgb = np.stack([r, g, b], axis=2)
            rgb = np.clip(rgb, 0, 1)

            ax_rgb.imshow(rgb)
            ax_rgb.set_title('RGB 合成图 (B04-B03-B02)')
            ax_rgb.axis('off')

        # 绘制标签
        ax_label = fig.add_subplot(gs[0, 2:])
        label_viz = np.zeros((*label.shape, 3))
        label_viz[label == 0] = [0.53, 0.81, 0.92]  # 浅蓝色 - 非云
        label_viz[label == 1] = [1.0, 0.71, 0.76]  # 粉红色 - 云
        ax_label.imshow(label_viz)
        ax_label.set_title('云标签 (蓝色=非云, 红色=云)')
        ax_label.axis('off')

        # 绘制各个波段
        band_names = {'B02': 'Blue (蓝)', 'B03': 'Green (绿)', 'B04': 'Red (红)', 'B08': 'NIR (近红外)'}

        for idx, (band, data) in enumerate(bands_data.items()):
            ax = fig.add_subplot(gs[1, idx])

            # 归一化显示
            if data.max() > 255:
                display_data = data / 10000.0
            else:
                display_data = data / 255.0
            display_data = np.clip(display_data, 0, 1)

            im = ax.imshow(display_data, cmap='gray' if band != 'B08' else 'viridis')
            ax.set_title(f'{band_names.get(band, band)}\n范围: [{data.min()}, {data.max()}]')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"样本可视化已保存: {save_path}")

        plt.show()

    def interactive_browser(self, dataset_filter: Optional[str] = None):
        """交互式样本浏览器"""
        # 筛选样本
        if dataset_filter:
            samples = [s for s in self.samples if s['dataset'] == dataset_filter]
        else:
            samples = self.samples

        if not samples:
            print("没有可用的样本")
            return

        samples = sorted(samples, key=lambda x: x['id'])
        current_idx = [0]  # 使用list以便在嵌套函数中修改

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#f0f0f0')

        # 创建主显示区域
        ax_main = fig.add_axes([0.05, 0.25, 0.5, 0.7])
        ax_label = fig.add_axes([0.57, 0.25, 0.38, 0.7])
        ax_bands = fig.add_axes([0.05, 0.05, 0.9, 0.15])

        # 按钮区域
        ax_prev = fig.add_axes([0.4, 0.02, 0.08, 0.04])
        ax_next = fig.add_axes([0.52, 0.02, 0.08, 0.04])
        ax_info = fig.add_axes([0.05, 0.02, 0.3, 0.04])

        btn_prev = Button(ax_prev, '◀ 上一个', color='#d0d0d0', hovercolor='#c0c0c0')
        btn_next = Button(ax_next, '下一个 ▶', color='#d0d0d0', hovercolor='#c0c0c0')
        ax_info.axis('off')

        info_text = ax_info.text(0, 0.5, '', fontsize=10, va='center')

        def load_sample(idx):
            """加载并显示样本"""
            sample = samples[idx]

            # 读取RGB
            bands_data = {}
            for band in ['B02', 'B03', 'B04', 'B08']:
                band_path = sample['image_dir'] / f"{band}.tif"
                if band_path.exists():
                    with rasterio.open(band_path) as src:
                        bands_data[band] = src.read(1)

            # 读取标签
            with rasterio.open(sample['label_path']) as src:
                label = src.read(1)

            # 显示RGB
            ax_main.clear()
            if all(b in bands_data for b in ['B04', 'B03', 'B02']):
                r = bands_data['B04'].astype(np.float32)
                g = bands_data['B03'].astype(np.float32)
                b = bands_data['B02'].astype(np.float32)

                if r.max() > 255:
                    r, g, b = r/10000.0, g/10000.0, b/10000.0
                else:
                    r, g, b = r/255.0, g/255.0, b/255.0

                rgb = np.clip(np.stack([r, g, b], axis=2), 0, 1)
                ax_main.imshow(rgb)
            ax_main.set_title(f'样本: {sample["id"]}', fontsize=12, fontweight='bold')
            ax_main.axis('off')

            # 显示标签
            ax_label.clear()
            label_viz = np.zeros((*label.shape, 3))
            label_viz[label == 0] = [0.53, 0.81, 0.92]
            label_viz[label == 1] = [1.0, 0.71, 0.76]
            ax_label.imshow(label_viz)

            # 计算云覆盖率
            cloud_ratio = np.sum(label == 1) / label.size * 100
            ax_label.set_title(f'标签 (云覆盖率: {cloud_ratio:.1f}%)', fontsize=12, fontweight='bold')
            ax_label.axis('off')

            # 显示波段缩略图
            ax_bands.clear()
            ax_bands.axis('off')

            n_bands = len(bands_data)
            if n_bands > 0:
                band_names = list(bands_data.keys())
                for i, (band, data) in enumerate(bands_data.items()):
                    ax_band = fig.add_axes([0.05 + i * (0.9/n_bands), 0.05, 0.9/n_bands - 0.02, 0.15])

                    if data.max() > 255:
                        display_data = data / 10000.0
                    else:
                        display_data = data / 255.0
                    display_data = np.clip(display_data, 0, 1)

                    ax_band.imshow(display_data, cmap='gray')
                    ax_band.set_title(band, fontsize=9)
                    ax_band.set_xticks([])
                    ax_band.set_yticks([])

            # 更新信息
            dataset_name = self.DATASET_NAMES.get(sample['dataset'], sample['dataset'])
            info_text.set_text(f'{idx+1}/{len(samples)} | {dataset_name} | 波段: {", ".join(bands_data.keys())}')

            plt.draw()

        def on_prev(event):
            if current_idx[0] > 0:
                current_idx[0] -= 1
                load_sample(current_idx[0])

        def on_next(event):
            if current_idx[0] < len(samples) - 1:
                current_idx[0] += 1
                load_sample(current_idx[0])

        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)

        # 添加键盘支持
        def on_key(event):
            if event.key == 'left' and current_idx[0] > 0:
                current_idx[0] -= 1
                load_sample(current_idx[0])
            elif event.key == 'right' and current_idx[0] < len(samples) - 1:
                current_idx[0] += 1
                load_sample(current_idx[0])

        fig.canvas.mpl_connect('key_press_event', on_key)

        # 加载第一个样本
        load_sample(0)

        plt.show()

    def compare_datasets(self, sample_count: int = 3, save_path: Optional[str] = None):
        """对比不同数据集的样本"""
        stats = self.compute_statistics()
        datasets = list(stats['datasets'].keys())

        # 每个数据集随机选样本
        fig, axes = plt.subplots(len(datasets), sample_count + 1,
                                figsize=(4 * (sample_count + 1), 4 * len(datasets)))

        if len(datasets) == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('跨数据集样本对比', fontsize=16, fontweight='bold')

        for i, dataset in enumerate(datasets):
            dataset_samples = [s for s in self.samples if s['dataset'] == dataset]
            if len(dataset_samples) == 0:
                continue

            selected = np.random.choice(len(dataset_samples),
                                      min(sample_count, len(dataset_samples)),
                                      replace=False)

            axes[i, 0].text(0.5, 0.5, self.DATASET_NAMES.get(dataset, dataset),
                           ha='center', va='center', fontsize=14, fontweight='bold',
                           transform=axes[i, 0].transAxes)
            axes[i, 0].axis('off')

            for j, idx in enumerate(selected):
                sample = dataset_samples[idx]

                # 读取RGB和标签
                try:
                    with rasterio.open(sample['image_dir'] / "B04.tif") as src:
                        r = src.read(1)
                    with rasterio.open(sample['image_dir'] / "B03.tif") as src:
                        g = src.read(1)
                    with rasterio.open(sample['image_dir'] / "B02.tif") as src:
                        b = src.read(1)
                    with rasterio.open(sample['label_path']) as src:
                        label = src.read(1)

                    # 归一化
                    if r.max() > 255:
                        r, g, b = r/10000.0, g/10000.0, b/10000.0
                    else:
                        r, g, b = r/255.0, g/255.0, b/255.0

                    rgb = np.clip(np.stack([r, g, b], axis=2), 0, 1)

                    # 创建叠加图
                    overlay = rgb.copy()
                    cloud_mask = label == 1
                    overlay[cloud_mask] = overlay[cloud_mask] * 0.5 + np.array([1, 0, 0]) * 0.5

                    ax = axes[i, j + 1]
                    ax.imshow(overlay)

                    cloud_ratio = np.sum(label == 1) / label.size * 100
                    ax.set_title(f'{sample["id"][:20]}...\n云: {cloud_ratio:.1f}%', fontsize=9)
                    ax.axis('off')

                except Exception as e:
                    axes[i, j + 1].text(0.5, 0.5, '读取失败', ha='center', va='center')
                    axes[i, j + 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存: {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description="遥感云分割数据集可视化工具")
    parser.add_argument("data_dir", type=str, help="统一数据集目录路径")
    parser.add_argument("--overview", action="store_true", help="显示数据集概览")
    parser.add_argument("--class-dist", action="store_true", help="显示类别分布")
    parser.add_argument("--visualize", type=str, help="可视化指定样本ID")
    parser.add_argument("--compare", action="store_true", help="对比不同数据集")
    parser.add_argument("--browse", action="store_true", help="交互式浏览")
    parser.add_argument("--dataset", type=str, help="筛选指定数据集")
    parser.add_argument("--save", type=str, help="保存图像路径")

    args = parser.parse_args()

    # 创建可视化器
    visualizer = DatasetVisualizer(args.data_dir)

    if args.overview or not any([args.class_dist, args.visualize, args.compare, args.browse]):
        visualizer.plot_dataset_overview(args.save)

    if args.class_dist:
        visualizer.plot_class_distribution_by_dataset(
            args.save.replace('.png', '_class_dist.png') if args.save else None
        )

    if args.visualize:
        visualizer.visualize_sample(args.visualize, args.save)

    if args.compare:
        visualizer.compare_datasets(
            sample_count=3,
            save_path=args.save.replace('.png', '_compare.png') if args.save else None
        )

    if args.browse:
        visualizer.interactive_browser(dataset_filter=args.dataset)


if __name__ == "__main__":
    main()
