"""
多数据集快速验证脚本

用于快速验证模型在不同数据集上的泛化性能。
从多个数据集中分别采样固定数量的样本，进行批量推理和指标计算。

Usage:
    python validate_multi_dataset.py --checkpoint experiments/best_model.pth \
                                     --config configs/cloudseg_cloudcover_4ch.yaml

Features:
    - 自动检测数据集目录结构
    - 支持原生4通道和RGB+NIR生成两种模式
    - 按数据集分组统计指标
    - 生成对比报告
"""
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model
from utils import load_config, SegmentationMetrics
from data import MultiDatasetSampler, CloudAugmentation
from utils.metrics import compute_iou


def get_device(device_arg: str = 'auto', gpu_id: int = 0):
    """获取设备"""
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device(f'cuda:{gpu_id}')
        else:
            return torch.device('cpu')
    elif device_arg == 'mps':
        return torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    elif device_arg == 'cuda':
        return torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    else:
        return torch.device('cpu')


def validate_multi_dataset(model, config, device):
    """
    多数据集验证
    
    从配置中读取多数据集配置，采样并验证
    """
    multi_cfg = config.get('multi_dataset_eval', {})
    
    if not multi_cfg.get('enabled', False):
        print("Multi-dataset evaluation not enabled in config")
        return None
    
    datasets = multi_cfg.get('datasets', {})
    samples_per_dataset = multi_cfg.get('samples_per_dataset', 20)
    batch_size = multi_cfg.get('eval_batch_size', 4)
    
    # 数据增强配置
    use_nir = config['data'].get('use_nir', True)
    train_cfg = config.get('training', {})
    aug_config = train_cfg.get('augmentation', {'enabled': False})
    aug_config['use_nir'] = use_nir
    transform = CloudAugmentation(aug_config)
    
    # 创建多数据集采样器
    print("\n" + "="*60)
    print("Multi-Dataset Validation Setup")
    print("="*60)
    
    sampler = MultiDatasetSampler(
        dataset_paths=datasets,
        samples_per_dataset=samples_per_dataset,
        target_size=aug_config.get('random_crop_size', [512, 512]),
        use_nir=use_nir,
        transform=transform
    )
    
    # 获取验证数据加载器
    val_loader = sampler.get_validation_loader(batch_size=batch_size)
    
    # 验证
    print("\n" + "="*60)
    print("Starting Multi-Dataset Validation")
    print("="*60)
    
    model.eval()
    
    # 按数据集分组统计
    results_by_dataset = defaultdict(lambda: {
        'predictions': [],
        'masks': [],
        'ious': [],
        'filenames': []
    })
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            filenames = batch['filename']
            dataset_names = batch['dataset']
            
            # 推理
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            preds = torch.argmax(logits, dim=1)
            
            # 按数据集分组
            for i in range(len(dataset_names)):
                dataset_name = dataset_names[i]
                pred = preds[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                
                # 计算IoU
                iou = compute_iou(pred, mask, num_classes=2)
                
                results_by_dataset[dataset_name]['ious'].append(iou)
                results_by_dataset[dataset_name]['filenames'].append(filenames[i])
    
    # 计算统计指标
    print("\n" + "="*60)
    print("Multi-Dataset Validation Results")
    print("="*60)
    
    overall_ious = []
    
    for dataset_name, results in sorted(results_by_dataset.items()):
        ious = results['ious']
        mean_iou = np.mean(ious) if ious else 0
        std_iou = np.std(ious) if ious else 0
        
        overall_ious.extend(ious)
        
        print(f"\n{dataset_name}:")
        print(f"  Samples: {len(ious)}")
        print(f"  Mean IoU: {mean_iou:.4f} (±{std_iou:.4f})")
        print(f"  Min IoU: {np.min(ious):.4f}")
        print(f"  Max IoU: {np.max(ious):.4f}")
    
    # 总体统计
    print("\n" + "-"*60)
    print(f"Overall:")
    print(f"  Total Samples: {len(overall_ious)}")
    print(f"  Mean IoU: {np.mean(overall_ious):.4f} (±{np.std(overall_ious):.4f})")
    print("="*60)
    
    # 保存详细结果
    results_dict = {}
    for dataset_name, results in results_by_dataset.items():
        results_dict[dataset_name] = {
            'mean_iou': float(np.mean(results['ious'])),
            'std_iou': float(np.std(results['ious'])),
            'min_iou': float(np.min(results['ious'])),
            'max_iou': float(np.max(results['ious'])),
            'num_samples': len(results['ious']),
            'samples': [
                {'filename': f, 'iou': float(iou)}
                for f, iou in zip(results['filenames'], results['ious'])
            ]
        }
    
    return results_dict


def main():
    parser = argparse.ArgumentParser(description='Multi-Dataset Validation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/cloudseg_cloudcover_4ch.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'mps', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (for CUDA)')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Override: specify datasets to validate (e.g., RICE2 HRC_WHU)')
    args = parser.parse_args()
    
    # 设置设备
    device = get_device(args.device, args.gpu)
    print(f"Using device: {device}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 如果命令行指定了数据集，覆盖配置
    if args.datasets:
        config['multi_dataset_eval'] = {
            'enabled': True,
            'datasets': {name: f"../Data/{name}" for name in args.datasets},
            'samples_per_dataset': 20,
            'eval_batch_size': 4
        }
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 使用保存的配置或当前配置
    model_config = checkpoint.get('config', config)
    model = build_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. Best mIoU: {checkpoint.get('best_miou', 'N/A')}")
    
    # 执行多数据集验证
    results = validate_multi_dataset(model, config, device)
    
    if results is None:
        print("\nNo results generated. Please check your configuration.")
        return
    
    # 保存结果
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_results}")
    
    # 打印总结
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    for dataset_name, metrics in sorted(results.items()):
        print(f"{dataset_name:15s}: mIoU={metrics['mean_iou']:.4f} (n={metrics['num_samples']})")
    print("="*60)


if __name__ == '__main__':
    main()
