"""
评估脚本 - 在测试集上评估模型性能
支持训练配置对齐，包括数据增强和预处理
"""
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model
from utils import load_config, SegmentationMetrics, sliding_window_inference
from utils.metrics import compute_iou
from data import UnifiedCloudDataset


def load_model(checkpoint_path: str, device: torch.device):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def create_eval_dataset(config, image_dir, mask_dir):
    """
    创建评估数据集，与训练配置对齐

    使用与训练时相同的预处理方式（channel_adaptive, normalize 等）
    """
    data_cfg = config.get('data', {})

    # 获取推理配置
    inference_cfg = data_cfg.get('inference', {})
    mode = inference_cfg.get('mode', 'sliding_window')
    window_size = inference_cfg.get('window_size', 512)
    stride = inference_cfg.get('stride', 256)

    # 获取数据预处理配置
    use_nir = data_cfg.get('use_nir', True)
    normalize = data_cfg.get('normalize', 'percentile')

    print(f"[Eval] Dataset config:")
    print(f"  - use_nir: {use_nir}")
    print(f"  - normalize: {normalize}")
    print(f"  - inference_mode: {mode}")
    print(f"  - window_size: {window_size}")
    print(f"  - stride: {stride}")

    # 创建临时数据集配置（用于单目录评估）
    eval_config = {
        'data': {
            'unified_data_dir': image_dir,  # 实际图像目录
            'datasets': [
                {'name': 'eval_custom'}
            ],
            'use_nir': use_nir,
            'normalize': normalize,
            'inference': inference_cfg,
        }
    }

    return eval_config, mode, window_size, stride


def load_image_with_preprocessing(img_path, config, device):
    """
    加载图像并应用与训练对齐的预处理

    支持:
    - 多通道图像 (RGB, RGB+NIR)
    - 多种归一化方式
    - 配置对齐的通道数
    """
    data_cfg = config.get('data', {})
    use_nir = data_cfg.get('use_nir', True)
    normalize = data_cfg.get('normalize', 'percentile')

    # 加载图像
    image = Image.open(img_path)

    # 处理多通道图像
    if image.mode == 'RGBA':
        # RGBA -> RGB
        image = image.convert('RGB')
    elif image.mode == 'L':
        # 灰度图 -> RGB
        image = image.convert('RGB')

    img_array = np.array(image)

    # 根据 use_nir 配置处理通道
    if use_nir and len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # 需要 4 通道但只有 3 通道 -> 扩展伪 NIR
        # 简单的 NIR 估计：使用绿色通道作为近似
        nir = img_array[:, :, 1:2]  # 绿色通道
        img_array = np.concatenate([img_array, nir], axis=2)
    elif not use_nir and len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        # 只需要 3 通道
        img_array = img_array[:, :, :3]

    # 转换为 tensor [C, H, W]
    if len(img_array.shape) == 2:
        # 单通道灰度
        img_array = np.expand_dims(img_array, axis=2)

    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

    # 归一化
    if normalize == 'percentile':
        # 1-99 百分位数归一化
        for c in range(img_tensor.shape[0]):
            p1, p99 = torch.quantile(img_tensor[c], torch.tensor([0.01, 0.99]))
            img_tensor[c] = (img_tensor[c] - p1) / (p99 - p1 + 1e-8)
    elif normalize == 'minmax':
        # Min-Max 归一化
        img_min = img_tensor.min()
        img_max = img_tensor.max()
        img_tensor = (img_tensor - img_min) / (img_max - img_min + 1e-8)
    elif normalize == 'standard':
        # 标准归一化（假设是 0-255 的图像）
        img_tensor = img_tensor / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img_tensor = (img_tensor - mean) / std
    else:
        # 默认归一化到 [0, 1]
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

    return img_tensor.to(device)


def evaluate_model(model, image_dir, mask_dir, device, config=None, mode_override=None):
    """
    评估模型性能

    Args:
        model: 待评估模型
        image_dir: 图像目录
        mask_dir: 标注目录
        device: 计算设备
        config: 训练配置（用于对齐预处理）
        mode_override: 覆盖推理模式（可选）

    Returns:
        metrics_dict: 包含各项评估指标的字典
    """
    # 获取推理配置
    inference_cfg = {}
    if config is not None:
        data_cfg = config.get('data', {})
        inference_cfg = data_cfg.get('inference', {})

    mode = mode_override or inference_cfg.get('mode', 'sliding_window')
    window_size = inference_cfg.get('window_size', 512)
    stride = inference_cfg.get('stride', 256)

    # 获取图像列表
    image_paths = []
    for ext in ['*.jpg', '*.png', '*.tif', '*.tiff', '*.TIF', '*.TIFF']:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths.sort()

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

    print(f"[Eval] Found {len(image_paths)} images")

    metrics = SegmentationMetrics(num_classes=2)
    results_per_image = []

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Evaluating"):
            # 加载图像和标注
            img_name = os.path.basename(img_path)
            name_wo_ext = os.path.splitext(img_name)[0]

            # 查找对应的 mask
            mask_candidates = [
                os.path.join(mask_dir, img_name),
                os.path.join(mask_dir, name_wo_ext + '.png'),
                os.path.join(mask_dir, name_wo_ext + '.jpg'),
                os.path.join(mask_dir, name_wo_ext + '.tif'),
                os.path.join(mask_dir, name_wo_ext + '_mask.png'),
                os.path.join(mask_dir, name_wo_ext + '_label.png'),
            ]

            mask_path = None
            for mc in mask_candidates:
                if os.path.exists(mc):
                    mask_path = mc
                    break

            if mask_path is None:
                print(f"Warning: Mask not found for {img_name}, skipping")
                continue

            # 加载 mask
            mask = Image.open(mask_path).convert('L')
            mask_np = np.array(mask)
            mask_np = (mask_np > 127).astype(np.int64)
            mask_tensor = torch.from_numpy(mask_np).long()

            # 加载并预处理图像
            image_tensor = load_image_with_preprocessing(img_path, config, device)

            # 确保通道数正确
            expected_channels = 4 if config.get('data', {}).get('use_nir', True) else 3
            if image_tensor.shape[0] != expected_channels:
                print(f"Warning: Image {img_name} has {image_tensor.shape[0]} channels, "
                      f"expected {expected_channels}. Adjusting...")
                if image_tensor.shape[0] < expected_channels:
                    # 复制最后一个通道
                    pad = expected_channels - image_tensor.shape[0]
                    image_tensor = torch.cat([image_tensor, image_tensor[-1:].repeat(pad, 1, 1)], dim=0)
                else:
                    image_tensor = image_tensor[:expected_channels]

            # 推理
            if mode == 'sliding_window':
                pred = sliding_window_inference(
                    model,
                    image_tensor,
                    window_size=window_size,
                    stride=stride,
                    num_classes=2
                )
            elif mode == 'simple':
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)

                # 处理语义增强（验证时不传入 mask）
                if hasattr(model, 'use_semantic_enhancement') and model.use_semantic_enhancement:
                    output = model(image_tensor, None)  # 验证时不使用语义增强
                else:
                    output = model(image_tensor)

                if isinstance(output, dict):
                    pred = output['logits'][0]
                else:
                    pred = output[0]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # 更新指标
            pred_class = torch.argmax(pred, dim=0)

            # 确保尺寸匹配
            if pred_class.shape != mask_tensor.shape:
                import torch.nn.functional as F
                pred_class = F.interpolate(
                    pred_class.unsqueeze(0).unsqueeze(0).float(),
                    size=mask_tensor.shape,
                    mode='nearest'
                ).squeeze().long()

            metrics.update(pred_class.unsqueeze(0), mask_tensor.unsqueeze(0))

            # 记录单张图像的 IoU
            pred_np = pred_class.cpu().numpy()
            image_iou = compute_iou(pred_np, mask_np, num_classes=2)
            results_per_image.append({
                'filename': img_name,
                'iou': image_iou
            })

    # 计算总体指标
    final_metrics = metrics.compute()
    final_metrics['per_image'] = results_per_image

    return final_metrics


def print_metrics(metrics):
    """打印评估指标"""
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Mean IoU:        {metrics['mIoU']:.4f}")
    print(f"Mean Dice:       {metrics['mDice']:.4f}")
    print(f"Pixel Accuracy:  {metrics['pixel_acc']:.4f}")
    print(f"Mean Precision:  {metrics['mean_precision']:.4f}")
    print(f"Mean Recall:     {metrics['mean_recall']:.4f}")
    print(f"Mean F1:         {metrics['mean_f1']:.4f}")
    print("-"*60)
    print("Per-class IoU:")
    for i, iou in enumerate(metrics['iou_per_class']):
        class_name = "Background" if i == 0 else "Cloud"
        print(f"  {class_name}: {iou:.4f}")

    # 显示每张图像的 IoU 统计
    if 'per_image' in metrics and len(metrics['per_image']) > 0:
        ious = [r['iou'] for r in metrics['per_image']]
        print("-"*60)
        print(f"Per-image IoU stats:")
        print(f"  Min: {min(ious):.4f}")
        print(f"  Max: {max(ious):.4f}")
        print(f"  Mean: {np.mean(ious):.4f}")
        print(f"  Std: {np.std(ious):.4f}")

    print("="*60)


def get_device(device_arg: str = 'auto', gpu_id: int = 0):
    """获取设备，优先 MPS (Apple Silicon)"""
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            print("✅ Using Apple MPS (Metal Performance Shaders)")
            return torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            print(f"✅ Using CUDA: {torch.cuda.get_device_name(device)}")
            return device
        else:
            print("⚠️  Using CPU")
            return torch.device('cpu')
    elif device_arg == 'mps':
        if torch.backends.mps.is_available():
            print("✅ Using Apple MPS")
            return torch.device('mps')
        else:
            print("⚠️  MPS not available, using CPU")
            return torch.device('cpu')
    elif device_arg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            print(f"✅ Using CUDA: {torch.cuda.get_device_name(device)}")
            return device
        else:
            print("⚠️  CUDA not available, using CPU")
            return torch.device('cpu')
    else:
        print("⚠️  Using CPU")
        return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate CloudSense-Net - 与训练配置对齐的评估脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用检查点中保存的配置进行评估（推荐）
  python eval.py --checkpoint experiments/best_model.pth \
                 --image_dir ../Data/test/images \
                 --mask_dir ../Data/test/masks

  # 指定推理模式
  python eval.py --checkpoint experiments/best_model.pth \
                 --image_dir ../Data/test/images \
                 --mask_dir ../Data/test/masks \
                 --mode sliding_window

  # 指定设备
  python eval.py --checkpoint experiments/best_model.pth \
                 --image_dir ../Data/test/images \
                 --mask_dir ../Data/test/masks \
                 --device cuda --gpu 0
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to test images')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Path to test masks')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['sliding_window', 'simple'],
                        help='Inference mode (override config if specified)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'mps', 'cuda', 'cpu'],
                        help='Device to use (auto prefers MPS on Apple Silicon)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (for CUDA)')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save detailed results (JSON)')
    args = parser.parse_args()

    print("="*60)
    print("CloudSense-Net Evaluation")
    print("="*60)

    # 设置设备
    device = get_device(args.device, args.gpu)

    # 加载模型和配置
    print("\nLoading model...")
    model, config = load_model(args.checkpoint, device)

    # 打印模型架构信息
    if hasattr(model, 'print_architecture'):
        model.print_architecture()

    # 评估
    print("\nStarting evaluation...")
    print(f"Image dir: {args.image_dir}")
    print(f"Mask dir: {args.mask_dir}")

    metrics = evaluate_model(
        model,
        args.image_dir,
        args.mask_dir,
        device,
        config=config,
        mode_override=args.mode
    )

    # 打印结果
    print_metrics(metrics)

    # 保存详细结果
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            # 移除不可序列化的内容
            save_metrics = {k: v for k, v in metrics.items() if k != 'per_image'}
            # 转换 numpy 类型为 Python 原生类型
            for k, v in save_metrics.items():
                if isinstance(v, np.ndarray):
                    save_metrics[k] = v.tolist()
                elif isinstance(v, np.floating):
                    save_metrics[k] = float(v)
            json.dump(save_metrics, f, indent=2)
        print(f"\nResults saved to {args.save_results}")

    # 保存每张图像的结果
    if args.save_results:
        import json
        per_image_path = args.save_results.replace('.json', '_per_image.json')
        with open(per_image_path, 'w') as f:
            json.dump(metrics['per_image'], f, indent=2)
        print(f"Per-image results saved to {per_image_path}")


if __name__ == '__main__':
    main()
