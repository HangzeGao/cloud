"""
评估脚本 - 在测试集上评估模型性能
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

from models import CloudSenseNet, build_model
from utils import load_config, SegmentationMetrics, sliding_window_inference
from utils.metrics import compute_iou


def load_model(checkpoint_path: str, device: torch.device):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def evaluate_model(model, image_dir, mask_dir, device, mode='sliding_window', config=None):
    """
    评估模型性能
    
    Returns:
        metrics_dict: 包含各项评估指标的字典
    """
    # 获取图像列表
    image_paths = []
    for ext in ['*.jpg', '*.png', '*.tif', '*.tiff']:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths.sort()
    
    metrics = SegmentationMetrics(num_classes=2)
    
    results_per_image = []
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Evaluating"):
            # 加载图像和标注
            image = Image.open(img_path).convert('RGB')
            
            img_name = os.path.basename(img_path)
            name_wo_ext = os.path.splitext(img_name)[0]
            
            # 查找对应的mask
            mask_candidates = [
                os.path.join(mask_dir, img_name),
                os.path.join(mask_dir, name_wo_ext + '.png'),
                os.path.join(mask_dir, name_wo_ext + '.jpg'),
                os.path.join(mask_dir, name_wo_ext + '.tif'),
            ]
            
            mask_path = None
            for mc in mask_candidates:
                if os.path.exists(mc):
                    mask_path = mc
                    break
            
            if mask_path is None:
                print(f"Warning: Mask not found for {img_name}, skipping")
                continue
            
            mask = Image.open(mask_path).convert('L')
            mask_np = np.array(mask)
            mask_np = (mask_np > 127).astype(np.int64)  # 二值化
            mask_tensor = torch.from_numpy(mask_np).long()
            
            # 预处理
            import torchvision.transforms as T
            image_tensor = T.ToTensor()(image).to(device)
            
            # 推理
            if mode == 'sliding_window':
                pred = sliding_window_inference(
                    model,
                    image_tensor,
                    window_size=512,
                    stride=256,
                    num_classes=2
                )
            else:
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                output = model(image_tensor)
                if isinstance(output, dict):
                    pred = output['logits'][0]
                else:
                    pred = output[0]
            
            # 更新指标
            pred_class = torch.argmax(pred, dim=0)
            metrics.update(pred_class.unsqueeze(0), mask_tensor.unsqueeze(0))
            
            # 记录单张图像的IoU
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
    print("="*60)


def get_device(device_arg: str = 'auto', gpu_id: int = 0):
    """获取设备，优先MPS (Apple Silicon)"""
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate CloudSense-Net')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to test images')
    parser.add_argument('--mask_dir', type=str, required=True,
                        help='Path to test masks')
    parser.add_argument('--mode', type=str, default='sliding_window',
                        choices=['sliding_window', 'simple'],
                        help='Inference mode')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'mps', 'cuda', 'cpu'],
                        help='Device to use (auto prefers MPS on Apple Silicon)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (for CUDA)')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save detailed results (JSON)')
    args = parser.parse_args()
    
    # 设置设备
    device = get_device(args.device, args.gpu)
    print(f"Using device: {device}")
    
    # 加载模型
    model, config = load_model(args.checkpoint, device)
    
    # 评估
    print("Starting evaluation...")
    metrics = evaluate_model(
        model,
        args.image_dir,
        args.mask_dir,
        device,
        mode=args.mode,
        config=config
    )
    
    # 打印结果
    print_metrics(metrics)
    
    # 保存详细结果
    if args.save_results:
        import json
        with open(args.save_results, 'w') as f:
            # 移除不可序列化的内容
            save_metrics = {k: v for k, v in metrics.items() if k != 'per_image'}
            json.dump(save_metrics, f, indent=2)
        print(f"\nResults saved to {args.save_results}")


if __name__ == '__main__':
    main()
