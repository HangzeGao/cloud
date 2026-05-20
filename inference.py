"""
推理脚本 - 支持任意尺寸输入
支持多种推理模式：滑动窗口、整图、多尺度
"""
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CloudSenseNet, build_model
from utils import load_config, sliding_window_inference, multi_scale_inference, whole_image_inference
from data.cloud_dataset import InferenceDataset


def load_model(checkpoint_path: str, device: torch.device):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Best mIoU: {checkpoint.get('best_miou', 'N/A')}")
    
    return model, config


def save_prediction(pred: torch.Tensor, save_path: str, original_size: tuple = None):
    """
    保存预测结果为图像
    
    Args:
        pred: 预测结果 [num_classes, H, W] 或 [H, W]
        save_path: 保存路径
        original_size: 原始尺寸 (W, H)
    """
    if pred.dim() == 3:
        pred = torch.argmax(pred, dim=0)
    
    pred_np = pred.cpu().numpy().astype(np.uint8) * 255  # 二值化，云为255
    
    # 调整到原始尺寸
    if original_size is not None:
        pred_img = Image.fromarray(pred_np)
        pred_img = pred_img.resize(original_size, Image.NEAREST)
        pred_np = np.array(pred_img)
    
    pred_img = Image.fromarray(pred_np)
    pred_img.save(save_path)


def inference_single_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    mode: str = 'sliding_window',
    config: dict = None
) -> torch.Tensor:
    """
    对单张图像进行推理
    
    Args:
        model: 模型
        image_path: 图像路径
        device: 设备
        mode: 推理模式
        config: 配置
        
    Returns:
        prediction: 预测结果
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    # 转换为Tensor
    image_tensor = T.ToTensor()(image).to(device)
    
    # 根据模式选择推理方法
    with torch.no_grad():
        if mode == 'sliding_window':
            inf_cfg = config.get('data', {}).get('inference', {})
            window_size = inf_cfg.get('window_size', 512)
            stride = inf_cfg.get('stride', 256)
            
            pred = sliding_window_inference(
                model,
                image_tensor,
                window_size=window_size,
                stride=stride,
                num_classes=2
            )
        
        elif mode == 'multi_scale':
            inf_cfg = config.get('data', {}).get('inference', {})
            scales = inf_cfg.get('scales', [0.5, 1.0, 1.5])
            
            pred = multi_scale_inference(
                model,
                image_tensor,
                scales=scales,
                num_classes=2,
                flip=True
            )
        
        elif mode == 'whole':
            pred = whole_image_inference(
                model,
                image_tensor,
                num_classes=2,
                max_size=2048
            )
        
        else:
            # 默认直接推理
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            output = model(image_tensor)
            if isinstance(output, dict):
                pred = output['logits'][0]
            else:
                pred = output[0] if output.dim() == 4 else output
    
    return pred, original_size


def get_device(device_arg: str = 'auto', gpu_id: int = 0):
    """获取设备，优先MPS (Apple Silicon)"""
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            print("✅ Using Apple MPS (Metal Performance Shaders)")
            return torch.device('mps')
        elif torch.cuda.is_available():
            print(f"✅ Using CUDA:{gpu_id}")
            return torch.device(f'cuda:{gpu_id}')
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
            print(f"✅ Using CUDA:{gpu_id}")
            return torch.device(f'cuda:{gpu_id}')
        else:
            print("⚠️  CUDA not available, using CPU")
            return torch.device('cpu')
    else:
        print("✅ Using CPU")
        return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='Inference with CloudSense-Net')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--mode', type=str, default='sliding_window',
                        choices=['sliding_window', 'multi_scale', 'whole', 'simple'],
                        help='Inference mode')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'mps', 'cuda', 'cpu'],
                        help='Device to use (auto prefers MPS on Apple Silicon)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id (for CUDA)')
    parser.add_argument('--save_prob', action='store_true',
                        help='Save probability maps instead of binary masks')
    args = parser.parse_args()
    
    # 设置设备
    device = get_device(args.device, args.gpu)
    
    # 加载模型
    model, config = load_model(args.checkpoint, device)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 获取输入文件列表
    if os.path.isdir(args.input):
        import glob
        input_files = []
        for ext in ['*.jpg', '*.png', '*.tif', '*.tiff']:
            input_files.extend(glob.glob(os.path.join(args.input, ext)))
    else:
        input_files = [args.input]
    
    print(f"Found {len(input_files)} images to process")
    
    # 处理每张图像
    for img_path in tqdm(input_files, desc="Inference"):
        try:
            # 推理
            pred, original_size = inference_single_image(
                model,
                img_path,
                device,
                mode=args.mode,
                config=config
            )
            
            # 获取文件名
            filename = os.path.basename(img_path)
            name_wo_ext = os.path.splitext(filename)[0]
            
            if args.save_prob:
                # 保存概率图
                prob = torch.softmax(pred, dim=0)
                cloud_prob = prob[1].cpu().numpy()  # 云类别的概率
                
                # 保存为16位PNG
                prob_img = (cloud_prob * 65535).astype(np.uint16)
                prob_pil = Image.fromarray(prob_img)
                prob_pil = prob_pil.resize(original_size, Image.BILINEAR)
                
                save_path = os.path.join(args.output, f"{name_wo_ext}_prob.png")
                prob_pil.save(save_path)
            else:
                # 保存分割结果
                save_path = os.path.join(args.output, f"{name_wo_ext}_mask.png")
                save_prediction(pred, save_path, original_size)
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"\nInference completed! Results saved to {args.output}")


if __name__ == '__main__':
    main()
