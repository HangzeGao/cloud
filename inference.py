"""
CloudSense-Net 推理脚本
========================

支持任意尺寸输入的云层分割推理，提供多种推理模式和图像处理策略。

主要功能:
- 滑动窗口推理: 适用于大图像，分块处理
- 多尺度推理 (TTA): 多尺度融合提高精度
- 整图推理: 直接处理完整图像
- 伪NIR通道生成: 从RGB生成近红外通道
- 自适应归一化: 智能图像归一化处理

使用示例:
    # 基础推理
    python inference.py --checkpoint model.pth --input image.jpg --output results/

    # 使用自动归一化推荐
    python inference.py --checkpoint model.pth --input image.jpg --output results/ --auto_norm

    # 指定归一化方法
    python inference.py --checkpoint model.pth --input image.jpg --output results/ --normalization robust

    # 多尺度推理
    python inference.py --checkpoint model.pth --input image.jpg --output results/ --mode multi_scale
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Union

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model
from utils import load_config
from utils.nir_generator import generate_pseudo_nir
from utils.inference_utils import (
    sliding_window_inference,
    multi_scale_inference,
    whole_image_inference,
    apply_inference_normalization,
    get_normalization_recommendation
)


# =============================================================================
# 常量定义
# =============================================================================

SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
DEFAULT_WINDOW_SIZE = 512
DEFAULT_STRIDE = 256
DEFAULT_MAX_SIZE = 2048


# =============================================================================
# 设备管理
# =============================================================================

class DeviceManager:
    """设备管理器 - 处理MPS/CUDA/CPU设备选择和配置"""
    
    PRIORITY_ORDER = ['mps', 'cuda', 'cpu']
    
    @staticmethod
    def get_device(preference: str = 'auto', gpu_id: int = 0) -> torch.device:
        """
        获取计算设备
        
        优先级: MPS (Apple Silicon) > CUDA > CPU
        
        Args:
            preference: 设备偏好 ('auto', 'mps', 'cuda', 'cpu')
            gpu_id: CUDA GPU ID
            
        Returns:
            torch.device: 选定的计算设备
        """
        if preference == 'auto':
            preference = DeviceManager._detect_best_device()
        
        device = DeviceManager._try_create_device(preference, gpu_id)
        DeviceManager._print_device_info(device)
        
        return device
    
    @staticmethod
    def _detect_best_device() -> str:
        """自动检测最佳可用设备"""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    @staticmethod
    def _try_create_device(device_type: str, gpu_id: int) -> torch.device:
        """尝试创建设备，失败则回退"""
        if device_type == 'mps':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            print("⚠️  MPS not available, falling back to CUDA/CPU")
            return DeviceManager._try_create_device('cuda', gpu_id)
            
        elif device_type == 'cuda':
            if torch.cuda.is_available():
                return torch.device(f'cuda:{gpu_id}')
            print("⚠️  CUDA not available, falling back to CPU")
            return torch.device('cpu')
            
        return torch.device('cpu')
    
    @staticmethod
    def _print_device_info(device: torch.device) -> None:
        """打印设备信息"""
        if device.type == 'mps':
            print("✅ Using Apple MPS (Metal Performance Shaders)")
        elif device.type == 'cuda':
            print(f"✅ Using CUDA:{device.index}")
            print(f"   Device: {torch.cuda.get_device_name(device)}")
        else:
            print("⚠️  Using CPU")


# =============================================================================
# 模型加载
# =============================================================================

class ModelLoader:
    """模型加载器"""
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        device: torch.device
    ) -> Tuple[torch.nn.Module, dict]:
        """
        加载训练好的模型
        
        Args:
            checkpoint_path: 检查点文件路径
            device: 目标设备
            
        Returns:
            Tuple[model, config]: 加载的模型和配置
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})
        
        # 构建并加载模型
        model = build_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # 打印信息
        print(f"✓ Loaded model from {checkpoint_path}")
        print(f"  Best mIoU: {checkpoint.get('best_miou', 'N/A')}")
        
        return model, config


# =============================================================================
# 图像处理
# =============================================================================

class ImageProcessor:
    """图像处理器 - 处理图像加载、预处理和通道生成"""
    
    def __init__(self, device: torch.device, config: Optional[dict] = None):
        self.device = device
        self.config = config or {}
        self.data_cfg = self.config.get('data', {})
    
    def load_image(self, image_path: str) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        加载图像并返回原始尺寸
        
        Args:
            image_path: 图像路径
            
        Returns:
            Tuple[image, original_size]: 图像对象和原始尺寸(W, H)
        """
        image = Image.open(image_path)
        original_size = image.size  # (W, H)
        return image, original_size
    
    def prepare_tensor(
        self,
        image: Image.Image,
        use_nir: bool = False,
        nir_method: str = 'ensemble'
    ) -> torch.Tensor:
        """
        将PIL图像转换为模型输入张量
        
        Args:
            image: PIL图像
            use_nir: 是否使用NIR通道
            nir_method: 伪NIR生成方法
            
        Returns:
            torch.Tensor: 处理后的图像张量 [C, H, W]
        """
        image_mode = image.mode
        
        if use_nir:
            return self._prepare_4channel_image(image, image_mode, nir_method)
        else:
            return self._prepare_3channel_image(image)
    
    def _prepare_3channel_image(self, image: Image.Image) -> torch.Tensor:
        """准备3通道RGB图像"""
        image = image.convert('RGB')
        tensor = T.ToTensor()(image).to(self.device)
        return tensor
    
    def _prepare_4channel_image(
        self,
        image: Image.Image,
        image_mode: str,
        nir_method: str
    ) -> torch.Tensor:
        """
        准备4通道RGB+NIR图像
        
        如果输入已有4通道(RGBA)，使用Alpha作为NIR。
        否则从RGB生成伪NIR通道。
        """
        if image_mode == 'RGBA':
            # 已有Alpha通道，直接使用
            return self._load_rgba_as_nir(image)
        elif image_mode in ('RGB', 'P'):
            # 从RGB生成伪NIR
            return self._generate_pseudo_nir_from_rgb(image, nir_method)
        else:
            # 其他格式，转换为RGB后生成NIR
            return self._generate_pseudo_nir_from_rgb(image.convert('RGB'), nir_method)
    
    def _load_rgba_as_nir(self, image: Image.Image) -> torch.Tensor:
        """加载RGBA图像，将Alpha作为NIR通道"""
        image_array = np.array(image)
        rgb = image_array[:, :, :3]
        nir = image_array[:, :, 3:4]
        image_4ch = np.concatenate([rgb, nir], axis=2)
        tensor = torch.from_numpy(image_4ch).permute(2, 0, 1).float().to(self.device) / 255.0
        return tensor
    
    def _generate_pseudo_nir_from_rgb(
        self,
        image: Image.Image,
        nir_method: str
    ) -> torch.Tensor:
        """从RGB图像生成伪NIR通道"""
        image = image.convert('RGB')
        image_array = np.array(image)
        
        # 生成伪NIR
        nir = generate_pseudo_nir(image_array, method=nir_method)
        
        # 组合4通道
        image_4ch = np.stack([
            image_array[:, :, 0],  # R
            image_array[:, :, 1],  # G
            image_array[:, :, 2],  # B
            nir.astype(np.uint8)    # 伪NIR
        ], axis=2)
        
        tensor = torch.from_numpy(image_4ch).permute(2, 0, 1).float().to(self.device) / 255.0
        return tensor
    
    def apply_normalization(
        self,
        tensor: torch.Tensor,
        method: Optional[str] = None,
        auto_recommend: bool = False
    ) -> Tuple[torch.Tensor, Optional[str]]:
        """
        应用推理归一化
        
        Args:
            tensor: 输入张量
            method: 归一化方法
            auto_recommend: 是否自动推荐方法
            
        Returns:
            Tuple[tensor, used_method]: 归一化后的张量和实际使用的方法
        """
        if auto_recommend and method is None:
            recommendation = get_normalization_recommendation(tensor)
            method = recommendation['method']
            print(f"  [Auto] Normalization: {method} - {recommendation['reason']}")
        
        if method:
            norm_params = self.data_cfg.get('inference_norm_params', {})
            tensor = apply_inference_normalization(tensor, method=method, **norm_params)
            return tensor, method
        
        return tensor, None


# =============================================================================
# 推理执行
# =============================================================================

class InferenceExecutor:
    """推理执行器"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: dict
    ):
        self.model = model
        self.device = device
        self.config = config
        self.inf_cfg = config.get('inference', {})
    
    def run(
        self,
        image_tensor: torch.Tensor,
        mode: str = 'sliding_window'
    ) -> torch.Tensor:
        """
        执行推理
        
        Args:
            image_tensor: 输入图像张量
            mode: 推理模式
            
        Returns:
            torch.Tensor: 预测结果
        """
        with torch.no_grad():
            if mode == 'sliding_window':
                return self._sliding_window_inference(image_tensor)
            elif mode == 'multi_scale':
                return self._multi_scale_inference(image_tensor)
            elif mode == 'whole':
                return self._whole_image_inference(image_tensor)
            else:
                return self._simple_inference(image_tensor)
    
    def _sliding_window_inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """滑动窗口推理"""
        window_size = self.inf_cfg.get('window_size', DEFAULT_WINDOW_SIZE)
        stride = self.inf_cfg.get('stride', DEFAULT_STRIDE)
        
        return sliding_window_inference(
            self.model,
            image_tensor,
            window_size=window_size,
            stride=stride,
            num_classes=2
        )
    
    def _multi_scale_inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """多尺度推理 (TTA)"""
        scales = self.inf_cfg.get('scales', [0.5, 1.0, 1.5])
        
        return multi_scale_inference(
            self.model,
            image_tensor,
            scales=scales,
            num_classes=2,
            flip=True
        )
    
    def _whole_image_inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """整图推理"""
        max_size = self.inf_cfg.get('max_size', DEFAULT_MAX_SIZE)
        
        return whole_image_inference(
            self.model,
            image_tensor,
            num_classes=2,
            max_size=max_size
        )
    
    def _simple_inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """简单直接推理"""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        output = self.model(image_tensor)
        
        if isinstance(output, dict):
            pred = output['logits'][0]
        else:
            pred = output[0] if output.dim() == 4 else output
        
        return pred


# =============================================================================
# 结果保存
# =============================================================================

class ResultSaver:
    """结果保存器"""
    
    @staticmethod
    def save_prediction(
        pred: torch.Tensor,
        save_path: str,
        original_size: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        保存分割预测结果为图像
        
        Args:
            pred: 预测结果 [num_classes, H, W] 或 [H, W]
            save_path: 保存路径
            original_size: 原始尺寸 (W, H)
        """
        # 获取类别预测
        if pred.dim() == 3:
            pred = torch.argmax(pred, dim=0)
        
        # 转换为numpy并二值化
        pred_np = pred.cpu().numpy().astype(np.uint8) * 255
        
        # 调整到原始尺寸
        pred_img = Image.fromarray(pred_np)
        if original_size is not None:
            pred_img = pred_img.resize(original_size, Image.NEAREST)
        
        pred_img.save(save_path)
    
    @staticmethod
    def save_probability_map(
        pred: torch.Tensor,
        save_path: str,
        original_size: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        保存概率图
        
        Args:
            pred: 预测结果 [num_classes, H, W]
            save_path: 保存路径
            original_size: 原始尺寸 (W, H)
        """
        prob = torch.softmax(pred, dim=0)
        cloud_prob = prob[1].cpu().numpy()  # 云类别概率
        
        # 转换为16位PNG
        prob_img = (cloud_prob * 65535).astype(np.uint16)
        prob_pil = Image.fromarray(prob_img)
        
        if original_size is not None:
            prob_pil = prob_pil.resize(original_size, Image.BILINEAR)
        
        prob_pil.save(save_path)


# =============================================================================
# 批处理管理
# =============================================================================

class BatchInferenceManager:
    """批处理推理管理器"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: dict,
        output_dir: str,
        mode: str = 'sliding_window',
        save_prob: bool = False,
        normalization: Optional[str] = None,
        auto_norm: bool = False,
        nir_method: str = 'ensemble'
    ):
        self.model = model
        self.device = device
        self.config = config
        self.output_dir = output_dir
        self.mode = mode
        self.save_prob = save_prob
        self.normalization = normalization
        self.auto_norm = auto_norm
        self.nir_method = nir_method
        
        # 初始化处理器
        self.image_processor = ImageProcessor(device, config)
        self.inference_executor = InferenceExecutor(model, device, config)
        
        # 获取配置
        self.data_cfg = config.get('data', {})
        self.use_nir = self.data_cfg.get('use_nir', False)
        self.nir_method = nir_method or self.data_cfg.get('nir_method', 'ensemble')
        self.normalization = normalization or self.data_cfg.get('inference_normalization')
    
    def process_single_image(self, image_path: str) -> bool:
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 加载图像
            image, original_size = self.image_processor.load_image(image_path)
            
            # 准备输入张量
            tensor = self.image_processor.prepare_tensor(
                image,
                use_nir=self.use_nir,
                nir_method=self.nir_method
            )
            
            # 应用归一化
            tensor, used_norm = self.image_processor.apply_normalization(
                tensor,
                method=self.normalization,
                auto_recommend=self.auto_norm
            )
            
            # 执行推理
            pred = self.inference_executor.run(tensor, mode=self.mode)
            
            # 保存结果
            self._save_result(pred, image_path, original_size)
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error processing {image_path}: {e}")
            return False
    
    def _save_result(
        self,
        pred: torch.Tensor,
        image_path: str,
        original_size: Tuple[int, int]
    ) -> None:
        """保存推理结果"""
        filename = os.path.basename(image_path)
        name_wo_ext = os.path.splitext(filename)[0]
        
        if self.save_prob:
            save_path = os.path.join(self.output_dir, f"{name_wo_ext}_prob.png")
            ResultSaver.save_probability_map(pred, save_path, original_size)
        else:
            save_path = os.path.join(self.output_dir, f"{name_wo_ext}_mask.png")
            ResultSaver.save_prediction(pred, save_path, original_size)
    
    def process_batch(self, input_paths: List[str]) -> dict:
        """
        批量处理图像
        
        Args:
            input_paths: 图像路径列表
            
        Returns:
            dict: 处理统计
        """
        stats = {'total': len(input_paths), 'success': 0, 'failed': 0}
        
        for img_path in tqdm(input_paths, desc="Inference"):
            if self.process_single_image(img_path):
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        return stats


# =============================================================================
# 输入处理
# =============================================================================

def collect_input_files(input_path: str) -> List[str]:
    """
    收集输入文件
    
    Args:
        input_path: 输入路径（文件或目录）
        
    Returns:
        List[str]: 图像文件路径列表
    """
    if os.path.isfile(input_path):
        return [input_path]
    
    # 目录模式
    input_files = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        input_files.extend(Path(input_path).glob(f"*{ext}"))
        # 也检查大写扩展名
        input_files.extend(Path(input_path).glob(f"*{ext.upper()}"))
    
    return [str(f) for f in input_files]


def validate_args(args: argparse.Namespace) -> None:
    """验证命令行参数"""
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")


# =============================================================================
# 配置打印
# =============================================================================

def print_inference_config(
    config: dict,
    nir_method: str,
    normalization: Optional[str],
    auto_norm: bool,
    mode: str
) -> None:
    """打印推理配置"""
    data_cfg = config.get('data', {})
    use_nir = data_cfg.get('use_nir', False)
    
    print("\n" + "=" * 60)
    print("Inference Configuration")
    print("=" * 60)
    
    # 模型配置
    print(f"Mode:             {mode}")
    print(f"Channels:         {'4 (RGB+NIR)' if use_nir else '3 (RGB)'}")
    
    if use_nir:
        print(f"NIR Method:       {nir_method}")
    
    # 归一化配置
    if auto_norm:
        print(f"Normalization:    auto-recommend")
    elif normalization:
        print(f"Normalization:    {normalization}")
    else:
        print(f"Normalization:    none (using training normalization)")
    
    print("=" * 60 + "\n")


# =============================================================================
# 主程序
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='CloudSense-Net Inference - Cloud Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础推理
  python inference.py --checkpoint model.pth --input image.jpg --output results/
  
  # 自动归一化推荐
  python inference.py --checkpoint model.pth --input image.jpg --output results/ --auto_norm
  
  # 指定归一化方法
  python inference.py --checkpoint model.pth --input image.jpg --output results/ --normalization robust
  
  # 多尺度推理
  python inference.py --checkpoint model.pth --input image.jpg --output results/ --mode multi_scale
  
  # 批量处理目录
  python inference.py --checkpoint model.pth --input images_dir/ --output results/ --auto_norm

归一化方法说明:
  - adaptive:   自适应归一化（自动选择最佳方法）★推荐
  - robust:     稳健归一化（抗云层/阴影异常值）
  - standard:   Z-score标准化
  - windowed:   窗口化局部归一化（处理光照不均）
  - log_scaling: 对数缩放（高动态范围图像）
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='模型检查点路径 (e.g., experiments/best_model.pth)'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入图像路径或目录'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录'
    )
    
    # 推理参数
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='sliding_window',
        choices=['sliding_window', 'multi_scale', 'whole', 'simple'],
        help='推理模式 (默认: sliding_window)'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'mps', 'cuda', 'cpu'],
        help='计算设备 (默认: auto，优先MPS)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='CUDA GPU ID (默认: 0)'
    )
    
    # 图像处理参数
    parser.add_argument(
        '--nir_method',
        type=str,
        default=None,
        choices=['physical', 'vegetation', 'guided', 'context', 'ensemble'],
        help='伪NIR生成方法（覆盖配置文件）'
    )
    parser.add_argument(
        '--normalization', '-n',
        type=str,
        default=None,
        choices=['adaptive', 'robust', 'standard', 'windowed', 'log_scaling', 'minmax'],
        help='推理归一化方法（覆盖配置文件）'
    )
    parser.add_argument(
        '--auto_norm', '-a',
        action='store_true',
        help='自动分析图像并推荐最佳归一化方法'
    )
    
    # 输出参数
    parser.add_argument(
        '--save_prob', '-p',
        action='store_true',
        help='保存概率图（16位PNG）而非二值掩码'
    )
    
    return parser


def main():
    """主推理流程"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 验证参数
    validate_args(args)
    
    # 创建设备
    device = DeviceManager.get_device(args.device, args.gpu)
    
    # 加载模型
    model, config = ModelLoader.load_checkpoint(args.checkpoint, device)
    
    # 确定配置
    data_cfg = config.get('data', {})
    use_nir = data_cfg.get('use_nir', False)
    nir_method = args.nir_method or data_cfg.get('nir_method', 'ensemble')
    normalization = args.normalization or data_cfg.get('inference_normalization')
    
    # 打印配置
    print_inference_config(
        config, nir_method, normalization, args.auto_norm, args.mode
    )
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 收集输入文件
    input_files = collect_input_files(args.input)
    if not input_files:
        print(f"No supported images found in: {args.input}")
        return
    
    print(f"Found {len(input_files)} images to process\n")
    
    # 创建批处理管理器
    batch_manager = BatchInferenceManager(
        model=model,
        device=device,
        config=config,
        output_dir=args.output,
        mode=args.mode,
        save_prob=args.save_prob,
        normalization=normalization,
        auto_norm=args.auto_norm,
        nir_method=nir_method
    )
    
    # 执行批处理
    stats = batch_manager.process_batch(input_files)
    
    # 打印结果
    print(f"\n{'=' * 60}")
    print("Inference Completed!")
    print(f"  Total:   {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"  Results: {args.output}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
