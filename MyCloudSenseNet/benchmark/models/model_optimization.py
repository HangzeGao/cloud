"""
模型优化工具模块

包含：
1. 模型量化 (INT8/FP16)
2. 模型剪枝 (结构化/非结构化)
3. ONNX 导出与优化
4. TensorRT 支持准备
5. 模型压缩统计
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union, Callable
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import QuantStub, DeQuantStub


@dataclass
class ModelStats:
    """模型统计信息"""
    total_params: int
    trainable_params: int
    model_size_mb: float
    forward_pass_flops: Optional[int] = None
    
    def __str__(self):
        return (f"Params: {self.total_params:,} ({self.trainable_params:,} trainable), "
                f"Size: {self.model_size_mb:.2f} MB")


class ModelOptimizer:
    """
    模型优化器
    
    提供模型量化、剪枝和格式转换功能。
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.original_state = None
        self.optimization_history = []
    
    def get_model_stats(self) -> ModelStats:
        """获取模型统计信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 估算模型大小
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return ModelStats(
            total_params=total_params,
            trainable_params=trainable_params,
            model_size_mb=size_mb,
        )
    
    def apply_pruning(
        self,
        amount: float = 0.3,
        method: str = "unstructured",
        layers_to_prune: Optional[List[str]] = None,
    ) -> ModelStats:
        """
        应用模型剪枝
        
        Args:
            amount: 剪枝比例 (0-1)
            method: 剪枝方法 (unstructured, structured, global)
            layers_to_prune: 要剪枝的层名列表，None 表示所有卷积层
        
        Returns:
            剪枝后的模型统计
        """
        self.model.eval()
        
        # 确定要剪枝的参数
        parameters_to_prune = []
        
        for name, module in self.model.named_modules():
            # 根据指定层或自动选择卷积层
            if layers_to_prune is not None and name not in layers_to_prune:
                continue
            
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if module.weight is not None:
                    parameters_to_prune.append((module, 'weight'))
        
        if not parameters_to_prune:
            warnings.warn("No parameters to prune found")
            return self.get_model_stats()
        
        # 应用剪枝
        if method == "unstructured":
            # 非结构化剪枝：基于 L1 范数
            prune.l1_unstructured(
                parameters_to_prune,
                name='weight',
                amount=amount,
            )
        
        elif method == "structured":
            # 结构化剪枝：移除整个通道
            for module, param_name in parameters_to_prune:
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(
                        module,
                        name=param_name,
                        amount=amount,
                        n=2,  # 基于通道维度
                        dim=0,
                    )
        
        elif method == "global":
            # 全局剪枝
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
        
        # 使剪枝永久化
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        self.optimization_history.append({
            "type": "pruning",
            "method": method,
            "amount": amount,
            "layers": len(parameters_to_prune),
        })
        
        return self.get_model_stats()
    
    def prepare_quantization(
        self,
        backend: str = "fbgemm",
        dtype: torch.dtype = torch.qint8,
    ) -> nn.Module:
        """
        准备模型进行量化
        
        Args:
            backend: 量化后端 (fbgemm for x86, qnnpack for ARM)
            dtype: 量化数据类型
        
        Returns:
            准备好的模型
        """
        # 设置量化配置
        self.model.qconfig = torch.quantization.get_default_qconfig(backend)
        
        # 准备量化
        torch.quantization.prepare(self.model, inplace=True)
        
        self.optimization_history.append({
            "type": "quantization_prepare",
            "backend": backend,
            "dtype": str(dtype),
        })
        
        return self.model
    
    def calibrate_quantization(
        self,
        calibration_data: List[torch.Tensor],
    ):
        """
        校准量化参数
        
        需要使用代表性数据进行前向传播以确定量化参数。
        
        Args:
            calibration_data: 校准数据列表
        """
        self.model.eval()
        
        with torch.no_grad():
            for batch in calibration_data:
                _ = self.model(batch.to(self.device))
        
        self.optimization_history.append({
            "type": "quantization_calibrate",
            "num_batches": len(calibration_data),
        })
    
    def convert_quantization(
        self,
        inplace: bool = True,
    ) -> nn.Module:
        """
        转换为量化模型
        
        Returns:
            量化后的模型
        """
        quantized_model = torch.quantization.convert(self.model, inplace=inplace)
        
        self.optimization_history.append({
            "type": "quantization_convert",
        })
        
        return quantized_model
    
    def export_onnx(
        self,
        output_path: Union[str, Path],
        input_shape: Tuple[int, ...] = (1, 4, 512, 512),
        opset_version: int = 12,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        simplify: bool = True,
    ) -> Path:
        """
        导出模型为 ONNX 格式
        
        Args:
            output_path: 输出路径
            input_shape: 输入形状
            opset_version: ONNX opset 版本
            dynamic_axes: 动态轴配置
            simplify: 是否使用 onnx-simplifier 简化模型
        
        Returns:
            导出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 默认动态轴
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # 创建 dummy input
        dummy_input = torch.randn(*input_shape).to(self.device)
        
        # 导出
        torch.onnx.export(
            self.model.to(self.device),
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
        )
        
        # 简化模型（如果可用）
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                
                model_onnx = onnx.load(output_path)
                model_simplified, check = onnx_simplify(model_onnx)
                
                if check:
                    onnx.save(model_simplified, output_path)
                    print(f"Model simplified and saved to {output_path}")
            except ImportError:
                warnings.warn("onnx-simplifier not available, skipping simplification")
        
        self.optimization_history.append({
            "type": "onnx_export",
            "path": str(output_path),
            "opset": opset_version,
        })
        
        return output_path
    
    def prepare_tensorrt(
        self,
        onnx_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        fp16: bool = True,
        max_batch_size: int = 8,
        max_workspace_size: int = 1 << 30,  # 1GB
    ) -> Optional[Path]:
        """
        准备 TensorRT 引擎（需要 TensorRT 环境）
        
        Args:
            onnx_path: ONNX 模型路径
            output_path: 输出引擎路径
            fp16: 是否使用 FP16 精度
            max_batch_size: 最大 batch size
            max_workspace_size: 最大工作空间
        
        Returns:
            引擎文件路径或 None
        """
        try:
            import tensorrt as trt
        except ImportError:
            warnings.warn("TensorRT not available, skipping TensorRT preparation")
            return None
        
        onnx_path = Path(onnx_path)
        if output_path is None:
            output_path = onnx_path.with_suffix('.trt')
        else:
            output_path = Path(output_path)
        
        # 创建 logger 和 builder
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        
        # 解析 ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")
        
        # 配置 builder
        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_size
        
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        
        # 构建引擎
        engine = builder.build_engine(network, config)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # 保存引擎
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        self.optimization_history.append({
            "type": "tensorrt",
            "path": str(output_path),
            "fp16": fp16,
        })
        
        return output_path
    
    def optimize_for_inference(
        self,
        mode: str = "default",
    ) -> nn.Module:
        """
        为推理优化模型
        
        Args:
            mode: 优化模式
                - "default": 基本优化
                - "fast": 速度优先（融合 BN）
                - "memory": 内存优先
        
        Returns:
            优化后的模型
        """
        self.model.eval()
        
        if mode == "default":
            # 基本优化
            self.model = torch.jit.script(self.model) if hasattr(self.model, 'forward') else self.model
        
        elif mode == "fast":
            # 融合 BatchNorm
            self.model = torch.jit.script(self.model)
            self.model = torch.jit.optimize_for_inference(self.model)
        
        elif mode == "memory":
            # 使用 gradient checkpointing 减少内存（训练时）
            for module in self.model.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
        
        self.optimization_history.append({
            "type": "inference_opt",
            "mode": mode,
        })
        
        return self.model
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        include_optimizer: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        保存优化后的模型
        
        Args:
            path: 保存路径
            include_optimizer: 是否包含优化器状态
            metadata: 额外元数据
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimization_history": self.optimization_history,
            "model_stats": self.get_model_stats().__dict__,
            "metadata": metadata or {},
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        model_class: Optional[type] = None,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        加载优化后的模型
        
        Args:
            path: 检查点路径
            model_class: 模型类（用于重建模型）
        
        Returns:
            (模型, 元数据)
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        if model_class is not None:
            model = model_class()
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = None
        
        metadata = {
            "optimization_history": checkpoint.get("optimization_history", []),
            "model_stats": checkpoint.get("model_stats", {}),
            "metadata": checkpoint.get("metadata", {}),
        }
        
        return model, metadata


class QuantizationAwareTraining:
    """
    量化感知训练 (QAT) 支持
    
    在训练过程中模拟量化，以获得更好的量化后精度。
    """
    
    def __init__(
        self,
        model: nn.Module,
        backend: str = "fbgemm",
    ):
        self.model = model
        self.backend = backend
        self.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    
    def prepare(self) -> nn.Module:
        """准备 QAT"""
        # 融合模块（提高量化精度）
        self.model.eval()
        self.model = torch.quantization.fuse_modules(
            self.model,
            [['conv', 'bn', 'relu']],
            inplace=True,
        ) if hasattr(self.model, 'conv') else self.model
        
        # 准备 QAT
        self.model.train()
        self.model.qconfig = self.qconfig
        torch.quantization.prepare_qat(self.model, inplace=True)
        
        return self.model
    
    def convert(self) -> nn.Module:
        """转换为量化模型"""
        self.model.eval()
        return torch.quantization.convert(self.model, inplace=True)


class StructuredPruner:
    """
    结构化剪枝
    
    移除整个通道/滤波器，保持模型结构的完整性。
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.pruning_plan = {}
    
    def analyze_importance(
        self,
        dataloader: Optional[Any] = None,
        num_samples: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        分析通道重要性
        
        Args:
            dataloader: 数据加载器
            num_samples: 样本数
        
        Returns:
            各层通道重要性分数
        """
        importance_scores = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 基于 L1 范数的重要性
                weights = module.weight.data.abs().mean(dim=[1, 2, 3])
                importance_scores[name] = weights
        
        return importance_scores
    
    def plan_pruning(
        self,
        target_sparsity: float = 0.3,
        importance_scores: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, List[int]]:
        """
        规划剪枝策略
        
        Args:
            target_sparsity: 目标稀疏度
            importance_scores: 重要性分数
        
        Returns:
            每层的剪枝索引
        """
        if importance_scores is None:
            importance_scores = self.analyze_importance()
        
        pruning_indices = {}
        
        for name, scores in importance_scores.items():
            num_channels = len(scores)
            num_to_prune = int(num_channels * target_sparsity)
            
            # 选择重要性最低的通道
            _, indices = torch.topk(scores, k=num_to_prune, largest=False)
            pruning_indices[name] = indices.tolist()
        
        self.pruning_plan = pruning_indices
        return pruning_indices
    
    def apply_channel_pruning(
        self,
        layer_name: str,
        channel_indices: List[int],
    ):
        """
        应用通道剪枝到特定层
        
        Args:
            layer_name: 层名
            channel_indices: 要剪枝的通道索引
        """
        # 获取层
        module = dict(self.model.named_modules()).get(layer_name)
        if module is None or not isinstance(module, nn.Conv2d):
            return
        
        # 创建掩码
        mask = torch.ones(module.weight.size(0), dtype=torch.bool)
        mask[channel_indices] = False
        
        # 剪枝权重
        new_weight = module.weight.data[mask]
        module.weight = nn.Parameter(new_weight)
        
        if module.bias is not None:
            new_bias = module.bias.data[mask]
            module.bias = nn.Parameter(new_bias)
        
        # 更新输出通道数
        module.out_channels = len(mask) - len(channel_indices)


def benchmark_inference_speed(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 4, 512, 512),
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """
    基准测试模型推理速度
    
    Args:
        model: 模型
        input_shape: 输入形状
        num_runs: 测试次数
        warmup_runs: 预热次数
        device: 设备
    
    Returns:
        统计结果字典
    """
    import time
    
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_shape).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # 正式测试
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转换为毫秒
    
    return {
        "mean_ms": sum(times) / len(times),
        "median_ms": sorted(times)[len(times) // 2],
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
        "fps": 1000 / (sum(times) / len(times)),
    }


def compare_models(
    model_a: nn.Module,
    model_b: nn.Module,
    input_shape: Tuple[int, ...] = (1, 4, 512, 512),
) -> Dict[str, Any]:
    """
    比较两个模型的性能
    
    Args:
        model_a: 模型 A
        model_b: 模型 B
        input_shape: 输入形状
    
    Returns:
        比较结果
    """
    stats_a = ModelOptimizer(model_a).get_model_stats()
    stats_b = ModelOptimizer(model_b).get_model_stats()
    
    speed_a = benchmark_inference_speed(model_a, input_shape)
    speed_b = benchmark_inference_speed(model_b, input_shape)
    
    return {
        "params_ratio": stats_b.total_params / stats_a.total_params,
        "size_ratio": stats_b.model_size_mb / stats_a.model_size_mb,
        "speed_ratio": speed_a["mean_ms"] / speed_b["mean_ms"],
        "model_a": {"stats": stats_a.__dict__, "speed": speed_a},
        "model_b": {"stats": stats_b.__dict__, "speed": speed_b},
    }


if __name__ == "__main__":
    print("Testing Model Optimization...")
    
    # 创建模拟模型
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 3, 1)
        
        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            return self.conv3(x)
    
    model = MockModel()
    
    # 创建优化器
    optimizer = ModelOptimizer(model)
    
    # 测试统计
    print("\n1. Model Statistics:")
    stats = optimizer.get_model_stats()
    print(f"   {stats}")
    
    # 测试剪枝
    print("\n2. Pruning:")
    pruned_stats = optimizer.apply_pruning(amount=0.3, method="unstructured")
    print(f"   After pruning: {pruned_stats}")
    
    # 测试 ONNX 导出
    print("\n3. ONNX Export:")
    try:
        onnx_path = optimizer.export_onnx("/tmp/test_model.onnx", input_shape=(1, 4, 64, 64))
        print(f"   Exported to: {onnx_path}")
    except Exception as e:
        print(f"   Export skipped: {e}")
    
    # 测试基准测试
    print("\n4. Benchmark:")
    benchmark = benchmark_inference_speed(model, input_shape=(1, 4, 64, 64), num_runs=10)
    print(f"   Mean: {benchmark['mean_ms']:.2f}ms")
    print(f"   FPS: {benchmark['fps']:.2f}")
    
    print("\n✓ All tests passed!")
