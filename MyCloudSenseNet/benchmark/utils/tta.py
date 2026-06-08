"""
Test-Time Augmentation (TTA) module - v2.0

包含:
1. 基础 TTA (D4群变换)
2. 自适应 TTA (根据置信度动态选择策略)
3. 不确定性引导 TTA
4. 集成 TTA
"""

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F


class TTAStrategy(Enum):
    """TTA 策略枚举"""
    NONE = "none"
    LIGHT = "light"
    STANDARD = "standard"
    HEAVY = "heavy"
    ADAPTIVE = "adaptive"


@dataclass
class TTAResult:
    """TTA 结果"""
    probabilities: torch.Tensor
    predictions: torch.Tensor
    confidence: torch.Tensor
    uncertainty: torch.Tensor
    num_augmentations: int
    inference_time_ms: float
    strategy_used: str


class TTAugmenter:
    """TTA 变换器 - D4群变换"""
    
    LIGHT = ["none", "rot180", "hflip"]
    STANDARD = ["none", "rot90", "rot180", "rot270", "hflip", "vflip", "diag", "anti_diag"]
    
    @staticmethod
    def apply(image: torch.Tensor, mode: str) -> torch.Tensor:
        """应用 TTA 变换"""
        if mode == "none":
            return image
        elif mode == "hflip":
            return torch.flip(image, dims=(-1,))
        elif mode == "vflip":
            return torch.flip(image, dims=(-2,))
        elif mode == "rot180":
            return torch.flip(image, dims=(-2, -1))
        elif mode == "rot90":
            return torch.rot90(image, k=1, dims=(-2, -1))
        elif mode == "rot270":
            return torch.rot90(image, k=3, dims=(-2, -1))
        elif mode == "diag":
            return image.transpose(-2, -1)
        elif mode == "anti_diag":
            return torch.flip(image.transpose(-2, -1), dims=(-2, -1))
        else:
            raise ValueError(f"Unknown TTA mode: {mode}")
    
    @staticmethod
    def undo(prediction: torch.Tensor, mode: str) -> torch.Tensor:
        """撤销 TTA 变换"""
        if mode in ("none", "hflip", "vflip", "rot180"):
            return TTAugmenter.apply(prediction, mode)
        elif mode == "rot90":
            return torch.rot90(prediction, k=3, dims=(-2, -1))
        elif mode == "rot270":
            return torch.rot90(prediction, k=1, dims=(-2, -1))
        elif mode == "diag":
            return prediction.transpose(-2, -1)
        elif mode == "anti_diag":
            return torch.flip(prediction, dims=(-2, -1)).transpose(-2, -1)
        else:
            raise ValueError(f"Unknown TTA mode: {mode}")


class AdaptiveTTA:
    """
    自适应 TTA
    
    根据预测置信度动态选择变换强度:
    - 高置信度 (> threshold): 轻量 TTA (3 transforms)
    - 中等置信度: 标准 TTA (8 transforms)
    - 低置信度 (< threshold / 2): 重度 TTA (16 transforms)
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.9,
        low_confidence_threshold: float = 0.7,
        light_transforms: Optional[List[str]] = None,
        standard_transforms: Optional[List[str]] = None,
        heavy_transforms: Optional[List[str]] = None,
        enable_uncertainty: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.enable_uncertainty = enable_uncertainty
        
        self.light_transforms = light_transforms or TTAugmenter.LIGHT
        self.standard_transforms = standard_transforms or TTAugmenter.STANDARD
        self.heavy_transforms = heavy_transforms or (TTAugmenter.STANDARD * 2)
    
    def _compute_confidence(self, logits: torch.Tensor) -> float:
        """计算预测置信度"""
        probs = F.softmax(logits, dim=1)
        return probs.max(dim=1)[0].mean().item()
    
    def _compute_uncertainty(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """计算预测不确定性"""
        if len(predictions) < 2:
            return torch.zeros_like(predictions[0][:, 0, :, :])
        
        stacked = torch.stack(predictions, dim=0)
        mean_pred = stacked.mean(dim=0)
        entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(dim=1)
        variance = stacked.var(dim=0).mean(dim=1)
        return (entropy + variance) / 2.0
    
    def predict(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
    ) -> TTAResult:
        """自适应 TTA 预测"""
        start_time = time.time()
        
        # 快速预测评估置信度
        with torch.no_grad():
            quick_logits = model(x)
            confidence = self._compute_confidence(quick_logits)
        
        # 选择策略
        if confidence >= self.confidence_threshold:
            transforms = self.light_transforms
            strategy = "light"
        elif confidence >= self.low_confidence_threshold:
            transforms = self.standard_transforms
            strategy = "standard"
        else:
            transforms = self.heavy_transforms
            strategy = "heavy"
        
        # 执行 TTA
        prob_sum = None
        predictions = []
        
        for mode in transforms:
            with torch.no_grad():
                x_aug = TTAugmenter.apply(x, mode)
                logits = model(x_aug)
                logits = TTAugmenter.undo(logits, mode)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
                
                if prob_sum is None:
                    prob_sum = probs
                else:
                    prob_sum = prob_sum + probs
        
        avg_probs = prob_sum / len(transforms)
        final_pred = torch.argmax(avg_probs, dim=1)
        final_confidence = avg_probs.max(dim=1)[0]
        
        uncertainty = (self._compute_uncertainty(predictions) 
                        if self.enable_uncertainty 
                        else torch.zeros_like(final_confidence))
        
        return TTAResult(
            probabilities=avg_probs,
            predictions=final_pred,
            confidence=final_confidence,
            uncertainty=uncertainty,
            num_augmentations=len(transforms),
            inference_time_ms=(time.time() - start_time) * 1000,
            strategy_used=strategy,
        )


def predict_with_tta(
    model: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    tta_strategy: str = "adaptive",
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    使用 TTA 进行预测
    
    Args:
        model: 模型前向函数
        x: 输入图像 [B, C, H, W]
        tta_strategy: TTA 策略 (none, adaptive, light, standard)
        **kwargs: 额外参数
    
    Returns:
        (概率, 元信息)
    """
    if tta_strategy == "none":
        with torch.no_grad():
            logits = model(x)
        probs = F.softmax(logits, dim=1)
        return probs, {"strategy": "none", "num_aug": 1}
    
    elif tta_strategy == "adaptive":
        tta = AdaptiveTTA(**kwargs)
        result = tta.predict(model, x)
        return result.probabilities, {
            "strategy": result.strategy_used,
            "num_aug": result.num_augmentations,
            "confidence": result.confidence.mean().item(),
            "uncertainty": result.uncertainty.mean().item(),
        }
    
    elif tta_strategy == "light":
        tta = AdaptiveTTA(
            confidence_threshold=1.0,
            light_transforms=TTAugmenter.LIGHT,
        )
        result = tta.predict(model, x)
        return result.probabilities, {"strategy": "light", "num_aug": 3}
    
    elif tta_strategy == "standard":
        tta = AdaptiveTTA(
            confidence_threshold=0.0,
            standard_transforms=TTAugmenter.STANDARD,
        )
        result = tta.predict(model, x)
        return result.probabilities, {"strategy": "standard", "num_aug": 8}
    
    else:
        raise ValueError(f"Unknown TTA strategy: {tta_strategy}")
