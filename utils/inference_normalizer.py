"""
推理阶段归一化工具 - 规范化图像归一化方法

提供多种针对遥感图像推理的标准化归一化方法，
确保输入数据的一致性和模型性能的稳定性。
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, Dict, List, Tuple
from enum import Enum


class NormalizationMethod(Enum):
    """支持的归一化方法枚举"""
    MINMAX = "minmax"                           # 简单Min-Max归一化
    STANDARD = "standard"                       # Z-score标准化
    ROBUST = "robust"                           # 稳健归一化（基于分位数）
    HISTOGRAM_MATCH = "histogram_match"         # 直方图匹配
    ADAPTIVE = "adaptive"                       # 自适应归一化
    REFERENCE_BASED = "reference_based"         # 参考图像归一化
    WINDOWED = "windowed"                     # 窗口化局部归一化
    LOG_SCALING = "log_scaling"                 # 对数缩放归一化


class BaseNormalizer:
    """归一化基类"""
    
    def __init__(self, method: NormalizationMethod):
        self.method = method
        self.is_fitted = False
        self.stats = {}
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'BaseNormalizer':
        """
        从数据中学习归一化参数
        
        Args:
            data: 输入数据，numpy [H, W, C] 或 torch [C, H, W] 或 [B, C, H, W]
        
        Returns:
            self
        """
        raise NotImplementedError
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        应用归一化
        
        Args:
            data: 输入数据
        
        Returns:
            归一化后的数据（保持原始类型）
        """
        raise NotImplementedError
    
    def fit_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """学习参数并应用归一化"""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """反归一化"""
        raise NotImplementedError
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """统一转换为numpy数组，保持通道在最后"""
        if isinstance(data, torch.Tensor):
            # 处理不同维度 [C, H, W] -> [H, W, C]
            if data.dim() == 3:
                return data.permute(1, 2, 0).cpu().numpy()
            elif data.dim() == 4:
                # [B, C, H, W] -> [B, H, W, C]
                return data.permute(0, 2, 3, 1).cpu().numpy()
            elif data.dim() == 2:
                return data.cpu().numpy()
        return np.array(data)
    
    def _to_tensor(self, data: np.ndarray, original_tensor: torch.Tensor) -> torch.Tensor:
        """转换回tensor，恢复原始维度"""
        tensor = torch.from_numpy(data)
        
        if original_tensor.dim() == 3:  # [C, H, W]
            # data is [H, W, C]
            tensor = tensor.permute(2, 0, 1)
        elif original_tensor.dim() == 4:  # [B, C, H, W]
            # data is [B, H, W, C]
            tensor = tensor.permute(0, 3, 1, 2)
        
        return tensor.to(original_tensor.device, dtype=original_tensor.dtype)


class RobustScaler(BaseNormalizer):
    """
    稳健归一化器 - 基于分位数的归一化
    
    使用四分位数而非最小/最大值，对异常值更加鲁棒。
    特别适用于遥感图像中可能存在的云层、阴影等异常高/低值。
    
    公式: x' = (x - median) / (q75 - q25 + epsilon)
    """
    
    def __init__(self, 
                 lower_percentile: float = 2.0,
                 upper_percentile: float = 98.0,
                 clip: bool = True,
                 per_channel: bool = True):
        super().__init__(NormalizationMethod.ROBUST)
        self.lower_p = lower_percentile
        self.upper_p = upper_percentile
        self.clip = clip
        self.per_channel = per_channel
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'RobustScaler':
        arr = self._to_numpy(data)
        
        if self.per_channel and arr.ndim >= 3:
            num_channels = arr.shape[-1]
            self.stats = {
                'medians': [],
                'iqr_scales': [],
                'p_lows': [],
                'p_highs': []
            }
            for c in range(num_channels):
                channel_data = arr[..., c].flatten()
                median = np.median(channel_data)
                p_low = np.percentile(channel_data, self.lower_p)
                p_high = np.percentile(channel_data, self.upper_p)
                iqr = p_high - p_low
                
                self.stats['medians'].append(median)
                self.stats['iqr_scales'].append(iqr if iqr > 1e-8 else 1.0)
                self.stats['p_lows'].append(p_low)
                self.stats['p_highs'].append(p_high)
        else:
            flat_data = arr.flatten()
            self.stats = {
                'median': np.median(flat_data),
                'iqr_scale': np.percentile(flat_data, self.upper_p) - np.percentile(flat_data, self.lower_p),
                'p_low': np.percentile(flat_data, self.lower_p),
                'p_high': np.percentile(flat_data, self.upper_p)
            }
            if self.stats['iqr_scale'] < 1e-8:
                self.stats['iqr_scale'] = 1.0
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if not self.is_fitted:
            return self.fit_transform(data)
        
        is_tensor = isinstance(data, torch.Tensor)
        arr = self._to_numpy(data).astype(np.float32)
        
        if self.per_channel and arr.ndim >= 3:
            for c in range(arr.shape[-1]):
                arr[..., c] = (arr[..., c] - self.stats['medians'][c]) / self.stats['iqr_scales'][c]
        else:
            arr = (arr - self.stats['median']) / self.stats['iqr_scale']
        
        if self.clip:
            # 裁剪到合理范围（通常是[-5, 5]）
            arr = np.clip(arr, -5, 5)
        
        if is_tensor:
            return self._to_tensor(arr, data)
        return arr
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(data, torch.Tensor)
        arr = self._to_numpy(data)
        
        if self.per_channel and arr.ndim >= 3:
            for c in range(arr.shape[-1]):
                arr[..., c] = arr[..., c] * self.stats['iqr_scales'][c] + self.stats['medians'][c]
        else:
            arr = arr * self.stats['iqr_scale'] + self.stats['median']
        
        if is_tensor:
            return self._to_tensor(arr, data)
        return arr


class HistogramNormalizer(BaseNormalizer):
    """
    直方图匹配归一化器
    
    将输入图像的直方图匹配到参考直方图（通常是训练集的统计分布）。
    这种方法可以消除不同传感器、不同时间获取图像的辐射差异。
    
    参考：
    - Nutan Chen et al., "Remote Sensing Image Classification with Deep Learning"
    """
    
    def __init__(self, 
                 num_bins: int = 256,
                 reference_histograms: Optional[Dict[int, np.ndarray]] = None):
        super().__init__(NormalizationMethod.HISTOGRAM_MATCH)
        self.num_bins = num_bins
        self.reference_histograms = reference_histograms or {}
    
    def set_reference(self, histograms: Dict[int, np.ndarray]):
        """设置参考直方图，key为通道索引"""
        self.reference_histograms = histograms
        self.is_fitted = True
    
    def compute_reference_from_batch(self, 
                                     images: List[Union[np.ndarray, torch.Tensor]],
                                     max_samples: int = 100) -> 'HistogramNormalizer':
        """
        从一批图像计算参考直方图
        
        Args:
            images: 图像列表
            max_samples: 最大采样数量
        """
        if len(images) > max_samples:
            indices = np.random.choice(len(images), max_samples, replace=False)
            images = [images[i] for i in indices]
        
        # 收集所有像素值
        all_values = []
        for img in images:
            arr = self._to_numpy(img)
            all_values.append(arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 3 else arr.flatten())
        
        combined = np.concatenate(all_values, axis=0)
        
        # 计算每个通道的直方图和CDF
        num_channels = combined.shape[-1] if combined.ndim > 1 else 1
        self.reference_histograms = {}
        
        for c in range(num_channels):
            channel_data = combined[:, c] if combined.ndim > 1 else combined
            hist, bin_edges = np.histogram(channel_data, bins=self.num_bins, range=(0, 1))
            cdf = np.cumsum(hist) / np.sum(hist)
            self.reference_histograms[c] = {
                'cdf': cdf,
                'bin_edges': bin_edges,
                'hist': hist
            }
        
        self.is_fitted = True
        return self
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'HistogramNormalizer':
        """如果未设置参考，使用数据本身作为参考（恒等变换）"""
        if not self.reference_histograms:
            return self.compute_reference_from_batch([data])
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if not self.is_fitted:
            raise ValueError("必须先设置参考直方图或调用fit()")
        
        is_tensor = isinstance(data, torch.Tensor)
        arr = self._to_numpy(data).astype(np.float32)
        
        # 确保数值在[0, 1]范围内
        arr = np.clip(arr, 0, 1)
        
        num_channels = arr.shape[-1] if arr.ndim >= 3 else 1
        
        for c in range(num_channels):
            if c not in self.reference_histograms:
                continue
            
            ref = self.reference_histograms[c]
            channel_data = arr[..., c] if arr.ndim >= 3 else arr
            
            # 计算输入的CDF
            hist_input, _ = np.histogram(channel_data.flatten(), 
                                         bins=self.num_bins, 
                                         range=(0, 1))
            cdf_input = np.cumsum(hist_input) / np.sum(hist_input)
            
            # 直方图匹配
            # 1. 将输入值映射到其CDF
            bin_indices = (channel_data * (self.num_bins - 1)).astype(int)
            bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
            cdf_values = cdf_input[bin_indices]
            
            # 2. 在参考CDF中找到最接近的值
            matched = np.interp(cdf_values, ref['cdf'], ref['bin_edges'][:-1])
            
            if arr.ndim >= 3:
                arr[..., c] = matched
            else:
                arr = matched
        
        if is_tensor:
            return self._to_tensor(arr, data)
        return arr
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        # 直方图匹配通常不可逆，返回原数据
        return data


class AdaptiveNormalizer(BaseNormalizer):
    """
    自适应归一化器
    
    根据图像内容自动选择最佳归一化策略。
    分析图像的统计特征，动态调整归一化参数。
    
    策略选择逻辑：
    - 高动态范围图像 -> RobustScaler
    - 低对比度图像 -> Contrast Enhancement
    - 正常图像 -> Standard normalization
    """
    
    def __init__(self, 
                 auto_select: bool = True,
                 preferred_method: Optional[NormalizationMethod] = None):
        super().__init__(NormalizationMethod.ADAPTIVE)
        self.auto_select = auto_select
        self.preferred_method = preferred_method
        self.selected_method: Optional[NormalizationMethod] = None
        self.sub_normalizer: Optional[BaseNormalizer] = None
    
    def _analyze_image(self, arr: np.ndarray) -> Dict[str, float]:
        """分析图像特征"""
        if arr.ndim >= 3:
            # 计算亮度通道（如果是RGB）
            if arr.shape[-1] == 3:
                luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            else:
                luminance = arr.mean(axis=-1)
        else:
            luminance = arr
        
        stats = {
            'mean': np.mean(luminance),
            'std': np.std(luminance),
            'min': np.min(luminance),
            'max': np.max(luminance),
            'p1': np.percentile(luminance, 1),
            'p99': np.percentile(luminance, 99),
        }
        
        # 计算动态范围
        stats['dynamic_range'] = stats['p99'] - stats['p1']
        
        # 检测异常值比例
        outliers_low = np.sum(luminance < stats['p1']) / luminance.size
        outliers_high = np.sum(luminance > stats['p99']) / luminance.size
        stats['outlier_ratio'] = outliers_low + outliers_high
        
        return stats
    
    def _select_method(self, stats: Dict[str, float]) -> NormalizationMethod:
        """根据图像特征选择归一化方法"""
        if self.preferred_method:
            return self.preferred_method
        
        # 决策逻辑
        if stats['outlier_ratio'] > 0.05 or stats['dynamic_range'] > 0.8:
            # 存在明显异常值或高动态范围 -> 稳健归一化
            return NormalizationMethod.ROBUST
        elif stats['dynamic_range'] < 0.1:
            # 低对比度 -> 标准归一化增强对比度
            return NormalizationMethod.STANDARD
        else:
            # 正常情况 -> Min-Max
            return NormalizationMethod.MINMAX
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'AdaptiveNormalizer':
        arr = self._to_numpy(data)
        stats = self._analyze_image(arr)
        
        self.selected_method = self._select_method(stats)
        self.stats['image_stats'] = stats
        self.stats['selected_method'] = self.selected_method.value
        
        # 创建对应的子归一化器
        if self.selected_method == NormalizationMethod.ROBUST:
            self.sub_normalizer = RobustScaler()
        elif self.selected_method == NormalizationMethod.STANDARD:
            self.sub_normalizer = StandardNormalizer()
        else:
            self.sub_normalizer = MinMaxNormalizer()
        
        self.sub_normalizer.fit(data)
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if not self.is_fitted:
            return self.fit_transform(data)
        
        return self.sub_normalizer.transform(data)
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self.sub_normalizer.inverse_transform(data)


class MinMaxNormalizer(BaseNormalizer):
    """简单Min-Max归一化器"""
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1), per_channel: bool = True):
        super().__init__(NormalizationMethod.MINMAX)
        self.feature_range = feature_range
        self.per_channel = per_channel
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'MinMaxNormalizer':
        arr = self._to_numpy(data)
        
        if self.per_channel and arr.ndim >= 3:
            self.stats = {'mins': [], 'maxs': []}
            for c in range(arr.shape[-1]):
                self.stats['mins'].append(arr[..., c].min())
                self.stats['maxs'].append(arr[..., c].max())
        else:
            self.stats = {'min': arr.min(), 'max': arr.max()}
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if not self.is_fitted:
            return self.fit_transform(data)
        
        is_tensor = isinstance(data, torch.Tensor)
        arr = self._to_numpy(data).astype(np.float32)
        
        low, high = self.feature_range
        
        if self.per_channel and arr.ndim >= 3:
            for c in range(arr.shape[-1]):
                min_val = self.stats['mins'][c]
                max_val = self.stats['maxs'][c]
                range_val = max_val - min_val if max_val > min_val else 1.0
                arr[..., c] = (arr[..., c] - min_val) / range_val * (high - low) + low
        else:
            min_val = self.stats['min']
            max_val = self.stats['max']
            range_val = max_val - min_val if max_val > min_val else 1.0
            arr = (arr - min_val) / range_val * (high - low) + low
        
        if is_tensor:
            return self._to_tensor(arr, data)
        return arr
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(data, torch.Tensor)
        arr = self._to_numpy(data)
        
        low, high = self.feature_range
        
        if self.per_channel and arr.ndim >= 3:
            for c in range(arr.shape[-1]):
                min_val = self.stats['mins'][c]
                max_val = self.stats['maxs'][c]
                range_val = max_val - min_val if max_val > min_val else 1.0
                arr[..., c] = (arr[..., c] - low) / (high - low) * range_val + min_val
        else:
            min_val = self.stats['min']
            max_val = self.stats['max']
            range_val = max_val - min_val if max_val > min_val else 1.0
            arr = (arr - low) / (high - low) * range_val + min_val
        
        if is_tensor:
            return self._to_tensor(arr, data)
        return arr


class StandardNormalizer(BaseNormalizer):
    """Z-score标准化归一化器"""
    
    def __init__(self, per_channel: bool = True, epsilon: float = 1e-8):
        super().__init__(NormalizationMethod.STANDARD)
        self.per_channel = per_channel
        self.epsilon = epsilon
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'StandardNormalizer':
        arr = self._to_numpy(data)
        
        if self.per_channel and arr.ndim >= 3:
            self.stats = {'means': [], 'stds': []}
            for c in range(arr.shape[-1]):
                self.stats['means'].append(arr[..., c].mean())
                self.stats['stds'].append(arr[..., c].std() + self.epsilon)
        else:
            self.stats = {'mean': arr.mean(), 'std': arr.std() + self.epsilon}
        
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if not self.is_fitted:
            return self.fit_transform(data)
        
        is_tensor = isinstance(data, torch.Tensor)
        arr = self._to_numpy(data).astype(np.float32)
        
        if self.per_channel and arr.ndim >= 3:
            for c in range(arr.shape[-1]):
                arr[..., c] = (arr[..., c] - self.stats['means'][c]) / self.stats['stds'][c]
        else:
            arr = (arr - self.stats['mean']) / self.stats['std']
        
        if is_tensor:
            return self._to_tensor(arr, data)
        return arr
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(data, torch.Tensor)
        arr = self._to_numpy(data)
        
        if self.per_channel and arr.ndim >= 3:
            for c in range(arr.shape[-1]):
                arr[..., c] = arr[..., c] * self.stats['stds'][c] + self.stats['means'][c]
        else:
            arr = arr * self.stats['std'] + self.stats['mean']
        
        if is_tensor:
            return self._to_tensor(arr, data)
        return arr


class WindowedNormalizer(BaseNormalizer):
    """
    窗口化局部归一化器
    
    使用滑动窗口进行局部归一化，适用于光照不均匀的图像。
    参考SCLAHE（对比度受限自适应直方图均衡化）思想。
    """
    
    def __init__(self, 
                 window_size: int = 64,
                 clip_limit: float = 4.0,
                 method: str = 'mean_std'):
        super().__init__(NormalizationMethod.WINDOWED)
        self.window_size = window_size
        self.clip_limit = clip_limit
        self.method = method  # 'mean_std' 或 'min_max'
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'WindowedNormalizer':
        # 窗口归一化不需要全局统计
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(data, torch.Tensor)
        arr = self._to_numpy(data).astype(np.float32)
        
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        
        h, w, c = arr.shape
        result = np.zeros_like(arr)
        
        for ch in range(c):
            channel = arr[:, :, ch]
            
            # 计算局部均值和标准差
            from scipy.ndimage import uniform_filter
            
            local_mean = uniform_filter(channel, size=self.window_size, mode='reflect')
            
            if self.method == 'mean_std':
                local_sq_mean = uniform_filter(channel**2, size=self.window_size, mode='reflect')
                local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0) + 1e-8)
                
                # 对比度受限
                local_std = np.clip(local_std, self.clip_limit, None)
                
                # 局部标准化
                result[:, :, ch] = (channel - local_mean) / local_std
            else:
                local_max = uniform_filter(channel, size=self.window_size, mode='reflect')
                # 使用形态学操作近似局部最大值和最小值
                from scipy.ndimage import maximum_filter, minimum_filter
                local_max = maximum_filter(channel, size=self.window_size, mode='reflect')
                local_min = minimum_filter(channel, size=self.window_size, mode='reflect')
                
                range_val = local_max - local_min + 1e-8
                result[:, :, ch] = (channel - local_min) / range_val
        
        if is_tensor:
            return self._to_tensor(result, data)
        return result.squeeze() if result.shape[-1] == 1 else result
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        # 局部归一化难以精确逆变换
        raise NotImplementedError("Windowed normalization is not easily invertible")


class LogScalingNormalizer(BaseNormalizer):
    """
    对数缩放归一化器
    
    适用于具有大动态范围的遥感图像。
    公式: x' = log(1 + x) / log(1 + max)
    """
    
    def __init__(self, base: float = np.e, epsilon: float = 1e-8):
        super().__init__(NormalizationMethod.LOG_SCALING)
        self.base = base
        self.epsilon = epsilon
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'LogScalingNormalizer':
        arr = self._to_numpy(data)
        self.stats = {'max_val': arr.max()}
        self.is_fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if not self.is_fitted:
            return self.fit_transform(data)
        
        is_tensor = isinstance(data, torch.Tensor)
        arr = self._to_numpy(data).astype(np.float32)
        
        # 确保非负
        arr = np.maximum(arr, 0)
        
        # 对数变换
        log_max = np.log(1 + self.stats['max_val']) / np.log(self.base)
        result = np.log(1 + arr) / np.log(self.base) / log_max
        
        if is_tensor:
            return self._to_tensor(result, data)
        return result
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(data, torch.Tensor)
        arr = self._to_numpy(data)
        
        log_max = np.log(1 + self.stats['max_val']) / np.log(self.base)
        result = self.base ** (arr * log_max) - 1
        
        if is_tensor:
            return self._to_tensor(result, data)
        return result


# ==================== 便捷函数 ====================

def create_normalizer(method: Union[str, NormalizationMethod], **kwargs) -> BaseNormalizer:
    """
    工厂函数：创建指定的归一化器
    
    Args:
        method: 归一化方法名称或枚举
        **kwargs: 传递给归一化器的参数
    
    Returns:
        对应的归一化器实例
    """
    if isinstance(method, str):
        method = NormalizationMethod(method.lower())
    
    normalizers = {
        NormalizationMethod.MINMAX: MinMaxNormalizer,
        NormalizationMethod.STANDARD: StandardNormalizer,
        NormalizationMethod.ROBUST: RobustScaler,
        NormalizationMethod.HISTOGRAM_MATCH: HistogramNormalizer,
        NormalizationMethod.ADAPTIVE: AdaptiveNormalizer,
        NormalizationMethod.WINDOWED: WindowedNormalizer,
        NormalizationMethod.LOG_SCALING: LogScalingNormalizer,
    }
    
    if method not in normalizers:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalizers[method](**kwargs)


def normalize_for_inference(
    image: Union[np.ndarray, torch.Tensor],
    method: str = 'adaptive',
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    推理归一化的便捷函数
    
    Args:
        image: 输入图像
        method: 归一化方法
        **kwargs: 额外参数
    
    Returns:
        归一化后的图像
    """
    normalizer = create_normalizer(method, **kwargs)
    return normalizer.fit_transform(image)


def batch_normalize(
    images: List[Union[np.ndarray, torch.Tensor]],
    method: str = 'robust',
    shared_stats: bool = True,
    **kwargs
) -> List[Union[np.ndarray, torch.Tensor]]:
    """
    批量归一化
    
    Args:
        images: 图像列表
        method: 归一化方法
        shared_stats: 是否使用共享统计（第一张图像的统计）
        **kwargs: 额外参数
    
    Returns:
        归一化后的图像列表
    """
    if not images:
        return []
    
    normalizer = create_normalizer(method, **kwargs)
    
    if shared_stats:
        # 使用第一张图像学习统计参数
        normalizer.fit(images[0])
        return [normalizer.transform(img) for img in images]
    else:
        # 每张图像独立归一化
        return [normalizer.fit_transform(img) for img in images]


# 预定义的常用配置
PRESET_CONFIGS = {
    'satellite': {
        'method': 'robust',
        'lower_percentile': 2.0,
        'upper_percentile': 98.0,
        'per_channel': True
    },
    'aerial': {
        'method': 'adaptive',
        'auto_select': True
    },
    'low_contrast': {
        'method': 'standard',
        'per_channel': True
    },
    'uneven_lighting': {
        'method': 'windowed',
        'window_size': 64,
        'method': 'mean_std'
    },
    'high_dynamic_range': {
        'method': 'log_scaling'
    }
}


def normalize_with_preset(
    image: Union[np.ndarray, torch.Tensor],
    preset: str = 'satellite'
) -> Union[np.ndarray, torch.Tensor]:
    """
    使用预定义配置进行归一化
    
    Args:
        image: 输入图像
        preset: 预设名称 ('satellite', 'aerial', 'low_contrast', 'uneven_lighting', 'high_dynamic_range')
    
    Returns:
        归一化后的图像
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")
    
    config = PRESET_CONFIGS[preset].copy()
    method = config.pop('method')
    return normalize_for_inference(image, method=method, **config)
