"""
Utilities Package
=================

提供CloudSense-Net训练和推理所需的各种工具函数和类。

主要模块:
- config: 配置管理
- metrics: 评估指标
- inference_utils: 推理工具
- inference_normalizer: 推理归一化
- nir_generator: 伪NIR生成

快速导入:
    >>> from utils import load_config, SegmentationMetrics
    >>> from utils import sliding_window_inference, generate_pseudo_nir
"""

# =============================================================================
# 配置管理
# =============================================================================

from .config import (
    load_config,
    save_config,
    merge_config,
    deep_update,
    get_config_value,
    set_config_value,
    get_default_config,
    validate_config,
    ConfigManager
)

# =============================================================================
# 评估指标
# =============================================================================

from .metrics import (
    SegmentationMetrics,
    SegmentationMetricsResult,
    AverageMeter,
    MetricsTracker,
    compute_iou_numpy,
    compute_dice_numpy,
    compute_pixel_accuracy,
    compute_confusion_matrix
)

# =============================================================================
# 推理工具
# =============================================================================

from .inference_utils import (
    # 推理函数
    sliding_window_inference,
    whole_image_inference,
    multi_scale_inference,
    patch_based_inference,
    
    # 权重生成
    create_gaussian_weight,
    create_linear_weight,
    
    # 归一化集成
    apply_inference_normalization,
    get_normalization_recommendation,
    create_inference_transform,
    normalize_for_inference,
    normalize_with_preset,
    batch_normalize,
    create_normalizer,
    NormalizationMethod,
    PRESET_CONFIGS,
    
    # 配置类
    InferenceMode,
    InferenceConfig
)

# =============================================================================
# 伪NIR生成
# =============================================================================

from .nir_generator import (
    generate_pseudo_nir,
    get_available_methods,
    get_method_info,
    NIRMethod,
    NIRGeneratorFactory,
    BaseNIRGenerator,
    PhysicalNIRGenerator,
    VegetationIndexNIRGenerator,
    GuidedFilterNIRGenerator,
    ContextAwareNIRGenerator,
    EnsembleNIRGenerator
)


# =============================================================================
# __all__ 定义
# =============================================================================

__all__ = [
    # config
    'load_config',
    'save_config',
    'merge_config',
    'deep_update',
    'get_config_value',
    'set_config_value',
    'get_default_config',
    'validate_config',
    'ConfigManager',
    
    # metrics
    'SegmentationMetrics',
    'SegmentationMetricsResult',
    'AverageMeter',
    'MetricsTracker',
    'compute_iou_numpy',
    'compute_dice_numpy',
    'compute_pixel_accuracy',
    'compute_confusion_matrix',
    
    # inference_utils
    'sliding_window_inference',
    'whole_image_inference',
    'multi_scale_inference',
    'patch_based_inference',
    'create_gaussian_weight',
    'create_linear_weight',
    'apply_inference_normalization',
    'get_normalization_recommendation',
    'create_inference_transform',
    'normalize_for_inference',
    'normalize_with_preset',
    'batch_normalize',
    'create_normalizer',
    'NormalizationMethod',
    'PRESET_CONFIGS',
    'InferenceMode',
    'InferenceConfig',
    
    # nir_generator
    'generate_pseudo_nir',
    'get_available_methods',
    'get_method_info',
    'NIRMethod',
    'NIRGeneratorFactory',
    'BaseNIRGenerator',
    'PhysicalNIRGenerator',
    'VegetationIndexNIRGenerator',
    'GuidedFilterNIRGenerator',
    'ContextAwareNIRGenerator',
    'EnsembleNIRGenerator'
]


# 版本信息
__version__ = '1.0.0'


def print_utils_summary():
    """打印utils包的功能摘要"""
    print("=" * 60)
    print("CloudSense-Net Utils Package")
    print("=" * 60)
    print("\n配置管理:")
    print("  - load_config, save_config, merge_config")
    print("  - ConfigManager")
    print("\n评估指标:")
    print("  - SegmentationMetrics (IoU, Dice, Pixel Accuracy)")
    print("  - AverageMeter, MetricsTracker")
    print("\n推理工具:")
    print("  - sliding_window_inference")
    print("  - multi_scale_inference")
    print("  - whole_image_inference")
    print("  - patch_based_inference")
    print("\n归一化:")
    print("  - apply_inference_normalization")
    print("  - NormalizationMethod (adaptive, robust, standard, etc.)")
    print("\n伪NIR生成:")
    print("  - generate_pseudo_nir")
    print("  - Physical/Vegetation/Guided/Context/Ensemble Generators")
    print("=" * 60)
