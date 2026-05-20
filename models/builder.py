"""
模型构建器 - 根据配置自动构建不同的模型组件
"""
import torch.nn as nn
from typing import Dict, Any


def build_model(config: Dict[str, Any]) -> nn.Module:
    """根据配置构建完整的CloudSenseNet模型"""
    from .cloudseg_model import CloudSenseNet
    return CloudSenseNet(config)


def build_backbone(backbone_type: str, config: Dict[str, Any], in_channels: int = 3) -> nn.Module:
    """构建编码器/主干网络
    
    Args:
        backbone_type: 主干网络类型
        config: 配置字典
        in_channels: 输入通道数 (3=RGB, 4=RGB+NIR)
    """
    if backbone_type == "swin":
        from .backbones.swin_backbone import SwinBackbone
        return SwinBackbone(in_channels=in_channels, **config.get('swin', {}))
    elif backbone_type == "convnext":
        from .backbones.convnext_backbone import ConvNeXtBackbone
        return ConvNeXtBackbone(in_channels=in_channels, **config.get('convnext', {}))
    elif backbone_type == "efficientnet":
        from .backbones.efficientnet_backbone import EfficientNetBackbone
        return EfficientNetBackbone(in_channels=in_channels, **config.get('efficientnet', {}))
    elif backbone_type == "resnet":
        from .backbones.resnet_backbone import ResNetBackbone
        return ResNetBackbone(in_channels=in_channels, **config.get('resnet', {}))
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


def build_semantic_enhancement(config: Dict[str, Any]) -> nn.Module:
    """构建语义增强模块（MSL）"""
    from .necks.semantic_enhancement import SemanticEnhancementModule
    # 移除非模型参数
    model_config = {k: v for k, v in config.items() if k != 'enabled'}
    return SemanticEnhancementModule(**model_config)


def build_fusion(fusion_type: str, config: Dict[str, Any], in_channels: list) -> nn.Module:
    """构建特征融合层"""
    if fusion_type == "fpn":
        from .necks.fpn_fusion import FPNFusion
        return FPNFusion(in_channels=in_channels, **config.get('fpn', {}))
    elif fusion_type == "bifpn":
        from .necks.bifpn_fusion import BiFPNFusion
        return BiFPNFusion(in_channels=in_channels, **config.get('bifpn', {}))
    elif fusion_type == "aspp":
        from .necks.aspp_fusion import ASPPFusion
        return ASPPFusion(in_channels=in_channels[-1], **config.get('aspp', {}))
    elif fusion_type == "none" or fusion_type is None:
        return None
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


def build_decoder(decoder_type: str, config: Dict[str, Any], encoder_channels: list) -> nn.Module:
    """构建解码器"""
    if decoder_type == "unetpp":
        from .decoders.unetpp_decoder import UNetPPDecoder
        return UNetPPDecoder(encoder_channels=encoder_channels, **config.get('unetpp', {}))
    elif decoder_type == "deeplabv3plus":
        from .decoders.deeplabv3plus_decoder import DeepLabV3PlusDecoder
        return DeepLabV3PlusDecoder(encoder_channels=encoder_channels, **config.get('deeplabv3plus', {}))
    elif decoder_type == "segformer":
        from .decoders.segformer_decoder import SegFormerDecoder
        return SegFormerDecoder(encoder_channels=encoder_channels, **config.get('segformer', {}))
    elif decoder_type == "upernet":
        from .decoders.upernet_decoder import UperNetDecoder
        return UperNetDecoder(encoder_channels=encoder_channels, **config.get('upernet', {}))
    elif decoder_type == "simple":
        from .decoders.upernet_decoder import SimpleDecoder
        return SimpleDecoder(encoder_channels=encoder_channels)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")


def build_loss(loss_config: Dict[str, Any]) -> nn.Module:
    """构建损失函数"""
    from .heads.losses import CombinedLoss
    return CombinedLoss(loss_config)
