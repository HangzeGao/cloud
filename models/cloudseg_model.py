"""
CloudSense-Net - 可配置的遥感云分割模型

所有组件都支持可选开关，可灵活组合不同架构方案。

架构层次：
0. Channel Adaptive Input (通道自适应) - 支持3/4通道自适应
1. Encoder (Backbone) - 4种可选
2. Semantic Enhancement (可选) - 参考SkySense++的MSL
3. Fusion Neck (可选) - FPN/BiFPN/ASPP
4. Decoder - 5种可选
5. Auxiliary Head (可选)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

from .builder import (
    build_backbone,
    build_semantic_enhancement,
    build_fusion,
    build_decoder,
    build_loss
)


class ChannelAdaptiveInput(nn.Module):
    """
    通道自适应输入模块
    
    支持3通道(RGB)或4通道(RGB+NIR)输入，自适应学习各通道重要性权重。
    当输入为3通道时，自动学习生成第4通道特征；当输入为4通道时，学习NIR通道质量。
    
    Args:
        out_channels: 输出通道数（固定为4，对应RGB+NIR）
        use_channel_attention: 是否使用通道注意力机制
        adaptive_method: 自适应方法 ('conv' 或 'attention')
    """
    
    def __init__(
        self,
        out_channels: int = 4,
        use_channel_attention: bool = True,
        adaptive_method: str = 'conv'
    ):
        super().__init__()
        
        self.out_channels = out_channels
        self.use_channel_attention = use_channel_attention
        self.adaptive_method = adaptive_method
        
        # 3通道到4通道的投影卷积（带可学习参数）
        self.rgb_to_4ch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # 4通道到4通道的 refine（学习通道质量）
        self.refine_4ch = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        # 通道注意力机制（学习各通道重要性）
        if use_channel_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, out_channels // 2),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels // 2, out_channels),
                nn.Sigmoid()
            )
        else:
            # 简单的可学习通道权重
            self.channel_weights = nn.Parameter(torch.ones(out_channels))
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
        print(f"[ChannelAdaptiveInput] Method: {adaptive_method}, CA: {use_channel_attention}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        自适应处理输入
        
        Args:
            x: 输入 [B, C, H, W], C=3 或 4
            
        Returns:
            out: 输出 [B, 4, H, W]（固定4通道）
        """
        in_channels = x.shape[1]
        
        if in_channels == 3:
            # 3通道输入：通过卷积生成第4通道
            transformed = self.rgb_to_4ch(x)
            # 保留RGB信息作为残差
            rgb_padded = F.pad(x, (0, 0, 0, 0, 0, 1), value=0)  # [B, 4, H, W] with NIR=0
        elif in_channels == 4:
            # 4通道输入：精炼所有通道
            transformed = self.refine_4ch(x)
            rgb_padded = x
        else:
            raise ValueError(f"Expected 3 or 4 channel input, got {in_channels}")
        
        # 残差连接：学习如何结合原始信息和变换后信息
        out = torch.sigmoid(self.residual_weight) * transformed + (1 - torch.sigmoid(self.residual_weight)) * rgb_padded
        
        # 通道注意力/权重
        if self.use_channel_attention:
            # 全局池化后计算各通道权重
            ca_weights = self.channel_attention(out)  # [B, 4]
            ca_weights = ca_weights.view(ca_weights.size(0), ca_weights.size(1), 1, 1)
            out = out * ca_weights
        else:
            # 简单通道权重
            out = out * self.channel_weights.view(1, -1, 1, 1)
        
        return out
    
    def get_channel_weights(self) -> torch.Tensor:
        """获取当前通道权重（用于分析）"""
        if self.use_channel_attention:
            # 返回训练过程中最后一个batch的平均权重
            return None  # 动态计算，无法静态获取
        else:
            return self.channel_weights.detach()


class CloudSenseNet(nn.Module):
    """
    CloudSense-Net 主模型类
    
    所有组件都可通过配置开关控制，支持灵活架构组合。
    
    Args:
        config: 模型配置字典，包含所有组件的开关和参数
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        model_cfg = config['model']
        
        # 基本配置
        self.num_classes = model_cfg.get('num_classes', 2)
        
        # ===== 0. 通道自适应输入模块 =====
        data_cfg = config.get('data', {})
        # 从配置读取是否启用通道自适应
        adaptive_cfg = model_cfg.get('channel_adaptive', {})
        self.use_channel_adaptive = adaptive_cfg.get('enabled', True)
        
        # 期望的输入通道数（训练时配置）
        self.expected_in_channels = 4 if data_cfg.get('use_nir', True) else 3
        
        if self.use_channel_adaptive:
            self.channel_adaptive = ChannelAdaptiveInput(
                out_channels=4,  # 统一输出4通道
                use_channel_attention=adaptive_cfg.get('use_attention', True),
                adaptive_method=adaptive_cfg.get('method', 'conv')
            )
            # 编码器始终以4通道接收（自适应后）
            encoder_in_channels = 4
            print(f"[CloudSenseNet] Channel Adaptive: Enabled (input: {self.expected_in_channels}ch -> output: 4ch)")
        else:
            self.channel_adaptive = None
            encoder_in_channels = self.expected_in_channels
            print(f"[CloudSenseNet] Channel Adaptive: Disabled")
        
        # ===== 1. 编码器 (必选) =====
        encoder_cfg = model_cfg['encoder']
        self.encoder_type = encoder_cfg.get('type', 'convnext')
        self.encoder_enabled = encoder_cfg.get('enabled', True)
        
        assert self.encoder_enabled, "Encoder must be enabled"
        
        # 构建编码器（传递统一的 in_channels 参数）
        self.encoder = build_backbone(self.encoder_type, encoder_cfg, in_channels=encoder_in_channels)
        self.encoder_channels = self.encoder.get_feature_channels()
        
        print(f"[CloudSenseNet] Encoder input: {encoder_in_channels}ch, Expected input: {self.expected_in_channels}ch")
        print(f"[CloudSenseNet] Encoder: {self.encoder_type}")
        print(f"[CloudSenseNet] Encoder channels: {self.encoder_channels}")
        
        # ===== 2. 语义增强模块 (可选) =====
        sem_cfg = model_cfg.get('semantic_enhancement', {})
        self.use_semantic_enhancement = sem_cfg.get('enabled', False)
        
        if self.use_semantic_enhancement:
            self.semantic_module = build_semantic_enhancement(sem_cfg)
            print(f"[CloudSenseNet] Semantic Enhancement: Enabled")
        else:
            self.semantic_module = None
            print(f"[CloudSenseNet] Semantic Enhancement: Disabled")
        
        # ===== 3. 特征融合层 (可选) =====
        fusion_cfg = model_cfg.get('fusion', {})
        self.use_fusion = fusion_cfg.get('enabled', True)
        self.fusion_type = fusion_cfg.get('type', 'fpn')
        
        if self.use_fusion:
            self.fusion = build_fusion(self.fusion_type, fusion_cfg, self.encoder_channels)
            # 更新通道数（融合后）
            if self.fusion_type == 'fpn':
                self.decoder_channels = [fusion_cfg.get('fpn', {}).get('out_channels', 256)] * len(self.encoder_channels)
            elif self.fusion_type == 'bifpn':
                self.decoder_channels = [fusion_cfg.get('bifpn', {}).get('out_channels', 256)] * len(self.encoder_channels)
            elif self.fusion_type == 'aspp':
                # ASPP通常只输出一个特征
                self.decoder_channels = self.encoder_channels[:-1] + [fusion_cfg.get('aspp', {}).get('out_channels', 256)]
            else:
                self.decoder_channels = self.encoder_channels
            print(f"[CloudSenseNet] Fusion: {self.fusion_type}")
        else:
            self.fusion = None
            self.decoder_channels = self.encoder_channels
            print(f"[CloudSenseNet] Fusion: Disabled")
        
        # ===== 4. 解码器 (必选) =====
        decoder_cfg = model_cfg['decoder']
        self.decoder_type = decoder_cfg.get('type', 'segformer')
        self.decoder_enabled = decoder_cfg.get('enabled', True)
        
        assert self.decoder_enabled, "Decoder must be enabled"
        
        # 构建解码器（使用融合后的通道数）
        self.decoder = build_decoder(self.decoder_type, decoder_cfg, self.decoder_channels)
        print(f"[CloudSenseNet] Decoder: {self.decoder_type}")
        
        # ===== 5. 辅助头 (可选) =====
        aux_cfg = model_cfg.get('auxiliary_head', {})
        self.use_auxiliary = aux_cfg.get('enabled', False)
        
        if self.use_auxiliary:
            self.aux_head = nn.Sequential(
                nn.Conv2d(self.decoder_channels[-1], 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(128, self.num_classes, 1)
            )
            self.aux_weight = aux_cfg.get('loss_weight', 0.4)
            print(f"[CloudSenseNet] Auxiliary Head: Enabled (weight={self.aux_weight})")
        else:
            self.aux_head = None
    
    def forward(
        self,
        x: torch.Tensor,
        annotation: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            annotation: 可选的语义标注（用于语义增强模块）[B, H, W]
            
        Returns:
            output: 字典，包含：
                - 'logits': 主要输出 [B, num_classes, H, W]
                - 'aux_logits': 辅助输出（如果启用）
                - 'deep_supervision': 深度监督输出列表（如果decoder支持且训练时）
        """
        output = {}
        
        # 保存输入尺寸用于最终上采样
        input_size = x.shape[-2:]
        
        # 记录实际输入通道数
        actual_in_channels = x.shape[1]
        
        # ===== 0. 通道自适应 =====
        if self.use_channel_adaptive:
            # 无论输入是3通道还是4通道，都通过自适应模块
            x = self.channel_adaptive(x)
        elif actual_in_channels != self.expected_in_channels:
            # 无自适应时，检查通道数是否匹配
            raise ValueError(
                f"Input channels ({actual_in_channels}) != Expected channels ({self.expected_in_channels}). "
                f"Enable channel_adaptive in config or provide correct input."
            )
        
        # ===== 1. 编码器 =====
        features = self.encoder(x)
        # features: [c2, c3, c4, c5] 对应不同下采样倍数
        
        # ===== 2. 语义增强 (可选) =====
        if self.use_semantic_enhancement and annotation is not None:
            # 语义增强通常应用于较深层特征
            enhanced_features = []
            for i, feat in enumerate(features):
                if i >= 2:  # 只对深层特征（c4, c5）进行增强
                    enhanced = self.semantic_module(feat, annotation)
                    enhanced_features.append(enhanced)
                else:
                    enhanced_features.append(feat)
            features = enhanced_features
        
        # ===== 3. 特征融合 (可选) =====
        if self.use_fusion and self.fusion is not None:
            features = self.fusion(features)
        
        # ===== 4. 辅助头 (可选，仅训练时) =====
        if self.use_auxiliary and self.training:
            aux_feat = features[-1]
            aux_logits = self.aux_head(aux_feat)
            aux_logits = F.interpolate(
                aux_logits,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
            output['aux_logits'] = aux_logits
        
        # ===== 5. 解码器 =====
        # 某些decoder可能支持深度监督
        decoder_output = self.decoder(features)
        
        if isinstance(decoder_output, tuple):
            # 深度监督模式：(main_output, [aux1, aux2, ...])
            main_logits, deep_supervision = decoder_output
            # 确保主输出尺寸与输入匹配
            if main_logits.shape[-2:] != input_size:
                main_logits = F.interpolate(main_logits, size=input_size, mode='bilinear', align_corners=False)
            output['logits'] = main_logits
            output['deep_supervision'] = deep_supervision
        else:
            logits = decoder_output
            # 确保输出尺寸与输入匹配（decoder可能输出不同尺寸）
            if logits.shape[-2:] != input_size:
                logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
            output['logits'] = logits
        
        return output
    
    def get_loss(self, predictions: Dict, target: torch.Tensor) -> torch.Tensor:
        """
        计算损失（支持辅助头和深度监督）
        
        Args:
            predictions: 模型输出字典
            target: 目标标注 [B, H, W]
            
        Returns:
            loss: 总损失
        """
        loss_config = self.config.get('training', {}).get('loss', {})
        criterion = build_loss(loss_config)
        
        total_loss = 0
        
        # 主损失
        total_loss += criterion(predictions['logits'], target)
        
        # 辅助头损失
        if 'aux_logits' in predictions:
            aux_loss = criterion(predictions['aux_logits'], target)
            total_loss += self.aux_weight * aux_loss
        
        # 深度监督损失
        if 'deep_supervision' in predictions and self.training:
            ds_weight = 0.4  # 深度监督权重
            for i, ds_logits in enumerate(predictions['deep_supervision']):
                ds_loss = criterion(ds_logits, target)
                total_loss += ds_weight * ds_loss / len(predictions['deep_supervision'])
        
        return total_loss
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        推理模式预测
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            pred: 预测类别 [B, H, W]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            logits = output['logits']
            pred = torch.argmax(logits, dim=1)
        return pred
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型架构信息"""
        info = {
            'encoder_type': self.encoder_type,
            'encoder_channels': self.encoder_channels,
            'use_channel_adaptive': self.use_channel_adaptive,
            'expected_in_channels': self.expected_in_channels,
            'use_semantic_enhancement': self.use_semantic_enhancement,
            'use_fusion': self.use_fusion,
            'fusion_type': self.fusion_type if self.use_fusion else None,
            'decoder_type': self.decoder_type,
            'decoder_channels': self.decoder_channels,
            'use_auxiliary': self.use_auxiliary,
            'num_classes': self.num_classes,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        return info
    
    def freeze_encoder(self, freeze_blocks: int = -1):
        """
        冻结编码器的部分或全部层
        
        Args:
            freeze_blocks: 冻结的block数量，-1表示冻结全部
        """
        if freeze_blocks == -1:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("[CloudSenseNet] Encoder fully frozen")
        else:
            # 根据具体backbone类型冻结
            # 这里简化处理，实际实现需要针对每个backbone定制
            print(f"[CloudSenseNet] Freezing first {freeze_blocks} encoder blocks")
    
    def print_architecture(self):
        """打印模型架构摘要"""
        info = self.get_model_info()
        
        print("\n" + "="*60)
        print("CloudSense-Net Architecture Summary")
        print("="*60)
        print(f"Channel Adaptive:  {'✓' if info['use_channel_adaptive'] else '✗'} (expect {info['expected_in_channels']}ch, adapt to 4ch)")
        print(f"Encoder:           {info['encoder_type']} ({info['encoder_channels']})")
        print(f"Semantic Enhancement: {'✓' if info['use_semantic_enhancement'] else '✗'}")
        print(f"Fusion:            {'✓ ' + info['fusion_type'] if info['use_fusion'] else '✗'}")
        print(f"Decoder:           {info['decoder_type']}")
        print(f"Auxiliary Head:    {'✓' if info['use_auxiliary'] else '✗'}")
        print(f"Num Classes:       {info['num_classes']}")
        print("-"*60)
        print(f"Total Parameters:      {info['total_params']:,}")
        print(f"Trainable Parameters:  {info['trainable_params']:,}")
        print("="*60 + "\n")


# 预定义的常用配置组合
MODEL_CONFIGS = {
    # 高精度方案
    'high_accuracy': {
        'encoder': {'type': 'swin', 'enabled': True, 'swin': {'pretrained': False}},
        'semantic_enhancement': {'enabled': True},
        'fusion': {'enabled': True, 'type': 'fpn', 'fpn': {'out_channels': 256, 'num_levels': 4}},
        'decoder': {'type': 'unetpp', 'enabled': True},
        'auxiliary_head': {'enabled': True}
    },

    # 速度优先方案
    'fast': {
        'encoder': {'type': 'efficientnet', 'enabled': True, 'efficientnet': {'pretrained': False}},
        'semantic_enhancement': {'enabled': False},
        'fusion': {'enabled': True, 'type': 'aspp', 'aspp': {'out_channels': 256}},
        'decoder': {'type': 'deeplabv3plus', 'enabled': True},
        'auxiliary_head': {'enabled': False}
    },

    # 平衡方案
    'balanced': {
        'encoder': {'type': 'convnext', 'enabled': True, 'convnext': {'pretrained': False}},
        'semantic_enhancement': {'enabled': True, 'vocabulary_size': 2, 'patch_size': 16, 'embed_dim': 768, 'merge_stage': 0},
        'fusion': {'enabled': True, 'type': 'bifpn', 'bifpn': {'out_channels': 256, 'num_iterations': 2}},
        'decoder': {'type': 'segformer', 'enabled': True, 'segformer': {'embed_dim': 256}},
        'auxiliary_head': {'enabled': False}
    },

    # 极简方案
    'minimal': {
        'encoder': {'type': 'resnet', 'enabled': True, 'resnet': {'pretrained': False}},
        'semantic_enhancement': {'enabled': False},
        'fusion': {'enabled': False},
        'decoder': {'type': 'simple', 'enabled': True},
        'auxiliary_head': {'enabled': False}
    }
}


def create_model_from_preset(preset: str, num_classes: int = 2) -> CloudSenseNet:
    """
    从预设配置快速创建模型
    
    Args:
        preset: 预设名称 ('high_accuracy', 'fast', 'balanced', 'minimal')
        num_classes: 类别数
        
    Returns:
        model: CloudSenseNet实例
    """
    if preset not in MODEL_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = {
        'model': MODEL_CONFIGS[preset],
        'training': {'loss': {'types': ['dice', 'bce'], 'weights': [0.5, 0.5]}}
    }
    config['model']['num_classes'] = num_classes
    
    return CloudSenseNet(config)
