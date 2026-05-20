"""
语义增强模块 (Semantic Enhancement Module)
参考 SkySense++ 的 MSL (Masked Semantic Learning) 设计
将像素级语义标注嵌入到特征中
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SemanticEnhancementModule(nn.Module):
    """
    语义增强模块 - 可选开关控制
    
    功能：
    1. 将语义标注转换为token嵌入
    2. 与图像特征融合（加权融合或拼接融合）
    3. 支持掩码操作（用于自监督预训练）
    
    Args:
        vocabulary_size: 语义类别数（云/背景 = 2）
        patch_size: patch大小
        embed_dim: 特征嵌入维度
        merge_stage: 融合阶段 (0:直接相加, 1+:分层融合)
    """
    
    def __init__(
        self,
        vocabulary_size: int = 2,
        patch_size: int = 16,
        embed_dim: int = 768,
        merge_stage: int = 0,
        use_mask_token: bool = False
    ):
        super().__init__()
        
        self.vocabulary_size = vocabulary_size + 1  # +1 for background/ignore
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.merge_stage = merge_stage
        self.use_mask_token = use_mask_token
        
        # 语义词汇表嵌入 - 每个类别对应一个可学习的token
        self.vocabulary_token = nn.Parameter(
            torch.zeros(self.vocabulary_size, embed_dim)
        )
        
        # 可选的掩码token（用于masked learning）
        if use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # 位置权重（用于加权平均池化）
        self.vocabulary_weight = nn.Parameter(
            torch.zeros(1, patch_size * patch_size)
        )
        
        # 融合投影层
        if merge_stage > 0:
            # 分层融合：将特征和语义token拼接后投影
            self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.vocabulary_token, std=0.02)
        nn.init.trunc_normal_(self.vocabulary_weight, std=0.02)
    
    def create_semantic_tokens(
        self,
        annotation: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        将语义标注图转换为语义token
        
        Args:
            annotation: [B, H, W] 语义标注图，值为类别索引
            mask: [B, H, W] 可选的掩码（用于masked learning）
            
        Returns:
            semantic_tokens: [B, H//patch_size, W//patch_size, embed_dim]
        """
        B, H, W = annotation.shape
        
        # 确保尺寸可被patch_size整除
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image size ({H}, {W}) must be divisible by patch_size {self.patch_size}"
        
        # 通过embedding lookup将标注转换为特征
        # annotation: [B, H, W] -> 展平为 [B*H*W]
        flat_annotation = annotation.reshape(-1)  # [B*H*W]
        
        # 索引查找: [B*H*W, embed_dim] -> reshape回 [B, H, W, embed_dim]
        semantic_features = F.embedding(flat_annotation, self.vocabulary_token)
        semantic_features = semantic_features.reshape(B, H, W, self.embed_dim)
        
        # 计算位置权重（学习每个patch内不同位置的重要性）
        weight = F.softmax(self.vocabulary_weight, dim=-1)  # [1, patch_size^2]
        weight = weight * (self.patch_size ** 2)  # 归一化后缩放
        
        # reshape weight为空间形式 [1, patch_size, patch_size, 1]
        weight = weight.reshape(1, self.patch_size, self.patch_size, 1)
        nph, npw = H // self.patch_size, W // self.patch_size
        weight = weight.repeat(1, nph, npw, 1)  # [1, H, W, 1]
        
        # 应用位置权重
        semantic_features = semantic_features * weight  # [B, H, W, embed_dim]
        
        # 使用平均池化将特征聚合到patch级别
        # 转换维度顺序用于池化: [B, H, W, C] -> [B, C, H, W]
        semantic_features = semantic_features.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 平均池化: [B, C, H, W] -> [B, C, H//patch_size, W//patch_size]
        semantic_tokens = F.avg_pool2d(
            semantic_features,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )  # [B, C, H//P, W//P]
        
        # 转回 [B, H//P, W//P, C]
        semantic_tokens = semantic_tokens.permute(0, 2, 3, 1)
        
        # 应用掩码（如果提供）
        if mask is not None and self.use_mask_token:
            # 将mask下采样到token级别
            mask_tokens = F.avg_pool2d(
                mask.unsqueeze(1).float(),
                kernel_size=self.patch_size,
                stride=self.patch_size
            )  # [B, 1, H//P, W//P]
            
            # 二值化
            mask_tokens = (mask_tokens > 0.5).float()
            
            # 应用mask token
            mask_token_expanded = self.mask_token.expand(B, mask_tokens.shape[2], mask_tokens.shape[3], -1)
            semantic_tokens = semantic_tokens * (1 - mask_tokens.permute(0, 2, 3, 1)) + \
                             mask_token_expanded * mask_tokens.permute(0, 2, 3, 1)
        
        return semantic_tokens
    
    def forward(
        self,
        features: torch.Tensor,
        annotation: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        前向传播：将语义信息与图像特征融合
        
        Args:
            features: [B, C, H, W] 图像特征
            annotation: [B, H_orig, W_orig] 原始尺寸的语义标注
            mask: [B, H_orig, W_orig] 可选掩码
            
        Returns:
            enhanced_features: [B, C, H, W] 增强后的特征
        """
        B, C, H, W = features.shape
        
        # 确保annotation尺寸正确（上采样或下采样到feature尺寸）
        if annotation.shape[-2:] != (H, W):
            annotation = F.interpolate(
                annotation.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()
        
        if mask is not None and mask.shape[-2:] != (H, W):
            mask = F.interpolate(
                mask.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1)
        
        # 创建语义token
        semantic_tokens = self.create_semantic_tokens(annotation, mask)
        # [B, H, W, C] -> [B, C, H, W]
        semantic_tokens = semantic_tokens.permute(0, 3, 1, 2)
        
        # 根据merge_stage选择融合方式
        if self.merge_stage == 0:
            # 直接加权融合
            enhanced_features = (features + semantic_tokens) * 0.5
        else:
            # 分层融合：拼接后投影
            # [B, C, H, W] 和 [B, C, H, W] -> [B, 2*C, H, W]
            concat_features = torch.cat([features, semantic_tokens], dim=1)
            # 调整维度用于线性层: [B, 2*C, H, W] -> [B, H, W, 2*C]
            concat_features = concat_features.permute(0, 2, 3, 1)
            # 投影: [B, H, W, 2*C] -> [B, H, W, C]
            enhanced_features = self.fusion_proj(concat_features)
            # 调整回 [B, C, H, W]
            enhanced_features = enhanced_features.permute(0, 3, 1, 2)
        
        return enhanced_features


class LightweightSemanticEnhancement(nn.Module):
    """
    轻量级语义增强模块 - 减少计算开销
    适用于对速度要求更高的场景
    """
    
    def __init__(
        self,
        vocabulary_size: int = 2,
        embed_dim: int = 256,
        reduction_ratio: int = 4
    ):
        super().__init__()
        
        self.vocabulary_size = vocabulary_size + 1
        self.embed_dim = embed_dim
        
        # 轻量级嵌入（使用1x1卷积代替复杂的token embedding）
        reduced_dim = embed_dim // reduction_ratio
        self.semantic_embed = nn.Sequential(
            nn.Conv2d(vocabulary_size + 1, reduced_dim, 1),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_dim, embed_dim, 1)
        )
        
        # 通道注意力融合
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim * 2, embed_dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 16, embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        features: torch.Tensor,
        annotation: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]
            annotation: [B, H, W] 类别索引
        """
        B, C, H, W = features.shape
        
        # 将annotation转换为one-hot
        annotation_onehot = F.one_hot(
            annotation.long(),
            num_classes=self.vocabulary_size
        ).permute(0, 3, 1, 2).float()  # [B, num_classes, H, W]
        
        # 调整尺寸
        if annotation_onehot.shape[-2:] != (H, W):
            annotation_onehot = F.interpolate(
                annotation_onehot,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        # 语义嵌入
        semantic_feat = self.semantic_embed(annotation_onehot)  # [B, C, H, W]
        
        # 通道注意力融合
        concat_feat = torch.cat([features, semantic_feat], dim=1)  # [B, 2*C, H, W]
        attention = self.channel_attention(concat_feat)  # [B, C, 1, 1]
        
        enhanced_features = features * attention + semantic_feat * (1 - attention)
        
        return enhanced_features
