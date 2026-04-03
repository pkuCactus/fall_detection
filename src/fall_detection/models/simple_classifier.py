import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBasicBlock(nn.Module):
    """BasicBlock: 两个3x3卷积层."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = x + self.model(x)
        return out


class SimpleResBlock(nn.Module):
    """ResBlock: 包含两个BasicBlock的残差块."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.model(x) + self.model2(x)
        return out


class SimpleFallClassifier(nn.Module):
    """轻量级单分支跌倒分类器（仅图像输入）.

    模型结构:
    - 两个 3x3 卷积层, stride=2, + BN + ReLU
    - ResBlock + BasicBlock + ResBlock + BasicBlock
    - Avg Pool 得到特征
    - 分类头: Linear + BN + Dropout + ReLU + Linear
    """

    def __init__(self, model_path: str = None, dropout: float = 0.1, num_classes: int = 2, fall_class_idx: int = 1):
        super().__init__()

        self.num_classes = num_classes
        self.fall_class_idx = fall_class_idx

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SimpleResBlock(64, 64, 1),
            nn.ReLU(inplace=True),
            SimpleBasicBlock(64, 64),
            nn.ReLU(inplace=True),
            SimpleResBlock(64, 128, 2),
            nn.ReLU(inplace=True),
            SimpleBasicBlock(128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

        # 加载预训练权重 (使用try/except避免TOCTOU问题)
        if model_path:
            try:
                self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
                self.eval()
            except FileNotFoundError as e:
                print(f"Warning: Failed to load model from {model_path}: {e}")
        else:
            # 尝试加载默认路径
            try:
                self.load_state_dict(torch.load("outputs/simple_classifier/best.pt", map_location='cpu'), strict=True)
                self.eval()
            except FileNotFoundError:
                print("Warning: Default model path not found, using random initialization.")  # 默认路径不存在，使用随机初始化

    def forward(self, roi):
        """
        前向推理.

        Args:
            roi: torch.Tensor, shape (B, 3, 96, 96)

        Returns:
            torch.Tensor: logits, shape (B, num_classes).
        """
        out = self.backbone(roi)
        out = self.classifier_head(out)
        return out
