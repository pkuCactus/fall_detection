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

    def __init__(self, model_path: str = None, dropout: float = 0.1):
        super().__init__()

        self.model = nn.Sequential(
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
        self.model2 = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

        # 加载预训练权重
        if model_path and os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.eval()
        elif os.path.exists("train/simple_classifier/best.pt"):
            self.load_state_dict(torch.load("train/simple_classifier/best.pt", map_location='cpu'))
            self.eval()

    def forward(self, roi):
        """
        前向推理.

        Args:
            roi: torch.Tensor, shape (B, 3, 96, 96)

        Returns:
            torch.Tensor: 概率值, shape (B, 2) 或标量.
        """
        out = self.model(roi)
        out = self.model2(out)
        return out