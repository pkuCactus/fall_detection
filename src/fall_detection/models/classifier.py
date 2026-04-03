import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FallClassifier(nn.Module):
    """轻量融合姿态分类器."""

    def __init__(self, model_path: str = None):
        super().__init__()
        # 图像分支: 3 -> 16 -> 32, 每层 stride=2, GAP
        self.img_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.img_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        # 关键点分支: 51 -> 32
        self.kpt_fc = nn.Linear(17 * 3, 32)
        # 运动分支: 8 -> 8
        self.motion_fc = nn.Linear(8, 8)
        # 融合: 72 -> 32 -> 1
        self.fusion_fc1 = nn.Linear(32 + 32 + 8, 32)
        self.dropout = nn.Dropout(0.3)
        self.fusion_fc2 = nn.Linear(32, 1)

        # 加载预训练权重
        if model_path and os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.eval()
        elif os.path.exists("train/classifier/best.pt"):
            self.load_state_dict(torch.load("train/classifier/best.pt", map_location='cpu'))
            self.eval()

    def forward(self, roi, kpts, motion):
        """
        前向推理.

        Args:
            roi: torch.Tensor, shape (B, 3, 96, 96) or (3, 96, 96).
            kpts: torch.Tensor, shape (B, 17, 3) or (17, 3).
            motion: torch.Tensor, shape (B, 8) or (8,).

        Returns:
            torch.Tensor: 概率值, shape (B, 1) 或标量.
        """
        # 处理单样本输入（numpy 或 torch, 无 batch 维度）
        if isinstance(roi, np.ndarray):
            roi = torch.from_numpy(roi)
        if isinstance(kpts, np.ndarray):
            kpts = torch.from_numpy(kpts)
        if isinstance(motion, np.ndarray):
            motion = torch.from_numpy(motion)

        # 确保是 float32
        roi = roi.float()
        kpts = kpts.float()
        motion = motion.float()

        # 添加 batch 维度
        single = False
        if roi.dim() == 3:
            roi = roi.unsqueeze(0)
            kpts = kpts.unsqueeze(0)
            motion = motion.unsqueeze(0)
            single = True

        # 图像分支
        x_img = F.relu(self.img_conv1(roi))   # (B,16,48,48)
        x_img = F.relu(self.img_conv2(x_img))  # (B,32,24,24)
        x_img = F.adaptive_avg_pool2d(x_img, (1, 1)).view(x_img.size(0), -1)  # (B,32)

        # 关键点分支
        x_kpt = F.relu(self.kpt_fc(kpts.view(kpts.size(0), -1)))  # (B,32)

        # 运动分支
        x_motion = F.relu(self.motion_fc(motion))  # (B,8)

        # 融合
        x = torch.cat([x_img, x_kpt, x_motion], dim=1)  # (B,72)
        x = F.relu(self.fusion_fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fusion_fc2(x))

        if single:
            return x.item()
        return x
