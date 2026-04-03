"""Data augmentation utilities for fall detection training."""

from typing import List, Dict, Any, Optional

import cv2
import numpy as np




class RandomMask:
    """随机mask掉图像中间一小部分，模拟遮挡."""

    def __init__(self, mask_ratio: float = 0.25, p: float = 0.3):
        self.mask_ratio = mask_ratio
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() > self.p:
            return img

        h, w = img.shape[:2]
        mask_h = int(h * self.mask_ratio)
        mask_w = int(w * self.mask_ratio)

        cx = np.random.randint(w // 4, 3 * w // 4)
        cy = np.random.randint(h // 4, 3 * h // 4)

        x1 = max(0, cx - mask_w // 2)
        y1 = max(0, cy - mask_h // 2)
        x2 = min(w, cx + mask_w // 2)
        y2 = min(h, cy + mask_h // 2)

        img = img.copy()
        img[y1:y2, x1:x2] = np.random.randint(0, 256, img[y1:y2, x1:x2].shape, dtype=img.dtype)

        return img



class RandomCropWithPadding:
    """随机crop，支持内缩和外扩."""

    def __init__(self, shrink_max: int = 3, expand_max: int = 25):
        self.shrink_max = shrink_max
        self.expand_max = expand_max

    def __call__(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox

        # 随机内缩(-shrink_max ~ 0) 或 外扩(0 ~ expand_max)
        margin = np.random.randint(-self.shrink_max, self.expand_max + 1)

        x1_new = max(0, int(x1 - margin))
        y1_new = max(0, int(y1 - margin))
        x2_new = min(w, int(x2 + margin))
        y2_new = min(h, int(y2 + margin))

        if x2_new <= x1_new or y2_new <= y1_new:
            return img[int(y1):int(y2), int(x1):int(x2)]

        return img[y1_new:y2_new, x1_new:x2_new]


class LetterBoxResize:
    """保持长宽比的resize，短边padding到目标大小."""

    def __init__(self, target_size: int = 96, fill_value: int = 114):
        self.target_size = target_size
        self.fill_value = fill_value

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if len(img.shape) == 3:
            result = np.full((self.target_size, self.target_size, 3), self.fill_value, dtype=np.uint8)
            y_offset = (self.target_size - new_h) // 2
            x_offset = (self.target_size - new_w) // 2
            result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        else:
            result = np.full((self.target_size, self.target_size), self.fill_value, dtype=np.uint8)
            y_offset = (self.target_size - new_h) // 2
            x_offset = (self.target_size - new_w) // 2
            result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return result


class TrainingAugmentation:
    """训练时的数据增强."""

    def __init__(self, aug_cfg: Dict[str, Any]):
        self.cfg = aug_cfg
        self.color_jitter_cfg = aug_cfg.get('color_jitter') or {}
        self.random_gray_cfg = aug_cfg.get('random_gray') or {}
        self.random_rotation_cfg = aug_cfg.get('random_rotation') or {}
        self.random_mask_cfg = aug_cfg.get('random_mask') or {}
        self.horizontal_flip_cfg = aug_cfg.get('horizontal_flip') or {}

        # Random Mask
        if self.random_mask_cfg.get('enabled', False):
            self.random_mask = RandomMask(
                mask_ratio=self.random_mask_cfg.get('mask_ratio', 0.2),
                p=self.random_mask_cfg.get('p', 0.3)
            )

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """颜色抖动."""
        brightness = self.color_jitter_cfg.get('brightness', 0.3)
        contrast = self.color_jitter_cfg.get('contrast', 0.3)
        saturation = self.color_jitter_cfg.get('saturation', 0.3)

        # 亮度
        alpha = np.random.uniform(1 - brightness, 1 + brightness)
        img = np.clip(img * alpha, 0, 255).astype(np.uint8)

        # 对比度
        beta = np.random.uniform(1 - contrast, 1 + contrast)
        mean = img.mean()
        img = np.clip((img - mean) * beta + mean, 0, 255).astype(np.uint8)

        # 饱和度（HSV空间）
        if np.random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= np.random.uniform(1 - saturation, 1 + saturation)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return img

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img: numpy array, shape (H, W, 3), BGR, uint8
        Returns:
            augmented image
        """
        # 水平翻转
        if self.horizontal_flip_cfg.get('enabled', False):
            if np.random.random() < self.horizontal_flip_cfg.get('p', 0.5):
                img = cv2.flip(img, 1)

        # 颜色抖动
        if self.color_jitter_cfg.get('enabled', False):
            img = self._color_jitter(img)

        # 随机灰度
        if self.random_gray_cfg.get('enabled', False):
            if np.random.random() < self.random_gray_cfg.get('p', 0.2):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 随机旋转
        if self.random_rotation_cfg.get('enabled', False):
            if np.random.random() < self.random_rotation_cfg.get('p', 0.3):
                angle_range = self.random_rotation_cfg.get('angle_range', [-5, 5])
                angle = np.random.uniform(angle_range[0], angle_range[1])
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                border_val = self.cfg.get('letterbox_fill_value', 114)
                img = cv2.warpAffine(
                    img, M, (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(border_val, border_val, border_val)
                )

        # 随机mask
        if self.random_mask_cfg.get('enabled', False):
            img = self.random_mask(img)

        return img


