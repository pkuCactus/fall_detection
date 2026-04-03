import argparse
import ast
import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import tqdm
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, "src")
from fall_detection.models.simple_classifier import SimpleFallClassifier


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


class CocoFallDataset(Dataset):
    """从COCO格式标注加载数据，支持动态crop和数据增强."""

    def __init__(
        self,
        image_dir: str,
        coco_json: str,
        transform: Optional[callable] = None,
        target_size: int = 96,
        use_letterbox: bool = True,
        fill_value: int = 114,
        person_category_id: int = 1,
        fall_category_id: int = 1,
        shrink_max: int = 3,
        expand_max: int = 25,
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        self.use_letterbox = use_letterbox
        self.person_category_id = person_category_id

        self.cropper = RandomCropWithPadding(shrink_max=shrink_max, expand_max=expand_max)
        self.letterbox = LetterBoxResize(target_size=target_size, fill_value=fill_value) if use_letterbox else None

        # 加载COCO格式数据
        with open(coco_json, 'r') as f:
            coco_data = json.load(f)

        self.images = {img['id']: img for img in coco_data['images']}
        self.categories = {cat['id']: cat for cat in coco_data.get('categories', [])}

        # 构建样本列表 (image_id, bbox, label)
        self.samples: List[Tuple[int, List[float], int]] = []
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            bbox = ann['bbox']  # [x, y, w, h] COCO格式
            # 转换为 [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h

            # 判断标签
            label = 0
            if ann.get('category_id') == fall_category_id:
                label = 1
            elif ann.get('attributes', {}).get('fall', 0) == 1:
                label = 1
            elif ann.get('fall', 0) == 1:
                label = 1

            self.samples.append((img_id, [x1, y1, x2, y2], label))

        # 缓存图像路径
        self.img_path_cache: Dict[int, str] = {}

    def __len__(self):
        return len(self.samples)

    def _get_image(self, img_id: int) -> np.ndarray:
        """加载图像，带缓存."""
        if img_id not in self.img_path_cache:
            img_info = self.images[img_id]
            file_name = img_info['file_name']
            img_path = os.path.join(self.image_dir, file_name)
            self.img_path_cache[img_id] = img_path
        else:
            img_path = self.img_path_cache[img_id]

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        return img

    def __getitem__(self, idx):
        img_id, bbox, label = self.samples[idx]

        # 加载完整图像
        img = self._get_image(img_id)

        # 随机crop ROI
        roi = self.cropper(img, bbox)

        # 数据增强
        if self.transform:
            roi = self.transform(roi)

        # Resize
        if self.use_letterbox and self.letterbox:
            roi = self.letterbox(roi)
        else:
            roi = cv2.resize(roi, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # To tensor: (H, W, C) -> (C, H, W), normalize to [0, 1]
        roi = torch.from_numpy(roi).permute(2, 0, 1).float() / 255.0

        return roi, torch.tensor(label, dtype=torch.long)


class VOCFallDataset(Dataset):
    """从Pascal VOC格式标注加载数据，支持动态crop和数据增强.

    支持多目录结构，自动从 ImageSets/Main/{split}.txt 加载图像列表。
    目录结构示例:
        data_dir/
        ├── JPEGImages/      # 图像文件
        ├── Annotations/     # XML标注
        └── ImageSets/
            └── Main/
                ├── train.txt
                ├── val.txt
                └── test.txt
    """

    def __init__(
        self,
        data_dirs: List[str],
        split: str = "train",
        transform: Optional[callable] = None,
        target_size: int = 96,
        use_letterbox: bool = True,
        fill_value: int = 114,
        fall_classes: Optional[List[str]] = None,
        normal_classes: Optional[List[str]] = None,
        shrink_max: int = 3,
        expand_max: int = 25,
    ):
        """
        Args:
            data_dirs: 数据目录列表，每个目录应包含 JPEGImages/, Annotations/, ImageSets/Main/
            split: 数据集划分 (train/val/test)，用于自动加载 ImageSets/Main/{split}.txt
            transform: 数据增强变换
            target_size: 目标输入尺寸
            use_letterbox: 是否使用letterbox保持长宽比
            fill_value: letterbox填充值
            fall_classes: 跌倒类别名称列表 (这些类别映射到label=1)，默认["fall"]
            normal_classes: 非跌倒类别名称列表 (这些类别映射到label=0)，默认None表示其他所有类别
            shrink_max: 最大内缩像素
            expand_max: 最大外扩像素
        """
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.use_letterbox = use_letterbox

        # 类别映射配置
        self.fall_classes = set(c.lower() for c in (fall_classes or ["fall"]))
        self.normal_classes = set(c.lower() for c in normal_classes) if normal_classes else None

        self.cropper = RandomCropWithPadding(shrink_max=shrink_max, expand_max=expand_max)
        self.letterbox = LetterBoxResize(target_size=target_size, fill_value=fill_value) if use_letterbox else None

        # 构建样本列表: (image_path, bbox, label)
        self.samples: List[Tuple[str, List[float], int]] = []

        # 为每个数据目录加载样本
        for data_dir in self.data_dirs:
            self._load_from_dir(data_dir)

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in VOC dataset. data_dirs={self.data_dirs}, split={split}")

    def _load_from_dir(self, data_dir: str):
        """从单个数据目录加载样本."""
        image_dir = os.path.join(data_dir, "JPEGImages")
        anno_dir = os.path.join(data_dir, "Annotations")
        image_set_file = os.path.join(data_dir, "ImageSets", "Main", f"{self.split}.txt")

        if not os.path.exists(anno_dir):
            print(f"Warning: Annotations directory not found: {anno_dir}")
            return

        # 加载图像列表
        if os.path.exists(image_set_file):
            with open(image_set_file, 'r') as f:
                image_ids = [line.strip().split()[0] for line in f if line.strip()]
        else:
            print(f"Warning: Image set file not found: {image_set_file}, scanning all XML files")
            image_ids = [
                os.path.splitext(f)[0]
                for f in os.listdir(anno_dir)
                if f.endswith('.xml')
            ]

        # 解析每个图像的标注
        for img_id in image_ids:
            anno_path = os.path.join(anno_dir, f"{img_id}.xml")
            if not os.path.exists(anno_path):
                continue

            # 查找图像文件
            image_path = self._find_image(image_dir, img_id)
            if image_path is None:
                print(f"Warning: Image not found for id: {img_id}")
                continue

            # 解析XML
            samples = self._parse_xml(anno_path, image_path)
            self.samples.extend(samples)

    def _find_image(self, image_dir: str, img_id: str) -> Optional[str]:
        """查找图像文件路径."""
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
            img_path = os.path.join(image_dir, f"{img_id}{ext}")
            if os.path.exists(img_path):
                return img_path
        return None

    def _parse_xml(self, anno_path: str, image_path: str) -> List[Tuple[str, List[float], int]]:
        """解析单个XML文件，返回样本列表."""
        samples = []

        try:
            tree = ET.parse(anno_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                # 获取类别名称
                name_elem = obj.find('name')
                if name_elem is None:
                    continue
                class_name = name_elem.text.strip().lower()

                # 判断标签
                label = self._class_to_label(class_name)
                if label is None:
                    # 未知类别，跳过
                    continue

                # 获取bbox
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue

                try:
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                except (AttributeError, ValueError, TypeError):
                    continue

                # 验证bbox有效性
                if xmax <= xmin or ymax <= ymin:
                    continue

                samples.append((image_path, [xmin, ymin, xmax, ymax], label))

        except ET.ParseError as e:
            print(f"Warning: Failed to parse {anno_path}: {e}")

        return samples

    def _class_to_label(self, class_name: str) -> Optional[int]:
        """将类别名称转换为标签.
        Returns:
            0: 非跌倒
            1: 跌倒
            None: 未知类别（跳过）
        """
        if class_name in self.fall_classes:
            return 1
        if self.normal_classes is None:
            # 未指定正常类别，其他所有类别都视为正常
            return 0
        if class_name in self.normal_classes:
            return 0
        # 未知类别
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, bbox, label = self.samples[idx]

        # 加载图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # 随机crop ROI
        roi = self.cropper(img, bbox)

        # 数据增强
        if self.transform:
            roi = self.transform(roi)

        # Resize
        if self.use_letterbox and self.letterbox:
            roi = self.letterbox(roi)
        else:
            roi = cv2.resize(roi, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # To tensor: (H, W, C) -> (C, H, W), normalize to [0, 1]
        roi = torch.from_numpy(roi).permute(2, 0, 1).float() / 255.0

        return roi, torch.tensor(label, dtype=torch.long)


class TrainingAugmentation:
    """训练时的数据增强."""

    def __init__(self, aug_cfg: Dict[str, Any]):
        self.cfg = aug_cfg
        self.color_jitter_cfg = aug_cfg.get('color_jitter', {})
        self.random_gray_cfg = aug_cfg.get('random_gray', {})
        self.random_rotation_cfg = aug_cfg.get('random_rotation', {})
        self.random_mask_cfg = aug_cfg.get('random_mask', {})
        self.horizontal_flip_cfg = aug_cfg.get('horizontal_flip', {})

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


class WarmupScheduler:
    """支持 warmup 的学习率调度器包装器.

    warmup 基于总 batch step 计算，每个 step 更新学习率。
    支持两种 warmup 策略:
    - linear: 从 warmup_init_lr 线性增加到初始学习率
    - constant: 保持 warmup_init_lr 不变，warmup 结束后跳到初始学习率
    """

    def __init__(self, optimizer, scheduler, warmup_steps: int = 0,
                 warmup_strategy: str = 'linear', warmup_init_lr: float = 1e-5):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.warmup_strategy = warmup_strategy
        self.warmup_init_lr = warmup_init_lr
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.warmup_finished = False

    def step(self, metrics=None):
        """Step the scheduler (每个 epoch 结束时调用)."""
        # Warmup 在 step_batch 中处理，这里只处理主 scheduler
        if self.warmup_finished and self.scheduler is not None:
            if metrics is not None and hasattr(self.scheduler, 'step'):
                self.scheduler.step(metrics)
            elif hasattr(self.scheduler, 'step'):
                self.scheduler.step()

    def step_batch(self):
        """每个 batch/step 调用，用于 warmup."""
        if self.current_step < self.warmup_steps:
            # Warmup 阶段
            if self.warmup_strategy == 'linear':
                # 线性增加学习率: lr = init_lr + (base_lr - init_lr) * (step / total_steps)
                alpha = self.current_step / self.warmup_steps
                for i, group in enumerate(self.optimizer.param_groups):
                    group['lr'] = self.warmup_init_lr + alpha * (self.base_lrs[i] - self.warmup_init_lr)
            elif self.warmup_strategy == 'constant':
                # 保持 warmup_init_lr 不变
                for group in self.optimizer.param_groups:
                    group['lr'] = self.warmup_init_lr
            self.current_step += 1
        elif not self.warmup_finished:
            # Warmup 刚结束，设置为基础学习率
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i]
            self.warmup_finished = True
            self.current_step += 1
        else:
            self.current_step += 1

    def get_last_lr(self):
        """获取当前学习率."""
        return [group['lr'] for group in self.optimizer.param_groups]

    @property
    def is_warmup(self):
        """是否还在 warmup 阶段."""
        return self.current_step < self.warmup_steps


def train_epoch(model, loader, optimizer, criterion, device, epoch: int = 0, rank: int = 0, log_interval: int = 50, scheduler=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    num_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 更新 warmup scheduler (每个 step)
        if scheduler is not None and isinstance(scheduler, WarmupScheduler):
            scheduler.step_batch()

        _, predicted = torch.max(outputs, 1)
        batch_correct = (predicted == labels).sum().item()
        batch_size = labels.size(0)

        total_correct += batch_correct
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        # 打印训练进度 (使用当前实际学习率)
        if rank == 0 and log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            batch_acc = batch_correct / batch_size
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch[{epoch}] Batch[{batch_idx + 1}/{num_batches}]  "
                  f"loss={loss.item():.4f} acc={batch_acc:.4f}  "
                  f"avg_loss={avg_loss:.4f} avg_acc={avg_acc:.4f}  lr={current_lr:.6f}")

    return total_loss, total_correct, total_samples


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm.tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return total_loss, total_correct, total_samples


def main():
    parser = argparse.ArgumentParser(description="Train simple image classifier from COCO or Pascal VOC annotations")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--override", type=str, default=None, help="Override config values, e.g., 'lr=0.01,batch_size=128'")
    args = parser.parse_args()

    # 加载配置文件
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 处理命令行覆盖
    if args.override:
        for override in args.override.split(','):
            key, value = override.split('=')
            keys = key.split('.')
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            # 尝试转换为适当类型 (使用ast.literal_eval更安全)
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass  # 保持字符串形式
            d[keys[-1]] = value

    # 提取配置
    data_cfg = cfg.get('data', {})
    model_cfg = cfg.get('model', {})
    input_cfg = cfg.get('input', {})
    crop_cfg = cfg.get('crop', {})
    aug_cfg = cfg.get('data_augmentation', {})
    lr_cfg = cfg.get('lr_scheduler', {})
    early_cfg = cfg.get('early_stopping', {})
    output_cfg = cfg.get('output', {})
    ddp_cfg = cfg.get('ddp', {})
    log_cfg = cfg.get('log', {})

    # DDP初始化
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        backend = ddp_cfg.get('backend', 'nccl')
        dist.init_process_group(backend=backend if torch.cuda.is_available() else 'gloo')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
        local_rank = 0

    # 设置随机种子 (在DDP初始化之后设置，确保所有进程使用相同种子)
    seed = cfg.get('seed', None)
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 为了完全可复现，但可能会降低性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if rank == 0:
            print(f"Random seed set to {seed}")

    if rank == 0:
        output_dir = output_cfg.get('dir', 'outputs/simple_classifier')
        os.makedirs(output_dir, exist_ok=True)
        # 保存使用的配置
        with open(os.path.join(output_dir, 'config_used.yaml'), 'w') as f:
            yaml.dump(cfg, f)

        # 判断数据格式类型 (coco 或 voc)
        data_format = data_cfg.get('format', 'coco').lower()
        if data_format == 'voc':
            print(f"Training simple classifier from Pascal VOC annotations")
        else:
            print(f"Training simple classifier from COCO annotations")
        print(f"  Config: {args.config}")
        print(f"  DDP: {ddp}, World size: {world_size}")
        print(f"  Use letterbox: {input_cfg.get('use_letterbox', True)}")
        print(f"  Data augmentation: {aug_cfg.get('enabled', True)}")

    if ddp:
        dist.barrier()

    # 判断数据格式类型
    data_format = data_cfg.get('format', 'coco').lower()
    image_dir = data_cfg.get('image_dir')

    if data_format == 'voc':
        # Pascal VOC 格式 - 支持多目录
        voc_cfg = cfg.get('voc', {})

        # 支持单个目录或多个目录列表
        train_dirs = voc_cfg.get('train_dirs', voc_cfg.get('data_dir'))
        if train_dirs is None:
            raise ValueError("voc.train_dirs or voc.data_dir must be specified in config for VOC format")
        if isinstance(train_dirs, str):
            train_dirs = [train_dirs]

        val_dirs = voc_cfg.get('val_dirs')
        if isinstance(val_dirs, str):
            val_dirs = [val_dirs]

        # 类别映射配置
        fall_classes = voc_cfg.get('fall_classes', ['fall'])
        normal_classes = voc_cfg.get('normal_classes')  # None表示其他所有类别都是正常

        if rank == 0:
            print(f"Loading training data from VOC directories: {train_dirs}")
            print(f"  Fall classes: {fall_classes}")
            if normal_classes:
                print(f"  Normal classes: {normal_classes}")
            else:
                print(f"  Normal classes: <all others>")

        train_dataset = VOCFallDataset(
            data_dirs=train_dirs,
            split='train',
            transform=None,
            target_size=input_cfg.get('target_size', 96),
            use_letterbox=input_cfg.get('use_letterbox', True),
            fill_value=input_cfg.get('fill_value', 114),
            fall_classes=fall_classes,
            normal_classes=normal_classes,
            shrink_max=crop_cfg.get('shrink_max', 3),
            expand_max=crop_cfg.get('expand_max', 25),
        )

        # 加载验证集（如果提供了）
        val_dataset = None
        if val_dirs:
            if rank == 0:
                print(f"Loading validation data from VOC directories: {val_dirs}")
            val_dataset = VOCFallDataset(
                data_dirs=val_dirs,
                split='val',
                transform=None,
                target_size=input_cfg.get('target_size', 96),
                use_letterbox=input_cfg.get('use_letterbox', True),
                fill_value=input_cfg.get('fill_value', 114),
                fall_classes=fall_classes,
                normal_classes=normal_classes,
                shrink_max=crop_cfg.get('shrink_max', 3),
                expand_max=crop_cfg.get('expand_max', 25),
            )
        else:
            if rank == 0:
                print("No validation set provided, training without validation")
    else:
        # COCO 格式 (默认)
        train_coco_json = data_cfg.get('train_coco_json')
        val_coco_json = data_cfg.get('val_coco_json')

        if not train_coco_json or not image_dir:
            raise ValueError("train_coco_json and image_dir must be specified in config for COCO format")

        if rank == 0:
            print(f"Loading training data from: {train_coco_json}")

        train_dataset = CocoFallDataset(
            image_dir=image_dir,
            coco_json=train_coco_json,
            transform=None,
            target_size=input_cfg.get('target_size', 96),
            use_letterbox=input_cfg.get('use_letterbox', True),
            fill_value=input_cfg.get('fill_value', 114),
            person_category_id=cfg.get('coco', {}).get('person_category_id', 1),
            fall_category_id=cfg.get('coco', {}).get('fall_category_id', 1),
            shrink_max=crop_cfg.get('shrink_max', 3),
            expand_max=crop_cfg.get('expand_max', 25),
        )

        # 加载验证集（如果提供了）
        val_dataset = None
        if val_coco_json and os.path.exists(val_coco_json):
            if rank == 0:
                print(f"Loading validation data from: {val_coco_json}")
            val_dataset = CocoFallDataset(
                image_dir=image_dir,
                coco_json=val_coco_json,
                transform=None,
                target_size=input_cfg.get('target_size', 96),
                use_letterbox=input_cfg.get('use_letterbox', True),
                fill_value=input_cfg.get('fill_value', 114),
                person_category_id=cfg.get('coco', {}).get('person_category_id', 1),
                fall_category_id=cfg.get('coco', {}).get('fall_category_id', 1),
                shrink_max=crop_cfg.get('shrink_max', 3),
                expand_max=crop_cfg.get('expand_max', 25),
            )
        else:
            if rank == 0:
                print("No validation set provided, training without validation")

    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        labels = [s[2] for s in train_dataset.samples]
        n_fall = sum(labels)
        n_normal = len(labels) - n_fall
        print(f"  Fall: {n_fall}, Normal: {n_normal}")

        if val_dataset:
            print(f"Val samples: {len(val_dataset)}")
            labels = [s[2] for s in val_dataset.samples]
            n_fall = sum(labels)
            n_normal = len(labels) - n_fall
            print(f"  Fall: {n_fall}, Normal: {n_normal}")

    # 为训练集和验证集设置不同的数据增强
    train_dataset.transform = None if not aug_cfg.get('enabled', True) else TrainingAugmentation(aug_cfg)
    if val_dataset:
        val_dataset.transform = None  # 验证时不做增强

    # Sampler和DataLoader
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get('batch_size', 64),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if ddp else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.get('batch_size', 64),
            sampler=val_sampler,
            shuffle=False,
            num_workers=cfg.get('num_workers', 4),
            pin_memory=True,
        )

    # 模型
    dropout = model_cfg.get('dropout', 0.3)
    fall_class_idx = model_cfg.get('fall_class_idx', 1)
    model = SimpleFallClassifier(dropout=dropout, fall_class_idx=fall_class_idx).to(device)
    if ddp:
        find_unused = ddp_cfg.get('find_unused_parameters', False)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=find_unused)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    lr = cfg.get('lr', 0.001)
    weight_decay = cfg.get('weight_decay', 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 学习率调度器
    scheduler_type = lr_cfg.get('type', 'plateau')
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max',
            factor=lr_cfg.get('factor', 0.5),
            patience=lr_cfg.get('patience', 10),
            min_lr=lr_cfg.get('min_lr', 1e-5)
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=lr_cfg.get('T_max', cfg.get('epochs', 100)),
            eta_min=lr_cfg.get('min_lr', 1e-5)
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_cfg.get('step_size', 30),
            gamma=lr_cfg.get('gamma', 0.1)
        )
    else:
        scheduler = None

    # 包装 warmup scheduler
    warmup_cfg = lr_cfg.get('warmup', {})
    if warmup_cfg.get('enabled', False):
        # 计算 warmup steps: warmup_epochs * len(train_loader)
        warmup_epochs = warmup_cfg.get('epochs', 5)
        warmup_steps = warmup_epochs * len(train_loader)
        scheduler = WarmupScheduler(
            optimizer,
            scheduler,
            warmup_steps=warmup_steps,
            warmup_strategy=warmup_cfg.get('strategy', 'linear'),  # 'linear' 或 'constant'
            warmup_init_lr=warmup_cfg.get('init_lr', 1e-5)
        )
        if rank == 0:
            print(f"Warmup enabled: {warmup_epochs} epochs = {warmup_steps} steps, "
                  f"strategy={warmup_cfg.get('strategy', 'linear')}, "
                  f"init_lr={warmup_cfg.get('init_lr', 1e-5)}")

    # 训练循环
    epochs = cfg.get('epochs', 100)
    best_acc = 0.0
    patience_counter = 0
    log_interval = log_cfg.get('interval', 1)

    # 导入时间模块用于计算剩余时间
    import time
    start_time = time.time()
    epoch_times = []

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]["lr"]

        # 训练
        t_loss, t_correct, t_samples = train_epoch(model, train_loader, optimizer, criterion, device, epoch=epoch, rank=rank, log_interval=log_cfg.get("batch_log_interval", 50), scheduler=scheduler)

        # 验证（如果有验证集）
        if val_loader:
            v_loss, v_correct, v_samples = eval_epoch(model, val_loader, criterion, device)
        else:
            v_loss, v_correct, v_samples = 0.0, 0, 0

        # 聚合指标
        if val_loader:
            metrics = torch.tensor([t_loss, t_correct, t_samples, v_loss, v_correct, v_samples],
                                   device=device, dtype=torch.float64)
        else:
            metrics = torch.tensor([t_loss, t_correct, t_samples, 0.0, 0, 0],
                                   device=device, dtype=torch.float64)

        if ddp:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        t_loss_avg = metrics[0].item() / max(1, metrics[2].item())
        t_acc = metrics[1].item() / max(1, metrics[2].item())

        if val_loader:
            v_loss_avg = metrics[3].item() / max(1, metrics[5].item())
            v_acc = metrics[4].item() / max(1, metrics[5].item())
        else:
            v_loss_avg = 0.0
            v_acc = t_acc

        # 计算本epoch耗时和剩余时间估计
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        # 使用移动平均估计剩余时间
        avg_epoch_time = sum(epoch_times[-10:]) / min(len(epoch_times), 10)  # 最近10个epoch的平均
        remaining_epochs = epochs - epoch
        remaining_time = avg_epoch_time * remaining_epochs
        remaining_mins = int(remaining_time / 60)
        remaining_secs = int(remaining_time % 60)

        if rank == 0:
            # 更新学习率
            if scheduler is not None:
                if isinstance(scheduler, WarmupScheduler):
                    # WarmupScheduler 处理 plateau 和普通 scheduler
                    if val_loader and scheduler.scheduler is not None and hasattr(scheduler.scheduler, 'mode'):
                        scheduler.step(v_acc)
                    else:
                        scheduler.step()
                elif scheduler_type == 'plateau' and val_loader:
                    scheduler.step(v_acc)
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            if epoch % log_interval == 0 or epoch == epochs:
                # 格式化剩余时间
                if remaining_mins >= 60:
                    remaining_hours = remaining_mins // 60
                    remaining_mins_remainder = remaining_mins % 60
                    remaining_str = f"{remaining_hours}h{remaining_mins_remainder}m"
                else:
                    remaining_str = f"{remaining_mins}m{remaining_secs}s"

                if val_loader:
                    print(f"Epoch {epoch}/{epochs}  "
                          f"train_loss={t_loss_avg:.4f} train_acc={t_acc:.4f}  "
                          f"val_loss={v_loss_avg:.4f} val_acc={v_acc:.4f}  "
                          f"lr={current_lr:.6f}  remain={remaining_str}")
                else:
                    print(f"Epoch {epoch}/{epochs}  "
                          f"train_loss={t_loss_avg:.4f} train_acc={t_acc:.4f}  "
                          f"lr={current_lr:.6f}  remain={remaining_str}")

            # 保存模型
            save_best = output_cfg.get('save_best', True)
            if val_loader and save_best and v_acc > best_acc:
                best_acc = v_acc
                patience_counter = 0
                save_path = os.path.join(output_cfg.get('dir', 'outputs/simple_classifier'), "best.pt")
                torch.save(model.module.state_dict() if ddp else model.state_dict(), save_path)
                if epoch % log_interval == 0:
                    print(f"  -> Saved best model (val_acc={v_acc:.4f})")
            elif not val_loader and epoch % output_cfg.get('save_every', 10) == 0:
                save_path = os.path.join(output_cfg.get('dir', 'outputs/simple_classifier'), f"epoch_{epoch}.pt")
                torch.save(model.module.state_dict() if ddp else model.state_dict(), save_path)
                print(f"  -> Saved checkpoint (epoch={epoch})")

            # 早停检查
            if early_cfg.get('enabled', True) and val_loader:
                if v_acc <= best_acc + early_cfg.get('min_delta', 0.001):
                    patience_counter += 1
                    if patience_counter >= early_cfg.get('patience', 20):
                        print(f"Early stopping at epoch {epoch}")
                        break
                else:
                    patience_counter = 0

    if rank == 0:
        if val_loader:
            print(f"Training done. Best val_acc={best_acc:.4f}")
        else:
            print(f"Training done. Final train_acc={t_acc:.4f}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
