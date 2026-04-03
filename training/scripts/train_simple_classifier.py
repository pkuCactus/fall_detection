import argparse
import ast
import json
import os
import sys
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return total_loss, total_correct, total_samples


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return total_loss, total_correct, total_samples


def main():
    parser = argparse.ArgumentParser(description="Train simple image classifier from COCO annotations")
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

    if rank == 0:
        output_dir = output_cfg.get('dir', 'outputs/simple_classifier')
        os.makedirs(output_dir, exist_ok=True)
        # 保存使用的配置
        with open(os.path.join(output_dir, 'config_used.yaml'), 'w') as f:
            yaml.dump(cfg, f)

        print(f"Training simple classifier from COCO annotations")
        print(f"  Config: {args.config}")
        print(f"  DDP: {ddp}, World size: {world_size}")
        print(f"  Use letterbox: {input_cfg.get('use_letterbox', True)}")
        print(f"  Data augmentation: {aug_cfg.get('enabled', True)}")

    if ddp:
        dist.barrier()

    # 加载训练集
    train_coco_json = data_cfg.get('train_coco_json')
    val_coco_json = data_cfg.get('val_coco_json')
    image_dir = data_cfg.get('image_dir')

    if not train_coco_json or not image_dir:
        raise ValueError("train_coco_json and image_dir must be specified in config")

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

    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        labels = [s[2] for s in train_dataset.samples]
        n_fall = sum(labels)
        n_normal = len(labels) - n_fall
        print(f"  Fall: {n_fall}, Normal: {n_normal}")

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
        if rank == 0:
            print(f"Val samples: {len(val_dataset)}")
            labels = [s[2] for s in val_dataset.samples]
            n_fall = sum(labels)
            n_normal = len(labels) - n_fall
            print(f"  Fall: {n_fall}, Normal: {n_normal}")
    else:
        if rank == 0:
            print("No validation set provided, training without validation")

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

    # 训练循环
    epochs = cfg.get('epochs', 100)
    best_acc = 0.0
    patience_counter = 0
    log_interval = log_cfg.get('interval', 1)

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 训练
        t_loss, t_correct, t_samples = train_epoch(model, train_loader, optimizer, criterion, device)

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

        if rank == 0:
            # 更新学习率
            if scheduler_type == 'plateau' and val_loader:
                scheduler.step(v_acc)
            elif scheduler and scheduler_type != 'plateau':
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            if epoch % log_interval == 0 or epoch == epochs:
                if val_loader:
                    print(f"Epoch {epoch}/{epochs}  "
                          f"train_loss={t_loss_avg:.4f} train_acc={t_acc:.4f}  "
                          f"val_loss={v_loss_avg:.4f} val_acc={v_acc:.4f}  lr={current_lr:.6f}")
                else:
                    print(f"Epoch {epoch}/{epochs}  "
                          f"train_loss={t_loss_avg:.4f} train_acc={t_acc:.4f}  lr={current_lr:.6f}")

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
