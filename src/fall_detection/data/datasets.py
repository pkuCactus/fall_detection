"""Dataset classes for fall detection training."""

import hashlib
import json
import os
import pickle
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import RandomCropWithPadding, FixedExpandCrop, LetterBoxResize


class LRUImageCache:
    """LRU 图像缓存队列，限制内存使用."""

    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: 最大缓存图像数量，默认1000张
        """
        self.max_size = max_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: int) -> Optional[np.ndarray]:
        """获取缓存的图像，LRU更新."""
        if key in self._cache:
            # 移动到末尾（最近使用）
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: int, image: np.ndarray) -> None:
        """添加图像到缓存."""
        if key in self._cache:
            # 已存在，更新并移到末尾
            self._cache.move_to_end(key)
            self._cache[key] = image
        else:
            # 新图像
            if len(self._cache) >= self.max_size:
                # 淘汰最旧的
                self._cache.popitem(last=False)
            self._cache[key] = image

    def clear(self) -> None:
        """清空缓存."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """计算缓存命中率."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._cache)


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
        cache_size: int = 1000,
        inference_mode: bool = False,
        inference_expand_px: int = 10,
    ):
        self.image_dir = image_dir
        self.target_size = target_size
        self.use_letterbox = use_letterbox
        self.person_category_id = person_category_id
        self.inference_mode = inference_mode

        # 推理模式下禁用所有数据增强，使用固定外扩crop
        if inference_mode:
            self.transform = None
            self.cropper = FixedExpandCrop(expand_px=inference_expand_px)
        else:
            self.transform = transform
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

        # LRU图像缓存
        self.image_cache = LRUImageCache(max_size=cache_size) if cache_size > 0 else None

    def __len__(self):
        return len(self.samples)

    def _get_image(self, img_id: int) -> np.ndarray:
        """加载图像，带LRU缓存."""
        # 先尝试从LRU缓存获取
        if self.image_cache is not None:
            cached_img = self.image_cache.get(img_id)
            if cached_img is not None:
                return cached_img

        # 缓存未命中，加载图像
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

        # 存入LRU缓存
        if self.image_cache is not None:
            self.image_cache.put(img_id, img)

        return img

    def __getitem__(self, idx):
        img_id, bbox, label = self.samples[idx]

        # 加载完整图像
        img = self._get_image(img_id)

        # crop ROI (训练时随机，推理时固定)
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
        cache_dir: Optional[str] = None,
        cache_size: int = 1000,
        inference_mode: bool = False,
        inference_expand_px: int = 10,
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
            cache_dir: 缓存目录，如果指定则尝试从缓存加载samples
            cache_size: LRU图像缓存大小，默认1000张图像，设为0禁用缓存
            inference_mode: 推理模式，禁用所有随机增强，使用固定外扩crop
            inference_expand_px: 推理时固定外扩像素数，默认10px
        """
        self.data_dirs = data_dirs if isinstance(data_dirs, list) else [data_dirs]
        self.split = split
        self.target_size = target_size
        self.use_letterbox = use_letterbox
        self.inference_mode = inference_mode

        # 类别映射配置
        self.fall_classes = set(c.lower() for c in (fall_classes or ["fall"]))
        self.normal_classes = set(c.lower() for c in normal_classes) if normal_classes else None

        # 推理模式下禁用所有数据增强，使用固定外扩crop
        if inference_mode:
            self.transform = None
            self.cropper = FixedExpandCrop(expand_px=inference_expand_px)
        else:
            self.transform = transform
            self.cropper = RandomCropWithPadding(shrink_max=shrink_max, expand_max=expand_max)

        self.letterbox = LetterBoxResize(target_size=target_size, fill_value=fill_value) if use_letterbox else None

        # 构建样本列表: (image_path, bbox, label)
        self.samples: List[Tuple[str, List[float], int]] = []

        # LRU图像缓存 - 使用图像路径作为key
        self._image_cache = LRUImageCache(max_size=cache_size) if cache_size > 0 else None
        self._image_path_to_key: Dict[str, int] = {}
        self._next_image_key = 0

        # 尝试从缓存加载
        cache_loaded = False
        if cache_dir:
            cache_loaded = self._try_load_from_cache(cache_dir, shrink_max, expand_max)

        if not cache_loaded:
            # 为每个数据目录加载样本
            self.total_images = 0
            for data_dir in self.data_dirs:
                self._load_from_dir(data_dir)
            print(f"Loaded {len(self.samples)} samples from {self.total_images} images across {len(self.data_dirs)} directories.")

            # 保存缓存
            if cache_dir:
                self._save_cache(cache_dir, shrink_max, expand_max)

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
            with open(image_set_file, 'r', encoding='utf-8') as f:
                image_ids = [line.rstrip('\n\r') for line in f if line.rstrip('\n\r')]
        else:
            print(f"Warning: Image set file not found: {image_set_file}, skipping this directory")
            return

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
            if not samples:
                print(f"Warning: No valid objects found in annotation: {anno_path}")
                continue
            self.samples.extend(samples)
            self.total_images += 1

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

    def _get_cache_key(self, shrink_max: int, expand_max: int) -> str:
        """生成缓存键，基于数据目录、split和配置."""
        key_data = {
            'data_dirs': sorted(self.data_dirs),
            'split': self.split,
            'fall_classes': sorted(self.fall_classes),
            'normal_classes': sorted(self.normal_classes) if self.normal_classes else None,
            'shrink_max': shrink_max,
            'expand_max': expand_max,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _try_load_from_cache(self, cache_dir: str, shrink_max: int, expand_max: int) -> bool:
        """尝试从缓存加载samples.
        Returns:
            True if loaded from cache, False otherwise
        """
        try:
            os.makedirs(cache_dir, exist_ok=True)
            cache_key = self._get_cache_key(shrink_max, expand_max)
            cache_file = os.path.join(cache_dir, f'voc_{self.split}_{cache_key}.pkl')

            if os.path.exists(cache_file):
                print(f"Loading samples from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                self.samples = cache_data['samples']
                self.total_images = cache_data.get('total_images', 0)
                print(f"Loaded {len(self.samples)} samples from {self.total_images} images (from cache)")
                return True
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
        return False

    def _save_cache(self, cache_dir: str, shrink_max: int, expand_max: int) -> None:
        """保存samples到缓存."""
        try:
            os.makedirs(cache_dir, exist_ok=True)
            cache_key = self._get_cache_key(shrink_max, expand_max)
            cache_file = os.path.join(cache_dir, f'voc_{self.split}_{cache_key}.pkl')

            cache_data = {
                'samples': self.samples,
                'total_images': self.total_images,
                'data_dirs': self.data_dirs,
                'split': self.split,
                'fall_classes': self.fall_classes,
                'normal_classes': self.normal_classes,
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved samples cache to: {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def __len__(self):
        return len(self.samples)

    def _get_image(self, image_path: str) -> np.ndarray:
        """加载图像，带LRU缓存."""
        # 获取或分配图像key
        if image_path not in self._image_path_to_key:
            self._image_path_to_key[image_path] = self._next_image_key
            self._next_image_key += 1
        image_key = self._image_path_to_key[image_path]

        # 尝试从LRU缓存获取
        if self._image_cache is not None:
            cached_img = self._image_cache.get(image_key)
            if cached_img is not None:
                return cached_img

        # 缓存未命中，加载图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # 存入LRU缓存
        if self._image_cache is not None:
            self._image_cache.put(image_key, img)

        return img

    @property
    def cache_hit_rate(self) -> float:
        """获取图像缓存命中率."""
        return self._image_cache.hit_rate if self._image_cache else 0.0

    def __getitem__(self, idx):
        image_path, bbox, label = self.samples[idx]

        # 加载图像（带LRU缓存）
        img = self._get_image(image_path)

        # crop ROI (训练时随机，推理时固定)
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


