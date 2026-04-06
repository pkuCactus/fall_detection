#!/usr/bin/env python3
"""Convert PASCAL VOC annotations to YOLO format with images as symlinks."""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import shutil


def parse_voc_xml(xml_path):
    """Parse PASCAL VOC XML file and extract bounding boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    boxes = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text.lower()
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        boxes.append({
            'class': class_name,
            'x_center': x_center,
            'y_center': y_center,
            'width': w,
            'height': h
        })

    return boxes


def get_class_mapping():
    """Map PASCAL VOC class names to YOLO class IDs.

    Returns:
        dict: {class_name: class_id}
    """
    # 8-class mapping for pose/state detection
    class_map = {
        # Standing
        'stand': 0,

        # Sitting
        'sit': 1,

        # Squatting
        'squat': 2,

        # Bending
        'bend': 3,

        # Half-up (crouching, between stand and fall)
        'half_up': 4,

        # Kneeling
        'kneel': 5,

        # Crawling
        'crawl': 6,

        # Fallen
        'fall_down': 7,
        'fall': 7,
        'falldown': 7,
        'fallen': 7,

        # Default person class (map to stand)
        'person': 0,
    }
    return class_map


def convert_dataset(data_dir, split_files, split_name, jpeg_dir):
    """Convert VOC dataset to YOLO format."""
    data_dir = Path(data_dir)

    # Create labels directory
    labels_dir = data_dir / 'labels' / split_name
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create images directory with symlinks
    images_split_dir = data_dir / 'images' / split_name
    images_split_dir.mkdir(parents=True, exist_ok=True)

    class_map = get_class_mapping()
    converted = 0
    skipped = 0

    for xml_file in tqdm(split_files, desc=f'Converting {split_name}'):
        try:
            boxes = parse_voc_xml(xml_file)

            # Map boxes to YOLO class IDs
            valid_boxes = []
            for box in boxes:
                class_name = box['class']
                if class_name in class_map:
                    box['class_id'] = class_map[class_name]
                    valid_boxes.append(box)

            if not valid_boxes:
                print(f"  Skipped {xml_file.name}: no valid classes found")
                skipped += 1
                continue

            # Create label file
            label_file = labels_dir / f"{xml_file.stem}.txt"
            with open(label_file, 'w') as f:
                for box in valid_boxes:
                    class_id = box['class_id']
                    f.write(f"{class_id} {box['x_center']:.6f} {box['y_center']:.6f} "
                           f"{box['width']:.6f} {box['height']:.6f}\n")

            # Create symlink for image
            img_file = jpeg_dir / f"{xml_file.stem}.jpg"
            if not img_file.exists():
                # Try other extensions
                for ext in ['.png', '.jpeg', '.bmp']:
                    img_file = jpeg_dir / f"{xml_file.stem}{ext}"
                    if img_file.exists():
                        break

            if img_file.exists():
                link_path = images_split_dir / img_file.name
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                os.symlink(os.path.abspath(img_file), link_path)

            converted += 1

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            skipped += 1

    return converted, skipped


def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train/val/test sets.

    Args:
        data_dir: Dataset directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio (can be 0)
        test_ratio: Test set ratio (can be 0)
    """
    import random

    data_dir = Path(data_dir)
    ann_dir = data_dir / 'Annotations'
    xml_files = sorted(list(ann_dir.glob('*.xml')))

    if not xml_files:
        print(f"Warning: No XML files found in {ann_dir}")
        return {'train': [], 'val': [], 'test': []}

    # Shuffle
    random.seed(42)
    random.shuffle(xml_files)

    n_total = len(xml_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    # Ensure at least 1 sample for non-zero ratios
    if train_ratio > 0 and n_train == 0:
        n_train = min(1, n_total)
    if val_ratio > 0 and n_val == 0 and n_total > n_train:
        n_val = min(1, n_total - n_train)
    if test_ratio > 0 and n_test == 0 and n_total > n_train + n_val:
        n_test = min(1, n_total - n_train - n_val)

    # Recalculate to ensure we don't exceed total
    splits = {
        'train': xml_files[:n_train],
        'val': xml_files[n_train:n_train + n_val] if val_ratio > 0 else [],
        'test': xml_files[n_train + n_val:n_train + n_val + n_test] if test_ratio > 0 else []
    }

    return splits


def create_data_yaml(data_dir):
    """Create YOLO data.yaml configuration file."""
    class_names = {
        0: 'stand',
        1: 'sit',
        2: 'squat',
        3: 'bend',
        4: 'half_up',
        5: 'kneel',
        6: 'crawl',
        7: 'fall_down'
    }

    names_str = '\n'.join([f"  {k}: {v}" for k, v in class_names.items()])

    yaml_content = f"""# YOLOv8 人体姿态检测数据集配置
# 自动生成于 VOC 格式转换
# 支持8类别姿态检测

path: {os.path.abspath(data_dir)}  # 数据集根目录

train: images/train
val: images/val
test: images/test

# 类别定义 (8种姿态类别)
names:
{names_str}

nc: 8
"""

    yaml_path = Path('data') / 'fall_detection.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created: {yaml_path}")
    print(f"Classes: {class_names}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC to YOLO format with symlinks for images'
    )
    parser.add_argument('--data-dir', default='data/mini',
                       help='Input VOC dataset directory (contains Annotations/, JPEGImages/)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1, can be 0)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1, can be 0)')
    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    print(f"Converting {args.data_dir} to YOLO format...")
    print(f"Split ratios: train={args.train_ratio:.2f}, val={args.val_ratio:.2f}, test={args.test_ratio:.2f}")

    data_dir = Path(args.data_dir)
    jpeg_dir = data_dir / 'JPEGImages'

    if not jpeg_dir.exists():
        print(f"Error: JPEGImages directory not found: {jpeg_dir}")
        return

    # Clean old directories
    images_dir = data_dir / 'images'
    if images_dir.exists():
        shutil.rmtree(images_dir)
        print(f"Cleaned old images directory")

    labels_dir = data_dir / 'labels'
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
        print(f"Cleaned old labels directory")

    # Split dataset
    splits = split_dataset(args.data_dir, args.train_ratio, args.val_ratio, args.test_ratio)

    # Convert each split
    total_converted = 0
    for split_name, xml_files in splits.items():
        if not xml_files:
            print(f"\nSkipping {split_name}: 0 files")
            continue
        print(f"\nProcessing {split_name}: {len(xml_files)} files")
        converted, skipped = convert_dataset(args.data_dir, xml_files, split_name, jpeg_dir)
        total_converted += converted
        print(f"  Converted: {converted}, Skipped: {skipped}")

    # Create data.yaml
    create_data_yaml(args.data_dir)

    print(f"\nConversion complete!")
    print(f"Total converted: {total_converted}")
    print(f"YOLO labels: {args.data_dir}/labels/{{train,val,test}}/")
    print(f"Images symlinks: {args.data_dir}/images/{{train,val,test}}/")
    print(f"Data config: data/fall_detection.yaml")


if __name__ == '__main__':
    main()
