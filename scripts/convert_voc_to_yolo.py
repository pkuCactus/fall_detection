#!/usr/bin/env python3
"""Convert PASCAL VOC annotations to YOLO format in-place."""

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


def convert_dataset(data_dir, split_files, split_name):
    """Convert VOC dataset to YOLO format."""
    data_dir = Path(data_dir)

    # Create labels directory
    labels_dir = data_dir / 'labels' / split_name
    labels_dir.mkdir(parents=True, exist_ok=True)

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
                skipped += 1
                continue

            # Create label file
            label_file = labels_dir / f"{xml_file.stem}.txt"
            with open(label_file, 'w') as f:
                for box in valid_boxes:
                    class_id = box['class_id']
                    f.write(f"{class_id} {box['x_center']:.6f} {box['y_center']:.6f} "
                           f"{box['width']:.6f} {box['height']:.6f}\n")

            converted += 1

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            skipped += 1

    return converted, skipped


def split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1):
    """Split dataset into train/val/test sets."""
    import random

    data_dir = Path(data_dir)
    ann_dir = data_dir / 'Annotations'
    xml_files = sorted(list(ann_dir.glob('*.xml')))

    # Shuffle
    random.seed(42)
    random.shuffle(xml_files)

    n_total = len(xml_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        'train': xml_files[:n_train],
        'val': xml_files[n_train:n_train + n_val],
        'test': xml_files[n_train + n_val:]
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

train: images
val: images
test: images

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
    parser = argparse.ArgumentParser(description='Convert PASCAL VOC to YOLO format in-place')
    parser.add_argument('--data-dir', default='data/mini',
                       help='Input VOC dataset directory')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio')
    args = parser.parse_args()

    print(f"Converting {args.data_dir} to YOLO format...")

    # Create images symlink if not exists
    data_dir = Path(args.data_dir)
    images_link = data_dir / 'images'
    if not images_link.exists():
        jpeg_dir = data_dir / 'JPEGImages'
        if jpeg_dir.exists():
            os.symlink('JPEGImages', images_link)
            print(f"Created symlink: images -> JPEGImages")

    # Clean old labels
    labels_dir = data_dir / 'labels'
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
        print(f"Cleaned old labels directory")

    # Split dataset
    splits = split_dataset(args.data_dir, args.train_ratio, args.val_ratio)

    # Convert each split
    total_converted = 0
    for split_name, xml_files in splits.items():
        if not xml_files:
            continue
        print(f"\nProcessing {split_name}: {len(xml_files)} files")
        converted, skipped = convert_dataset(args.data_dir, xml_files, split_name)
        total_converted += converted
        print(f"  Converted: {converted}, Skipped: {skipped}")

    # Create data.yaml
    create_data_yaml(args.data_dir)

    print(f"\nConversion complete!")
    print(f"Total converted: {total_converted}")
    print(f"YOLO labels: {args.data_dir}/labels/")
    print(f"Images: {args.data_dir}/images/ -> JPEGImages/")
    print(f"Data config: data/fall_detection.yaml")


if __name__ == '__main__':
    main()
