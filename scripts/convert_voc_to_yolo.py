#!/usr/bin/env python3
"""Convert PASCAL VOC annotations to YOLO format."""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm


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
    """Map PASCAL VOC class names to YOLO class IDs."""
    # For person detection: map all person-related classes to 'person' (0)
    person_classes = {
        'person', 'stand', 'sit', 'squat', 'bend',
        'fall', 'fall_down', 'falldown', 'fallen',
        'kneel', 'half_up', 'crawl'
    }

    return person_classes


def convert_dataset(data_dir, output_dir, split='train'):
    """Convert VOC dataset to YOLO format."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Create output directories
    images_output = output_dir / 'images' / split
    labels_output = output_dir / 'labels' / split
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    # Find all XML files
    ann_dir = data_dir / 'Annotations'
    xml_files = list(ann_dir.glob('*.xml'))

    person_classes = get_class_mapping()
    converted = 0
    skipped = 0

    for xml_file in tqdm(xml_files, desc=f'Converting {split}'):
        try:
            boxes = parse_voc_xml(xml_file)

            # Filter only person-related boxes
            person_boxes = [b for b in boxes if b['class'] in person_classes]

            if not person_boxes:
                skipped += 1
                continue

            # Create label file
            label_file = labels_output / f"{xml_file.stem}.txt"
            with open(label_file, 'w') as f:
                for box in person_boxes:
                    # Class 0 is 'person'
                    f.write(f"0 {box['x_center']:.6f} {box['y_center']:.6f} "
                           f"{box['width']:.6f} {box['height']:.6f}\n")

            # Copy/symlink image
            img_file = data_dir / 'JPEGImages' / f"{xml_file.stem}.jpg"
            if img_file.exists():
                img_output = images_output / img_file.name
                if not img_output.exists():
                    if img_output.is_symlink():
                        img_output.unlink()
                    os.symlink(os.path.abspath(img_file), img_output)

            converted += 1

        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            skipped += 1

    print(f"Converted: {converted}, Skipped: {skipped}")
    return converted


def create_data_yaml(output_dir, data_dir):
    """Create YOLO data.yaml configuration file."""
    yaml_content = f"""# YOLOv8 人体检测数据集配置
# 自动生成于 VOC 格式转换

path: {os.path.abspath(output_dir)}  # 数据集根目录

train: images/train
val: images/val
test: images/test

# 类别定义
names:
  0: person

nc: 1
"""

    yaml_path = Path(data_dir) / 'fall_detection.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created: {yaml_path}")


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


def main():
    parser = argparse.ArgumentParser(description='Convert PASCAL VOC to YOLO format')
    parser.add_argument('--data-dir', default='data/mini',
                       help='Input VOC dataset directory')
    parser.add_argument('--output-dir', default='data/mini_yolo',
                       help='Output YOLO dataset directory')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio')
    args = parser.parse_args()

    print(f"Converting {args.data_dir} to YOLO format...")

    # Split dataset
    splits = split_dataset(args.data_dir, args.train_ratio, args.val_ratio)

    # Convert each split
    for split_name, xml_files in splits.items():
        if not xml_files:
            continue
        print(f"\nProcessing {split_name}: {len(xml_files)} files")

        # Create temporary directory with only this split's files
        split_dir = Path(args.output_dir) / f'_{split_name}'
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create symlinks for this split
        (split_dir / 'Annotations').mkdir(exist_ok=True)
        (split_dir / 'JPEGImages').mkdir(exist_ok=True)

        for xml_file in xml_files:
            src_ann = os.path.abspath(xml_file)
            dst_ann = split_dir / 'Annotations' / xml_file.name
            if not dst_ann.exists():
                os.symlink(src_ann, dst_ann)

            img_name = xml_file.stem + '.jpg'
            src_img = os.path.abspath(Path(args.data_dir) / 'JPEGImages' / img_name)
            dst_img = split_dir / 'JPEGImages' / img_name
            if not dst_img.exists():
                os.symlink(src_img, dst_img)

        # Convert this split
        convert_dataset(split_dir, args.output_dir, split_name)

        # Cleanup
        import shutil
        shutil.rmtree(split_dir)

    # Create data.yaml
    create_data_yaml(args.output_dir, 'data')

    print("\nConversion complete!")
    print(f"YOLO dataset: {args.output_dir}")
    print(f"Data config: data/fall_detection.yaml")


if __name__ == '__main__':
    main()
