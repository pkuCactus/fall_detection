#!/usr/bin/env python3
"""Convert PASCAL VOC annotations to YOLO format with multiple dataset support.

Features:
1. Support multiple VOC datasets conversion with train/val/test directory splits
2. Use ImageSets/Main/{train,val,test}.txt for splits when use_imagesets=true
3. Support configurable class mapping: VOC class -> YOLO class name -> ID
4. Designed for YOLO-World training compatibility

Configuration format (configs/tools/voc_to_yolo_example.yaml):
    datasets:
      train_dirs:
        - "data/dataset1"
        - "data/dataset2"
      val_dirs:
        - "data/dataset1"
        - "data/dataset3"
      test_dirs:
        - "data/dataset4"

    class_mapping:
      stand: "person"
      sitting: "person"
      fall_down: "person"

    names:
      - "person"  # class 0

    output:
      dir: "data/yolo"
      images_dir: "images"
      labels_dir: "labels"
      yaml_path: "data/fall_detection.yaml"

    use_imagesets: true
    copy_images: true
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import yaml
from tqdm import tqdm
import shutil
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def get_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    """Get image dimensions using PIL.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (width, height) or None if failed
    """
    if not PIL_AVAILABLE:
        return None
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None


def fix_xml_size(xml_path: Path, correct_width: int, correct_height: int) -> bool:
    """Fix XML file with incorrect or zero size values.

    Args:
        xml_path: Path to XML annotation file
        correct_width: Correct width from actual image
        correct_height: Correct height from actual image

    Returns:
        True if XML was modified, False otherwise
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size_elem = root.find('size')
        if size_elem is None:
            size_elem = ET.SubElement(root, 'size')
            width_elem = ET.SubElement(size_elem, 'width')
            height_elem = ET.SubElement(size_elem, 'height')
            depth_elem = ET.SubElement(size_elem, 'depth')
            depth_elem.text = '3'
        else:
            width_elem = size_elem.find('width')
            height_elem = size_elem.find('height')
            if width_elem is None:
                width_elem = ET.SubElement(size_elem, 'width')
            if height_elem is None:
                height_elem = ET.SubElement(size_elem, 'height')

        orig_width = int(width_elem.text) if width_elem.text else 0
        orig_height = int(height_elem.text) if height_elem.text else 0

        if orig_width != correct_width or orig_height != correct_height:
            width_elem.text = str(correct_width)
            height_elem.text = str(correct_height)

            # Write back with proper formatting
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            return True

        return False
    except Exception as e:
        print(f"    Warning: Failed to fix XML {xml_path}: {e}")
        return False


def parse_voc_xml(xml_path: Path, image_path: Optional[Path] = None) -> Tuple[List[Dict], int, int, bool]:
    """Parse PASCAL VOC XML file and extract bounding boxes.

    Args:
        xml_path: Path to XML annotation file
        image_path: Optional path to corresponding image for size validation

    Returns:
        Tuple of (boxes, width, height, was_fixed)
        boxes: list of dict with keys: class, x_center, y_center, width, height
        was_fixed: True if XML was modified to fix size issues
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image size from XML
    size = root.find('size')
    if size is not None:
        width_elem = size.find('width')
        height_elem = size.find('height')
        width = int(width_elem.text) if width_elem is not None and width_elem.text else 0
        height = int(height_elem.text) if height_elem is not None and height_elem.text else 0
    else:
        width, height = 0, 0

    was_fixed = False

    # Check if size is invalid (0 or missing)
    if width <= 0 or height <= 0:
        if image_path and image_path.exists():
            actual_size = get_image_size(image_path)
            if actual_size:
                width, height = actual_size
                # Try to fix the XML file
                was_fixed = fix_xml_size(xml_path, width, height)
                if was_fixed:
                    print(f"    Fixed XML size: {xml_path.name} -> {width}x{height}")
            else:
                raise ValueError(f"Cannot read image size from {image_path}")
        else:
            raise ValueError(f"Invalid size {width}x{height} in {xml_path} and no image available")

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

        # Clamp values to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        boxes.append({
            'class': class_name,
            'x_center': x_center,
            'y_center': y_center,
            'width': w,
            'height': h
        })

    return boxes, width, height, was_fixed


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_class_mapping(config: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Get class name mapping and class name to ID mapping from config.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (voc_to_yolo_name, yolo_name_to_id)
        - voc_to_yolo_name: {voc_class_name: yolo_class_name}
        - yolo_name_to_id: {yolo_class_name: class_id}
    """
    class_mapping = config.get('class_mapping', {})

    # Convert to lowercase for case-insensitive matching
    voc_to_yolo_name = {k.lower(): v for k, v in class_mapping.items()}

    # Build YOLO name to ID mapping from names list
    names = config.get('names', [])
    if not names:
        # If no names list, extract unique yolo class names from class_mapping
        unique_names = sorted(set(voc_to_yolo_name.values()))
        names = unique_names

    yolo_name_to_id = {name: idx for idx, name in enumerate(names)}

    return voc_to_yolo_name, yolo_name_to_id


def get_output_config(config: Dict[str, Any]) -> Dict[str, Path]:
    """Get output configuration from config.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with output paths
    """
    output_cfg = config.get('output', {})

    output_dir = Path(output_cfg.get('dir', 'data/yolo'))
    images_dir = output_cfg.get('images_dir', 'images')
    labels_dir = output_cfg.get('labels_dir', 'labels')
    yaml_path = Path(output_cfg.get('yaml_path', output_dir / 'data.yaml'))

    return {
        'output_dir': output_dir,
        'images_dir': images_dir,
        'labels_dir': labels_dir,
        'yaml_path': yaml_path,
    }


def read_imageset_split(data_dir: Path, split_name: str) -> Optional[Set[str]]:
    """Read image set from ImageSets/Main/{split}.txt file.

    Args:
        data_dir: Dataset directory
        split_name: Split name (train, val, test, trainval)

    Returns:
        Set of image IDs (stems) or None if file doesn't exist.
        Each non-empty line is treated as a complete image ID (supports names with spaces).
    """
    split_file = data_dir / 'ImageSets' / 'Main' / f'{split_name}.txt'
    if not split_file.exists():
        return None

    image_ids = set()
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')  # Only strip newline characters
            if line:
                # Treat entire line as image ID (supports names with spaces)
                image_ids.add(line)
    return image_ids


def get_image_path(jpeg_dir: Path, image_id: str) -> Optional[Path]:
    """Find image file with various extensions.

    Args:
        jpeg_dir: Directory containing images
        image_id: Image ID (stem)

    Returns:
        Path to image file or None if not found
    """
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        img_path = jpeg_dir / f'{image_id}{ext}'
        if img_path.exists():
            return img_path
    return None


def get_xml_files_from_dir(data_dir: Path) -> Dict[str, Path]:
    """Get all XML annotation files from a directory.

    Args:
        data_dir: VOC dataset directory (contains Annotations/)

    Returns:
        Dictionary mapping image_id to XML file path
    """
    ann_dir = data_dir / 'Annotations'
    if not ann_dir.exists():
        return {}
    return {f.stem: f for f in ann_dir.glob('*.xml')}


def convert_dataset_split(
    data_dirs: List[Path],
    output_dir: Path,
    split_name: str,
    voc_to_yolo_name: Dict[str, str],
    yolo_name_to_id: Dict[str, int],
    images_subdir: str = 'images',
    labels_subdir: str = 'labels',
    use_imagesets: bool = True,
    copy_images: bool = True,
) -> Tuple[int, int]:
    """Convert a dataset split (train/val/test) from multiple source directories.

    Args:
        data_dirs: List of source VOC dataset directories
        output_dir: Output directory for YOLO format
        split_name: Split name (train, val, test)
        voc_to_yolo_name: Mapping from VOC class names to YOLO class names
        yolo_name_to_id: Mapping from YOLO class names to class IDs
        images_subdir: Subdirectory name for images
        labels_subdir: Subdirectory name for labels
        use_imagesets: Whether to use ImageSets/Main/ for splits
        copy_images: Whether to copy images instead of symlinking

    Returns:
        Tuple of (converted_count, skipped_count)
    """
    # Create output directories
    labels_split_dir = output_dir / labels_subdir / split_name
    labels_split_dir.mkdir(parents=True, exist_ok=True)

    images_split_dir = output_dir / images_subdir / split_name
    images_split_dir.mkdir(parents=True, exist_ok=True)

    # Directory for images without labels (for manual annotation later)
    no_labels_dir = output_dir / 'no_labels' / split_name
    no_labels_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    no_label_count = 0
    processed_ids = set()  # Track processed image IDs to avoid duplicates

    for data_dir in data_dirs:
        print(f"  Processing {data_dir}...")

        ann_dir = data_dir / 'Annotations'
        jpeg_dir = data_dir / 'JPEGImages'

        if not ann_dir.exists():
            print(f"    Warning: Annotations directory not found: {ann_dir}")
            continue
        if not jpeg_dir.exists():
            print(f"    Warning: JPEGImages directory not found: {jpeg_dir}")
            continue

        # Get XML files for this split
        if use_imagesets:
            image_ids = read_imageset_split(data_dir, split_name)
            if image_ids is None:
                print(f"    Warning: ImageSets/{split_name}.txt not found, using all XMLs")
                image_ids = set(get_xml_files_from_dir(data_dir).keys())
        else:
            # Use all XMLs in the directory
            image_ids = set(get_xml_files_from_dir(data_dir).keys())

        # Filter out already processed IDs
        image_ids = image_ids - processed_ids
        processed_ids.update(image_ids)

        if not image_ids:
            print(f"    No images to process in this directory")
            continue

        all_xml_files = get_xml_files_from_dir(data_dir)

        for image_id in tqdm(image_ids, desc=f"  Converting {split_name}", leave=False):
            img_file = get_image_path(jpeg_dir, image_id)

            if image_id not in all_xml_files:
                # No XML annotation - copy image to no_labels for manual annotation
                if img_file:
                    dest_path = no_labels_dir / img_file.name
                    if dest_path.exists() or dest_path.is_symlink():
                        dest_path.unlink()
                    if copy_images:
                        shutil.copy2(img_file, dest_path)
                    else:
                        os.symlink(os.path.abspath(img_file), dest_path)
                    no_label_count += 1
                else:
                    print(f"    Warning: {image_id}.xml and image not found, skipping")
                    skipped += 1
                continue

            xml_file = all_xml_files[image_id]

            try:
                boxes, _, _, was_fixed = parse_voc_xml(xml_file, img_file)

                # Map boxes to YOLO class IDs
                valid_boxes = []
                for box in boxes:
                    voc_class = box['class']
                    if voc_class in voc_to_yolo_name:
                        yolo_class = voc_to_yolo_name[voc_class]
                        if yolo_class in yolo_name_to_id:
                            box['class_id'] = yolo_name_to_id[yolo_class]
                            valid_boxes.append(box)
                        else:
                            print(f"    Warning: YOLO class '{yolo_class}' not in names list")
                    else:
                        print(f"    Warning: VOC class '{voc_class}' not in class_mapping")

                if not valid_boxes:
                    # Check if this is an empty scene (no objects at all) or filtered scene
                    if len(boxes) == 0:
                        # Empty scene - will create empty label file as negative sample
                        pass
                    else:
                        # Has objects but none in class_mapping - warning already printed above
                        print(f"    Note: {xml_file.name} has {len(boxes)} objects but none in class_mapping")

                # Create label file (may be empty for negative samples)
                label_file = labels_split_dir / f"{image_id}.txt"
                with open(label_file, 'w') as f:
                    for box in valid_boxes:
                        class_id = box['class_id']
                        f.write(f"{class_id} {box['x_center']:.6f} {box['y_center']:.6f} "
                               f"{box['width']:.6f} {box['height']:.6f}\n")

                # Handle image file
                img_file = get_image_path(jpeg_dir, image_id)
                if img_file:
                    link_path = images_split_dir / img_file.name

                    # Remove existing file/link
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()

                    if copy_images:
                        shutil.copy2(img_file, link_path)
                    else:
                        os.symlink(os.path.abspath(img_file), link_path)

                converted += 1

            except Exception as e:
                print(f"    Error processing {xml_file}: {e}")
                skipped += 1

    if no_label_count > 0:
        print(f"  {split_name}: Converted {converted}, Skipped {skipped}, No labels: {no_label_count} (copied to no_labels/)")
    else:
        print(f"  {split_name}: Converted {converted}, Skipped {skipped}")
    return converted, skipped, no_label_count


def create_yolo_yaml(
    config: Dict[str, Any],
    output_cfg: Dict[str, Path],
) -> Path:
    """Create YOLO dataset YAML configuration file.

    Args:
        config: Configuration dictionary
        output_cfg: Output configuration dictionary

    Returns:
        Path to created YAML file
    """
    names = config.get('names', [])
    if not names:
        # Extract from class_mapping values
        class_mapping = config.get('class_mapping', {})
        names = sorted(set(class_mapping.values()))

    names_str = '\n'.join([f"  {k}: {v}" for k, v in enumerate(names)])
    nc = len(names)

    output_dir = output_cfg['output_dir']
    images_dir = output_cfg['images_dir']
    yaml_path = output_cfg['yaml_path']

    yaml_content = f"""# YOLO dataset configuration
# Auto-generated from VOC conversion

path: {os.path.abspath(output_dir)}  # Dataset root directory

train: {images_dir}/train
val: {images_dir}/val
test: {images_dir}/test

# Class definitions
names:
{names_str}

nc: {nc}
"""

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nCreated: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC to YOLO format for YOLO-World training'
    )
    parser.add_argument('--config', '-c', type=Path, required=True,
                       help='YAML config file (see configs/tools/voc_to_yolo_example.yaml)')
    parser.add_argument('--output-dir', '-o', type=Path,
                       help='Override output directory from config')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print configuration without converting')

    args = parser.parse_args()

    # Load configuration
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Get class mappings
    voc_to_yolo_name, yolo_name_to_id = get_class_mapping(config)

    # Get output configuration
    output_cfg = get_output_config(config)
    if args.output_dir:
        output_cfg['output_dir'] = args.output_dir
        # Update yaml_path if it was using default
        if output_cfg['yaml_path'] == output_cfg['output_dir'] / 'data.yaml':
            output_cfg['yaml_path'] = args.output_dir / 'data.yaml'

    # Get dataset directories
    datasets_cfg = config.get('datasets', {})
    train_dirs = [Path(d) for d in datasets_cfg.get('train_dirs', [])]
    val_dirs = [Path(d) for d in datasets_cfg.get('val_dirs', [])]
    test_dirs = [Path(d) for d in datasets_cfg.get('test_dirs', [])]

    # Get processing options
    use_imagesets = config.get('use_imagesets', True)
    copy_images = config.get('copy_images', True)

    # Print configuration
    print("=" * 60)
    print("VOC to YOLO Conversion Configuration")
    print("=" * 60)
    print(f"\nClass Mapping (VOC -> YOLO -> ID):")
    for voc_name, yolo_name in sorted(voc_to_yolo_name.items()):
        class_id = yolo_name_to_id.get(yolo_name, '?')
        print(f"  {voc_name} -> {yolo_name} ({class_id})")

    print(f"\nDataset Directories:")
    print(f"  Train: {[str(d) for d in train_dirs]}")
    print(f"  Val:   {[str(d) for d in val_dirs]}")
    print(f"  Test:  {[str(d) for d in test_dirs] if test_dirs else 'None'}")

    print(f"\nOutput Configuration:")
    print(f"  Output Dir: {output_cfg['output_dir']}")
    print(f"  Images Dir: {output_cfg['images_dir']}")
    print(f"  Labels Dir: {output_cfg['labels_dir']}")
    print(f"  YAML Path:  {output_cfg['yaml_path']}")

    print(f"\nProcessing Options:")
    print(f"  Use ImageSets: {use_imagesets}")
    print(f"  Copy Images:   {copy_images}")

    # Warn about symlink issues with YOLO
    if not copy_images:
        print("\n" + "!" * 60)
        print("WARNING: Using symbolic links for images (copy_images: false)")
        print("!" * 60)
        print("YOLO may fail to find annotation files because:")
        print("  1. YOLO resolves symbolic links to their real paths")
        print("  2. It then looks for labels in the original image directory")
        print("  3. Instead of: output_dir/labels/")
        print("     It looks in: original_dataset/labels/")
        print("\nRecommendation: Set copy_images: true unless disk space is limited.")
        print("!" * 60)

    if args.dry_run:
        print("\nDry run mode - exiting without conversion")
        return

    # Create output directory
    output_cfg['output_dir'].mkdir(parents=True, exist_ok=True)

    # Convert each split
    results = {}

    if train_dirs:
        print(f"\n{'='*60}")
        print("Processing Train Split")
        print(f"{'='*60}")
        results['train'] = convert_dataset_split(
            data_dirs=train_dirs,
            output_dir=output_cfg['output_dir'],
            split_name='train',
            voc_to_yolo_name=voc_to_yolo_name,
            yolo_name_to_id=yolo_name_to_id,
            images_subdir=output_cfg['images_dir'],
            labels_subdir=output_cfg['labels_dir'],
            use_imagesets=use_imagesets,
            copy_images=copy_images,
        )

    if val_dirs:
        print(f"\n{'='*60}")
        print("Processing Val Split")
        print(f"{'='*60}")
        results['val'] = convert_dataset_split(
            data_dirs=val_dirs,
            output_dir=output_cfg['output_dir'],
            split_name='val',
            voc_to_yolo_name=voc_to_yolo_name,
            yolo_name_to_id=yolo_name_to_id,
            images_subdir=output_cfg['images_dir'],
            labels_subdir=output_cfg['labels_dir'],
            use_imagesets=use_imagesets,
            copy_images=copy_images,
        )

    if test_dirs:
        print(f"\n{'='*60}")
        print("Processing Test Split")
        print(f"{'='*60}")
        results['test'] = convert_dataset_split(
            data_dirs=test_dirs,
            output_dir=output_cfg['output_dir'],
            split_name='test',
            voc_to_yolo_name=voc_to_yolo_name,
            yolo_name_to_id=yolo_name_to_id,
            images_subdir=output_cfg['images_dir'],
            labels_subdir=output_cfg['labels_dir'],
            use_imagesets=use_imagesets,
            copy_images=copy_images,
        )

    # Create YOLO YAML
    yaml_path = create_yolo_yaml(config, output_cfg)

    # Print summary
    print(f"\n{'='*60}")
    print("Conversion Summary")
    print(f"{'='*60}")
    total_conv = 0
    total_skip = 0
    total_no_label = 0
    for split_name, (conv, skip, no_label) in results.items():
        if no_label > 0:
            print(f"  {split_name}: Converted {conv}, Skipped {skip}, No labels: {no_label}")
        else:
            print(f"  {split_name}: Converted {conv}, Skipped {skip}")
        total_conv += conv
        total_skip += skip
        total_no_label += no_label
    if total_no_label > 0:
        print(f"  Total: Converted {total_conv}, Skipped {total_skip}, No labels: {total_no_label}")
        print(f"\n  Note: {total_no_label} images without labels were copied to no_labels/{{train,val,test}}/")
        print(f"        for manual annotation.")
    else:
        print(f"  Total: Converted {total_conv}, Skipped {total_skip}")
    print(f"\nOutput directory: {output_cfg['output_dir']}")
    print(f"  Labels: {output_cfg['output_dir'] / output_cfg['labels_dir']}")
    print(f"  Images: {output_cfg['output_dir'] / output_cfg['images_dir']}")
    print(f"  Config: {yaml_path}")


if __name__ == '__main__':
    main()
