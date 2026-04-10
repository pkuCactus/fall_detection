"""检查数据集 bbox 大小范围的统计工具.

支持 COCO 和 Pascal VOC 格式.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        description="检查数据集 bbox 大小范围统计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # COCO 格式
  python check_bbox_stats.py --format coco --json data/annotations/train.json --image-dir data/images

  # VOC 格式 (单目录)
  python check_bbox_stats.py --format voc --data-dir data/VOC2007 --split train

  # VOC 格式 (多目录)
  python check_bbox_stats.py --format voc --data-dir data/VOC2007,data/VOC2012 --split train
        """,
    )
    parser.add_argument("--format", choices=["coco", "voc"], required=True, help="数据集格式: coco 或 voc")

    # COCO 格式参数
    parser.add_argument("--json", type=str, help="COCO 标注 JSON 文件路径")
    parser.add_argument("--image-dir", type=str, help="COCO 图像根目录")

    # VOC 格式参数
    parser.add_argument("--data-dir", type=str, help="VOC 数据目录，多个目录用逗号分隔")
    parser.add_argument("--split", type=str, default="train", help="VOC 数据划分: train/val/test (默认: train)")

    # 通用参数
    parser.add_argument("--min-size", type=int, default=0, help="过滤小于此尺寸的 bbox (默认: 0)")
    parser.add_argument("--max-size", type=int, default=0, help="过滤大于此尺寸的 bbox (默认: 0, 表示不限制)")
    parser.add_argument("--output", type=str, help="输出统计结果到文件 (可选)")

    return parser.parse_args()


def load_coco_bboxes(json_path: str) -> Tuple[List[Dict], Dict[int, Dict], Dict[int, str]]:
    """加载 COCO 格式的 bbox.

    Returns:
        bboxes: list of dict with keys: img_id, bbox [x1,y1,x2,y2], w, h, area, label
        images: dict mapping img_id to image info
        categories: dict mapping category_id to category_name
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    bboxes = []

    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        bbox = ann["bbox"]  # [x, y, w, h]
        x, y, w, h = bbox
        x2, y2 = x + w, y + h

        bboxes.append(
            {
                "img_id": img_id,
                "bbox": [x, y, x2, y2],  # [x1, y1, x2, y2]
                "w": w,
                "h": h,
                "area": w * h,
                "category_id": ann.get("category_id", 0),
                "category_name": categories.get(ann.get("category_id", 0), "unknown"),
                "image_file": images.get(img_id, {}).get("file_name", "unknown"),
            }
        )

    return bboxes, images, categories


def load_voc_bboxes(data_dirs: List[str], split: str) -> List[Dict]:
    """加载 Pascal VOC 格式的 bbox.

    Returns:
        bboxes: list of dict with keys: img_path, bbox [x1,y1,x2,y2], w, h, area, label
    """
    bboxes = []

    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        image_dir = data_dir / "JPEGImages"
        anno_dir = data_dir / "Annotations"
        split_file = data_dir / "ImageSets" / "Main" / f"{split}.txt"

        if not split_file.exists():
            print(f"Warning: Split file not found: {split_file}")
            continue

        # 读取图像列表
        with open(split_file, "r") as f:
            image_ids = [line.strip() for line in f if line.strip()]

        for img_id in image_ids:
            anno_path = anno_dir / f"{img_id}.xml"
            if not anno_path.exists():
                continue

            # 查找图像文件
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                candidate = image_dir / f"{img_id}{ext}"
                if candidate.exists():
                    img_path = str(candidate)
                    break

            if img_path is None:
                continue

            # 解析 XML
            try:
                tree = ET.parse(anno_path)
                root = tree.getroot()

                for obj in root.findall("object"):
                    name_elem = obj.find("name")
                    if name_elem is None:
                        continue

                    class_name = name_elem.text.strip()
                    bndbox = obj.find("bndbox")
                    if bndbox is None:
                        continue

                    try:
                        xmin = float(bndbox.find("xmin").text)
                        ymin = float(bndbox.find("ymin").text)
                        xmax = float(bndbox.find("xmax").text)
                        ymax = float(bndbox.find("ymax").text)
                    except (AttributeError, ValueError, TypeError):
                        continue

                    w = xmax - xmin
                    h = ymax - ymin

                    if w <= 0 or h <= 0:
                        continue

                    bboxes.append(
                        {
                            "img_id": img_id,
                            "img_path": img_path,
                            "bbox": [xmin, ymin, xmax, ymax],
                            "w": w,
                            "h": h,
                            "area": w * h,
                            "class_name": class_name,
                        }
                    )
            except ET.ParseError as e:
                print(f"Warning: Failed to parse {anno_path}: {e}")

    return bboxes


def compute_stats(values: np.ndarray) -> Dict[str, Any]:
    """计算数值统计信息."""
    if len(values) == 0:
        return {"count": 0}

    return {
        "count": len(values),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def format_stats(name: str, stats: Dict[str, Any], unit: str = "px") -> str:
    """格式化统计信息为字符串."""
    if stats["count"] == 0:
        return f"{name}: No data\n"

    lines = [
        f"\n{'=' * 50}",
        f"{name}",
        f"{'=' * 50}",
        f"  Count:      {stats['count']:,}",
        f"  Min:        {stats['min']:.1f} {unit}",
        f"  Max:        {stats['max']:.1f} {unit}",
        f"  Mean:       {stats['mean']:.1f} {unit}",
        f"  Median:     {stats['median']:.1f} {unit}",
        f"  Std:        {stats['std']:.1f} {unit}",
        f"  Percentiles:",
        f"    5%:       {stats['p5']:.1f} {unit}",
        f"    25%:      {stats['p25']:.1f} {unit}",
        f"    75%:      {stats['p75']:.1f} {unit}",
        f"    95%:      {stats['p95']:.1f} {unit}",
        f"    99%:      {stats['p99']:.1f} {unit}",
    ]
    return "\n".join(lines)


def analyze_aspect_ratios(widths: np.ndarray, heights: np.ndarray) -> str:
    """分析宽高比分布."""
    ratios = widths / (heights + 1e-6)

    # 分类统计
    square = np.sum((ratios >= 0.9) & (ratios <= 1.1))
    landscape = np.sum(ratios > 1.1)
    portrait = np.sum(ratios < 0.9)

    total = len(ratios)

    lines = [
        f"\n{'=' * 50}",
        "Aspect Ratio Distribution",
        f"{'=' * 50}",
        f"  Square (0.9-1.1):     {square:,} ({square / total * 100:.1f}%)",
        f"  Landscape (>1.1):     {landscape:,} ({landscape / total * 100:.1f}%)",
        f"  Portrait (<0.9):      {portrait:,} ({portrait / total * 100:.1f}%)",
        "",
        "  Aspect Ratio Stats:",
        f"    Min:   {np.min(ratios):.2f}",
        f"    Max:   {np.max(ratios):.2f}",
        f"    Mean:  {np.mean(ratios):.2f}",
        f"    Median: {np.median(ratios):.2f}",
    ]
    return "\n".join(lines)


def analyze_per_class_stats(bboxes: List[Dict], format_type: str) -> str:
    """分析每个类别的 bbox 统计信息."""
    from collections import defaultdict

    # 按类别分组
    class_bboxes = defaultdict(list)
    for bbox in bboxes:
        if format_type == "coco":
            # COCO格式使用category_name，如果没有则用category_id
            key = bbox.get("category_name", str(bbox.get("category_id", "unknown")))
        else:
            key = bbox.get("class_name", "unknown")
        class_bboxes[key].append(bbox)

    lines = [
        f"\n{'=' * 50}",
        "Per-Class Statistics",
        f"{'=' * 50}",
    ]

    # 按样本数排序
    sorted_classes = sorted(class_bboxes.items(), key=lambda x: len(x[1]), reverse=True)

    total = len(bboxes)

    for class_key, class_bbox_list in sorted_classes:
        widths = np.array([b["w"] for b in class_bbox_list])
        heights = np.array([b["h"] for b in class_bbox_list])
        areas = widths * heights

        count = len(class_bbox_list)
        pct = count / total * 100

        lines.append(f"\n  Class: {class_key}")
        lines.append(f"    Count:   {count:,} ({pct:.1f}%)")
        lines.append(f"    Width:   min={np.min(widths):.0f}, max={np.max(widths):.0f}, median={np.median(widths):.0f}")
        lines.append(
            f"    Height:  min={np.min(heights):.0f}, max={np.max(heights):.0f}, median={np.median(heights):.0f}"
        )
        lines.append(f"    Area:    min={np.min(areas):.0f}, max={np.max(areas):.0f}, median={np.median(areas):.0f}")

        # 宽高比统计
        ratios = widths / (heights + 1e-6)
        lines.append(f"    Aspect:  min={np.min(ratios):.2f}, max={np.max(ratios):.2f}, median={np.median(ratios):.2f}")

    # 添加汇总
    lines.append(f"\n  {'=' * 46}")
    lines.append(f"  Total: {total:,} bboxes across {len(sorted_classes)} classes")

    return "\n".join(lines)


def analyze_size_distribution(widths: np.ndarray, heights: np.ndarray) -> str:
    """分析尺寸分布（按面积分段）."""
    areas = widths * heights

    # 定义面积区间 (像素)
    bins = [
        (0, 32 * 32, "Tiny (0-1K)"),
        (32 * 32, 64 * 64, "Small (1K-4K)"),
        (64 * 64, 128 * 128, "Medium (4K-16K)"),
        (128 * 128, 256 * 256, "Large (16K-64K)"),
        (256 * 256, 512 * 512, "XLarge (64K-256K)"),
        (512 * 512, float("inf"), "Huge (>256K)"),
    ]

    lines = [
        f"\n{'=' * 50}",
        "Size Distribution (by Area)",
        f"{'=' * 50}",
    ]

    for min_area, max_area, label in bins:
        count = np.sum((areas >= min_area) & (areas < max_area))
        pct = count / len(areas) * 100
        lines.append(f"  {label:20s}: {count:6,} ({pct:5.1f}%)")

    return "\n".join(lines)


def find_extreme_bboxes(bboxes: List[Dict], widths: np.ndarray, heights: np.ndarray, n: int = 5) -> str:
    """找出极端尺寸的 bbox."""
    areas = widths * heights

    # 最小的 n 个
    smallest_idx = np.argsort(areas)[:n]
    # 最大的 n 个
    largest_idx = np.argsort(areas)[-n:][::-1]

    lines = [
        f"\n{'=' * 50}",
        f"Extreme BBoxes (Top {n} smallest/largest)",
        f"{'=' * 50}",
        "\n  Smallest:",
    ]

    for i, idx in enumerate(smallest_idx, 1):
        bbox = bboxes[idx]
        lines.append(
            f"    {i}. {bbox['w']:.0f}x{bbox['h']:.0f} (area={widths[idx] * heights[idx]:.0f}) - {bbox.get('image_file', bbox.get('img_path', 'unknown'))}"
        )

    lines.append("\n  Largest:")
    for i, idx in enumerate(largest_idx, 1):
        bbox = bboxes[idx]
        lines.append(
            f"    {i}. {bbox['w']:.0f}x{bbox['h']:.0f} (area={widths[idx] * heights[idx]:.0f}) - {bbox.get('image_file', bbox.get('img_path', 'unknown'))}"
        )

    return "\n".join(lines)


def main():
    """主函数."""
    args = parse_args()

    print("Loading dataset...")

    # 加载数据
    if args.format == "coco":
        if not args.json or not args.image_dir:
            print("Error: --json and --image-dir are required for COCO format")
            sys.exit(1)

        if not os.path.exists(args.json):
            print(f"Error: JSON file not found: {args.json}")
            sys.exit(1)

        bboxes, images, categories = load_coco_bboxes(args.json)
        print(f"Loaded {len(bboxes)} bboxes from COCO: {args.json}")
        print(f"Categories: {list(categories.values())}")

    else:  # VOC format
        if not args.data_dir:
            print("Error: --data-dir is required for VOC format")
            sys.exit(1)

        data_dirs = [d.strip() for d in args.data_dir.split(",")]
        bboxes = load_voc_bboxes(data_dirs, args.split)
        print(f"Loaded {len(bboxes)} bboxes from VOC: {data_dirs} (split={args.split})")

    if len(bboxes) == 0:
        print("No bounding boxes found!")
        sys.exit(1)

    # 提取尺寸
    widths = np.array([b["w"] for b in bboxes])
    heights = np.array([b["h"] for b in bboxes])
    areas = widths * heights

    # 过滤
    if args.min_size > 0:
        mask = (widths >= args.min_size) & (heights >= args.min_size)
        widths = widths[mask]
        heights = heights[mask]
        areas = areas[mask]
        bboxes = [b for i, b in enumerate(bboxes) if mask[i]]
        print(f"After filtering (min_size={args.min_size}): {len(bboxes)} bboxes")

    if args.max_size > 0:
        mask = (widths <= args.max_size) & (heights <= args.max_size)
        widths = widths[mask]
        heights = heights[mask]
        areas = areas[mask]
        bboxes = [b for i, b in enumerate(bboxes) if mask[i]]
        print(f"After filtering (max_size={args.max_size}): {len(bboxes)} bboxes")

    # 计算统计信息
    width_stats = compute_stats(widths)
    height_stats = compute_stats(heights)
    area_stats = compute_stats(areas)

    # 生成报告
    report = []
    report.append(format_stats("Width Statistics", width_stats, "px"))
    report.append(format_stats("Height Statistics", height_stats, "px"))
    report.append(format_stats("Area Statistics", area_stats, "px²"))
    report.append(analyze_aspect_ratios(widths, heights))
    report.append(analyze_size_distribution(widths, heights))
    report.append(analyze_per_class_stats(bboxes, args.format))
    report.append(find_extreme_bboxes(bboxes, widths, heights, n=5))

    report_text = "\n".join(report)

    # 输出
    print(report_text)

    # 保存到文件
    if args.output:
        with open(args.output, "w") as f:
            f.write(report_text)
        print(f"\nReport saved to: {args.output}")

    # 打印建议
    print(f"\n{'=' * 50}")
    print("Recommendations")
    print(f"{'=' * 50}")

    median_w = width_stats["median"]
    median_h = height_stats["median"]
    print(f"  - Median bbox size: {median_w:.0f} x {median_h:.0f}")
    print(f"  - Suggested input size for classifier: {int(max(median_w, median_h) * 1.5)} (1.5x max median)")

    # 检查是否有异常小的 bbox
    tiny_count = np.sum(areas < 32 * 32)
    if tiny_count > 0:
        print(f"  - Warning: {tiny_count} tiny bboxes (< 32x32, {tiny_count / len(areas) * 100:.1f}%)")

    # 检查是否有异常大的 bbox
    huge_count = np.sum(areas > 512 * 512)
    if huge_count > 0:
        print(f"  - Warning: {huge_count} huge bboxes (> 512x512, {huge_count / len(areas) * 100:.1f}%)")

    print(f"\n  95% of bboxes are within: {width_stats['p95']:.0f} x {height_stats['p95']:.0f}")


if __name__ == "__main__":
    main()
