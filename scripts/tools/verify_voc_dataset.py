#!/usr/bin/env python3
"""校验 VOC 数据集完整性，处理图像和标注不匹配的情况.

功能:
1. 检查 ImageSets/Main/{split}.txt 中列出的所有 item
2. 只存在 label 不存在图像 -> 删除 label
3. 只存在图像不存在 label -> 移动图像到 no_label 目录
4. 同时更新 ImageSets 文件，移除无效 item

用法:
    python verify_voc_dataset.py data/VOC --move-no-label
    python verify_voc_dataset.py data/VOC --dry-run
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        description="校验 VOC 数据集完整性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 校验并修复数据集（移动无标注图像到 no_label 目录）
  python verify_voc_dataset.py data/VOC --move-no-label

  # 仅检查不修改（dry-run 模式）
  python verify_voc_dataset.py data/VOC --dry-run

  # 直接删除无标注图像（不移动）
  python verify_voc_dataset.py data/VOC --delete-no-label

  # 指定特定 split
  python verify_voc_dataset.py data/VOC --splits train val
        """,
    )

    parser.add_argument(
        "data_dir",
        type=str,
        help="VOC 数据集根目录（包含 JPEGImages/, Annotations/, ImageSets/）",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="要检查的 split 列表 (默认: train val test)",
    )

    parser.add_argument(
        "--move-no-label",
        action="store_true",
        help="将无标注的图像移动到 no_label 目录（默认行为）",
    )

    parser.add_argument(
        "--delete-no-label",
        action="store_true",
        help="直接删除无标注的图像（危险！）",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="仅检查不修改，输出将要执行的操作",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="静默模式",
    )

    return parser.parse_args()


def find_image_file(jpeg_dir: Path, item_id: str) -> Path | None:
    """查找 item 对应的图像文件.

    Args:
        jpeg_dir: JPEGImages 目录
        item_id: item ID（不含扩展名）

    Returns:
        图像文件路径或 None
    """
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]:
        img_path = jpeg_dir / f"{item_id}{ext}"
        if img_path.exists():
            return img_path
    return None


def check_item(
    item_id: str,
    jpeg_dir: Path,
    anno_dir: Path,
) -> Tuple[bool, bool]:
    """检查 item 的图像和标注是否存在.

    Args:
        item_id: item ID
        jpeg_dir: 图像目录
        anno_dir: 标注目录

    Returns:
        (image_exists, label_exists)
    """
    img_path = find_image_file(jpeg_dir, item_id)
    xml_path = anno_dir / f"{item_id}.xml"

    return img_path is not None, xml_path.exists()


def verify_split(
    data_dir: Path,
    split: str,
    move_no_label: bool,
    delete_no_label: bool,
    dry_run: bool,
    verbose: bool,
) -> Dict[str, int]:
    """校验指定 split.

    Args:
        data_dir: 数据集根目录
        split: split 名称
        move_no_label: 是否移动无标注图像
        delete_no_label: 是否删除无标注图像
        dry_run: 是否仅检查
        verbose: 是否打印详细信息

    Returns:
        统计信息字典
    """
    jpeg_dir = data_dir / "JPEGImages"
    anno_dir = data_dir / "Annotations"
    imageset_file = data_dir / "ImageSets" / "Main" / f"{split}.txt"
    no_label_dir = data_dir / "no_label" / split

    stats = {
        "total": 0,
        "valid": 0,
        "no_image": 0,
        "no_label": 0,
        "removed_from_list": 0,
    }

    if not imageset_file.exists():
        if verbose:
            print(f"  Warning: ImageSets file not found: {imageset_file}")
        return stats

    # 读取 item 列表
    with open(imageset_file, "r", encoding="utf-8") as f:
        items = [line.strip() for line in f if line.strip()]

    stats["total"] = len(items)

    if not items:
        if verbose:
            print(f"  No items in {split}.txt")
        return stats

    # 检查每个 item
    valid_items: List[str] = []
    items_to_process = tqdm(items, desc=f"  Checking {split}", disable=not TQDM_AVAILABLE or not verbose)

    for item_id in items_to_process:
        has_image, has_label = check_item(item_id, jpeg_dir, anno_dir)

        if has_image and has_label:
            # 正常情况：都有
            valid_items.append(item_id)
            stats["valid"] += 1

        elif not has_image and has_label:
            # 有 label 无图像 -> 删除 label
            xml_path = anno_dir / f"{item_id}.xml"
            if verbose:
                print(f"  [NO IMAGE] {item_id}: removing label {xml_path.name}")

            if not dry_run:
                try:
                    xml_path.unlink()
                except OSError as e:
                    print(f"  Error deleting {xml_path}: {e}")

            stats["no_image"] += 1
            stats["removed_from_list"] += 1

        elif has_image and not has_label:
            # 有图像无 label -> 移动或删除图像
            img_path = find_image_file(jpeg_dir, item_id)
            if verbose:
                print(f"  [NO LABEL] {item_id}: image={img_path.name if img_path else '?'}")

            if not dry_run:
                if delete_no_label:
                    # 直接删除
                    try:
                        img_path.unlink()
                    except OSError as e:
                        print(f"  Error deleting {img_path}: {e}")

                elif move_no_label or True:  # 默认移动
                    # 移动到新目录
                    no_label_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = no_label_dir / img_path.name
                    try:
                        shutil.move(str(img_path), str(dest_path))
                        if verbose:
                            print(f"    -> moved to {dest_path}")
                    except OSError as e:
                        print(f"  Error moving {img_path}: {e}")

            stats["no_label"] += 1
            stats["removed_from_list"] += 1

        else:
            # 都没有
            if verbose:
                print(f"  [BOTH MISSING] {item_id}")
            stats["removed_from_list"] += 1

    # 更新 ImageSets 文件
    if not dry_run and stats["removed_from_list"] > 0:
        if verbose:
            print(f"  Updating {imageset_file}: {len(valid_items)}/{len(items)} items remain")

        # 备份原文件
        backup_path = imageset_file.with_suffix(".txt.bak")
        if not backup_path.exists():
            shutil.copy2(imageset_file, backup_path)

        # 写入新列表
        with open(imageset_file, "w", encoding="utf-8") as f:
            for item_id in valid_items:
                f.write(f"{item_id}\n")

    return stats


def main():
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()

    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)

    # 检查必要目录
    jpeg_dir = data_dir / "JPEGImages"
    anno_dir = data_dir / "Annotations"
    imagesets_dir = data_dir / "ImageSets" / "Main"

    if not jpeg_dir.exists():
        print(f"Error: JPEGImages directory not found: {jpeg_dir}")
        sys.exit(1)

    if not anno_dir.exists():
        print(f"Error: Annotations directory not found: {anno_dir}")
        sys.exit(1)

    if not imagesets_dir.exists():
        print(f"Error: ImageSets/Main directory not found: {imagesets_dir}")
        sys.exit(1)

    verbose = not args.quiet

    if verbose:
        print(f"Verifying VOC dataset: {data_dir}")
        print(f"Splits to check: {', '.join(args.splits)}")
        if args.dry_run:
            print("[DRY RUN - no changes will be made]")
        print("-" * 60)

    # 处理每个 split
    total_stats = {
        "total": 0,
        "valid": 0,
        "no_image": 0,
        "no_label": 0,
        "removed_from_list": 0,
    }

    for split in args.splits:
        if verbose:
            print(f"\nProcessing split: {split}")

        stats = verify_split(
            data_dir,
            split,
            move_no_label=args.move_no_label,
            delete_no_label=args.delete_no_label,
            dry_run=args.dry_run,
            verbose=verbose,
        )

        for key in total_stats:
            total_stats[key] += stats[key]

    # 输出总结
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total items checked:     {total_stats['total']}")
        print(f"Valid (both exist):      {total_stats['valid']}")
        print(f"No image (label only):   {total_stats['no_image']} -> labels removed")
        print(f"No label (image only):   {total_stats['no_label']} -> images moved to no_label/")
        print(f"Removed from lists:      {total_stats['removed_from_list']}")

        if args.dry_run:
            print("\n[DRY RUN - no changes were made]")
            print("Run without --dry-run to apply changes")

        print("=" * 60)


if __name__ == "__main__":
    main()
