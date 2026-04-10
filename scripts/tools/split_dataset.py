#!/usr/bin/env python
"""Split VOC format dataset into train/val/test sets.

This script takes a VOC format dataset (JPEGImages and Annotations) and creates
ImageSets/Main/{train,val,test}.txt files for dataset splitting.
"""

import argparse
import os
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split VOC dataset into train/val/test sets")
    parser.add_argument("-i", "--image-dir", required=True, help="Path to JPEGImages directory")
    parser.add_argument("-a", "--anno-dir", required=True, help="Path to Annotations directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory (will create VOC structure)")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--copy", action="store_true", help="Copy images and annotations to output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio})")
        return 1

    # Setup paths
    image_dir = Path(args.image_dir)
    anno_dir = Path(args.anno_dir)
    output_dir = Path(args.output)

    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return 1

    if not anno_dir.exists():
        print(f"Error: Annotation directory not found: {anno_dir}")
        return 1

    # Get list of image files (without extension)
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        for img_path in image_dir.glob(ext):
            # Check if corresponding annotation exists
            anno_path = anno_dir / f"{img_path.stem}.xml"
            if anno_path.exists():
                image_files.append(img_path.stem)

    if not image_files:
        print(f"Error: No images with annotations found")
        return 1

    print(f"Found {len(image_files)} images with annotations")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(image_files)

    n_total = len(image_files)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)

    train_files = image_files[:n_train]
    val_files = image_files[n_train : n_train + n_val]
    test_files = image_files[n_train + n_val :]

    print(f"Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # Create output directories
    output_images = output_dir / "JPEGImages"
    output_annotations = output_dir / "Annotations"
    output_imagesets = output_dir / "ImageSets" / "Main"

    output_images.mkdir(parents=True, exist_ok=True)
    output_annotations.mkdir(parents=True, exist_ok=True)
    output_imagesets.mkdir(parents=True, exist_ok=True)

    # Write ImageSets files
    (output_imagesets / "train.txt").write_text("\n".join(train_files) + "\n")
    (output_imagesets / "val.txt").write_text("\n".join(val_files) + "\n")
    (output_imagesets / "test.txt").write_text("\n".join(test_files) + "\n")

    print(f"Created ImageSets in {output_imagesets}")

    # Copy files if requested
    if args.copy:
        import shutil

        print("Copying images and annotations...")

        for stem in image_files:
            # Find image file
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                src_img = image_dir / f"{stem}{ext}"
                if src_img.exists():
                    shutil.copy2(src_img, output_images / f"{stem}{ext}")
                    break

            # Copy annotation
            src_anno = anno_dir / f"{stem}.xml"
            if src_anno.exists():
                shutil.copy2(src_anno, output_annotations / f"{stem}.xml")

        print(f"Copied {len(image_files)} images and annotations to {output_dir}")

    print("\nDone!")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(
        f"  ├── JPEGImages/       ({len(image_files)} images)" if args.copy else "  ├── JPEGImages/       (not copied)"
    )
    print(f"  ├── Annotations/      ({len(image_files)} xmls)" if args.copy else "  ├── Annotations/      (not copied)")
    print(f"  └── ImageSets/")
    print(f"      └── Main/")
    print(f"          ├── train.txt ({len(train_files)} images)")
    print(f"          ├── val.txt   ({len(val_files)} images)")
    print(f"          └── test.txt  ({len(test_files)} images)")

    return 0


if __name__ == "__main__":
    exit(main())
