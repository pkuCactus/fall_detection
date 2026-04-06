"""Split VOC format dataset into train/val/test sets.

Creates ImageSets/Main/{train,val,test}.txt files for VOC format dataset.
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def split_dataset(
    image_dir: str,
    anno_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy_files: bool = False,
):
    """
    Split dataset into train/val/test sets.

    Args:
        image_dir: Directory containing JPEGImages
        anno_dir: Directory containing Annotations (XML files)
        output_dir: Output directory for split dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed
        copy_files: If True, copy files; otherwise create symlinks/split files only
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)

    # Find all images with corresponding annotations
    image_dir = Path(image_dir)
    anno_dir = Path(anno_dir)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']

    valid_samples = []
    for img_path in image_dir.iterdir():
        if img_path.suffix in image_extensions:
            anno_path = anno_dir / f"{img_path.stem}.xml"
            if anno_path.exists():
                valid_samples.append(img_path.stem)

    print(f"Found {len(valid_samples)} valid samples (with both image and annotation)")

    # Shuffle and split
    random.shuffle(valid_samples)

    n_train = int(len(valid_samples) * train_ratio)
    n_val = int(len(valid_samples) * val_ratio)
    # Test gets remaining samples to avoid rounding issues

    train_ids = valid_samples[:n_train]
    val_ids = valid_samples[n_train:n_train + n_val]
    test_ids = valid_samples[n_train + n_val:]

    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Create output structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create directories
    jpeg_dir = output_dir / "JPEGImages"
    anno_out_dir = output_dir / "Annotations"
    sets_dir = output_dir / "ImageSets" / "Main"

    for d in [jpeg_dir, anno_out_dir, sets_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Write ImageSets
    def write_set(ids, name):
        with open(sets_dir / f"{name}.txt", "w") as f:
            for id in ids:
                f.write(f"{id}\n")

    write_set(train_ids, "train")
    write_set(val_ids, "val")
    write_set(test_ids, "test")
    write_set(train_ids + val_ids, "trainval")

    print(f"Written ImageSets to: {sets_dir}")

    # Copy or link files
    def process_files(ids, split_name):
        for id in ids:
            # Find image with any extension
            src_img = None
            for ext in image_extensions:
                candidate = image_dir / f"{id}{ext}"
                if candidate.exists():
                    src_img = candidate
                    break

            if src_img is None:
                print(f"Warning: Image not found for {id}")
                continue

            src_anno = anno_dir / f"{id}.xml"

            dst_img = jpeg_dir / src_img.name
            dst_anno = anno_out_dir / f"{id}.xml"

            if copy_files:
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_anno, dst_anno)
            else:
                # Create symlinks if not exists
                if not dst_img.exists():
                    os.symlink(src_img.absolute(), dst_img)
                if not dst_anno.exists():
                    os.symlink(src_anno.absolute(), dst_anno)

        print(f"{split_name}: processed {len(ids)} files")

    process_files(train_ids, "train")
    process_files(val_ids, "val")
    process_files(test_ids, "test")

    print(f"\nDataset split complete!")
    print(f"Output directory: {output_dir}")
    print(f"\nVOC structure:")
    print(f"  {output_dir}/JPEGImages/     - Images")
    print(f"  {output_dir}/Annotations/     - XML annotations")
    print(f"  {output_dir}/ImageSets/Main/  - Train/val/test splits")


def main():
    parser = argparse.ArgumentParser(description="Split VOC dataset into train/val/test")
    parser.add_argument("--image-dir", "-i", required=True, help="Directory with images")
    parser.add_argument("--anno-dir", "-a", required=True, help="Directory with XML annotations")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinks")
    args = parser.parse_args()

    split_dataset(
        args.image_dir,
        args.anno_dir,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.copy,
    )


if __name__ == "__main__":
    main()
