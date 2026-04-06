#!/usr/bin/env python3
"""Convert multi-class YOLO labels to single class.

将所有类别ID映射到0（person）。
"""

import argparse
import shutil
from pathlib import Path


def convert_labels(input_dir, output_dir):
    """Convert all label files to single class (class 0)."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory not found: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Converting: {input_path} -> {output_path}")

    label_files = list(input_path.rglob("*.txt"))
    print(f"Found {len(label_files)} label files")

    converted = 0
    skipped = 0

    for label_file in label_files:
        # 保持目录结构
        rel_path = label_file.relative_to(input_path)
        out_file = output_path / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(label_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # 将类别ID改为0，保留bbox坐标
                parts[0] = '0'
                new_lines.append(' '.join(parts) + '\n')

        with open(out_file, 'w') as f:
            f.writelines(new_lines)
        converted += 1

    print(f"Done! Converted {converted} files. All classes mapped to 0 (person).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO labels to single class")
    parser.add_argument("--input-dir", required=True, help="Input labels directory (e.g., data/mini/labels)")
    parser.add_argument("--output-dir", required=True, help="Output labels directory (e.g., data/mini/labels_single)")
    args = parser.parse_args()

    convert_labels(args.input_dir, args.output_dir)
