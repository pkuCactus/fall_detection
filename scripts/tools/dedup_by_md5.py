#!/usr/bin/env python3
"""根据 MD5 对指定目录的文件进行去重.

如果去重后目录为空，则删除对应目录.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def calculate_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """计算文件的 MD5 哈希值.

    Args:
        file_path: 文件路径
        chunk_size: 读取块大小

    Returns:
        MD5 哈希字符串
    """
    md5_hash = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except (IOError, OSError) as e:
        print(f"Warning: Cannot read {file_path}: {e}")
        return ""


def get_files_recursive(directory: Path) -> List[Path]:
    """递归获取目录下所有文件.

    Args:
        directory: 目标目录

    Returns:
        文件路径列表
    """
    files = []
    for item in directory.rglob("*"):
        if item.is_file():
            files.append(item)
    return sorted(files)


def dedup_directory(
    directory: Path,
    dry_run: bool = False,
    remove_empty_dirs: bool = True,
    verbose: bool = True,
) -> Tuple[int, int, int, List[Path]]:
    """对目录进行 MD5 去重.

    Args:
        directory: 目标目录
        dry_run: 是否只显示而不实际删除
        remove_empty_dirs: 是否删除空目录
        verbose: 是否打印详细信息

    Returns:
        (保留文件数, 删除文件数, 删除目录数, 空目录列表)
    """
    if not directory.exists():
        print(f"Error: Directory does not exist: {directory}")
        return 0, 0, 0, []

    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}")
        return 0, 0, 0, []

    files = get_files_recursive(directory)

    if not files:
        print(f"No files found in: {directory}")
        # 如果目录本身为空，标记为待删除
        if remove_empty_dirs:
            return 0, 0, 0, [directory]
        return 0, 0, 0, []

    if verbose:
        print(f"Scanning {len(files)} files in: {directory}")

    # MD5 -> (文件路径, 文件大小) 列表
    md5_map: Dict[str, List[Tuple[Path, int]]] = {}

    # 计算所有文件的 MD5
    for file_path in files:
        file_size = file_path.stat().st_size

        # 快速跳过：如果已经有一个相同大小的文件，才计算 MD5
        md5_key = None
        for existing_md5, existing_files in md5_map.items():
            if existing_files and existing_files[0][1] == file_size:
                # 潜在重复，计算 MD5
                md5_key = calculate_md5(file_path)
                break
        else:
            # 没有相同大小的文件，计算 MD5
            md5_key = calculate_md5(file_path)

        if not md5_key:
            continue

        if md5_key not in md5_map:
            md5_map[md5_key] = []
        md5_map[md5_key].append((file_path, file_size))

    # 去重：保留每个 MD5 组的第一个文件，删除其余
    kept = 0
    deleted = 0
    deleted_dirs: Set[Path] = set()

    for md5_key, file_list in md5_map.items():
        if len(file_list) > 1:
            if verbose:
                print(f"\nDuplicate group (MD5: {md5_key[:16]}...):")
                for i, (fp, size) in enumerate(file_list):
                    marker = " [KEEP]" if i == 0 else " [DELETE]"
                    print(f"  {fp} ({size} bytes){marker}")

            # 保留第一个，删除其余的
            for file_path, _ in file_list[1:]:
                if not dry_run:
                    try:
                        os.remove(file_path)
                        deleted_dirs.add(file_path.parent)
                        deleted += 1
                    except OSError as e:
                        print(f"Error deleting {file_path}: {e}")
                else:
                    deleted += 1

            kept += 1
        else:
            kept += 1

    # 检查并删除空目录
    empty_dirs_to_remove: List[Path] = []
    if remove_empty_dirs:
        # 收集所有子目录（从深到浅）
        all_dirs = sorted(
            [d for d in directory.rglob("*") if d.is_dir()],
            key=lambda x: len(x.parts),
            reverse=True,
        )

        # 加上根目录
        all_dirs.append(directory)

        for dir_path in all_dirs:
            # 检查目录是否为空（或只包含空子目录）
            try:
                remaining = list(dir_path.iterdir())
                # 过滤掉即将被删除的子目录
                remaining = [
                    r for r in remaining if r not in empty_dirs_to_remove and r.exists()
                ]

                if not remaining:
                    empty_dirs_to_remove.append(dir_path)
                    if not dry_run:
                        try:
                            dir_path.rmdir()
                            if verbose:
                                print(f"  Removed empty dir: {dir_path}")
                        except OSError as e:
                            print(f"Warning: Cannot remove dir {dir_path}: {e}")
            except OSError:
                pass

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Summary for: {directory}")
        print(f"  Files kept:   {kept}")
        print(f"  Files deleted: {deleted}")
        print(f"  Empty dirs removed: {len(empty_dirs_to_remove)}")
        if dry_run:
            print("  [DRY RUN - no changes made]")
        print(f"{'=' * 50}")

    return kept, deleted, len(empty_dirs_to_remove), empty_dirs_to_remove


def main():
    parser = argparse.ArgumentParser(
        description="根据 MD5 对目录文件去重，空目录自动删除",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 去重单个目录
  python dedup_by_md5.py data/images

  # 去重多个目录
  python dedup_by_md5.py data/train data/val data/test

  # 模拟运行（不实际删除）
  python dedup_by_md5.py --dry-run data/images

  # 保留空目录
  python dedup_by_md5.py --keep-empty-dirs data/images
        """,
    )

    parser.add_argument(
        "directories",
        nargs="+",
        help="要处理的目录（可指定多个）",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="模拟运行，显示将要删除的文件但不实际删除",
    )

    parser.add_argument(
        "--keep-empty-dirs",
        "-k",
        action="store_true",
        help="保留空目录（默认删除）",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="静默模式，只输出 summary",
    )

    args = parser.parse_args()

    total_kept = 0
    total_deleted = 0
    total_dirs_removed = 0

    for directory in args.directories:
        dir_path = Path(directory).resolve()
        print(f"\nProcessing: {dir_path}")

        kept, deleted, dirs_removed, _ = dedup_directory(
            dir_path,
            dry_run=args.dry_run,
            remove_empty_dirs=not args.keep_empty_dirs,
            verbose=not args.quiet,
        )

        total_kept += kept
        total_deleted += deleted
        total_dirs_removed += dirs_removed

    # 最终总结
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total files kept:    {total_kept}")
    print(f"Total files deleted: {total_deleted}")
    print(f"Total dirs removed:  {total_dirs_removed}")
    if args.dry_run:
        print("\n[DRY RUN - no changes were made]")
        print("Remove --dry-run to perform actual deletion")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
