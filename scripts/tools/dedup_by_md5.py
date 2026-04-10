#!/usr/bin/env python3
"""根据 MD5 对指定目录的文件进行去重.

如果去重后目录为空，则删除对应目录.
"""

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Global lock for thread-safe progress bar updates
try:
    from threading import Lock
    _lock = Lock()
except ImportError:
    _lock = None


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


def calculate_md5_wrapper(
    file_info: Tuple[Path, int],
    progress_cb=None
) -> Tuple[str, Path, int]:
    """包装函数用于并行计算 MD5.

    Args:
        file_info: (file_path, file_size) 元组
        progress_cb: 进度回调函数

    Returns:
        (md5_hash, file_path, file_size)
    """
    file_path, file_size = file_info
    md5_hash = calculate_md5(file_path)
    if progress_cb:
        progress_cb()
    return md5_hash, file_path, file_size


def dedup_directory(
    directory: Path,
    dry_run: bool = False,
    remove_empty_dirs: bool = True,
    verbose: bool = True,
    workers: int = 1,
) -> Tuple[int, int, int, List[Path]]:
    """对目录进行 MD5 去重.

    Args:
        directory: 目标目录
        dry_run: 是否只显示而不实际删除
        remove_empty_dirs: 是否删除空目录
        verbose: 是否打印详细信息
        workers: 并行工作线程数，默认为1（串行）

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
        if workers > 1:
            print(f"Using {workers} workers for parallel processing")

    # MD5 -> (文件路径, 文件大小) 列表
    md5_map: Dict[str, List[Tuple[Path, int]]] = {}

    # 收集文件信息 (path, size)
    file_infos = [(f, f.stat().st_size) for f in files]

    # 按大小分组，找出可能需要计算MD5的文件（大小相同的文件才需要计算MD5）
    size_map: Dict[int, List[Path]] = defaultdict(list)
    for fp, size in file_infos:
        size_map[size].append(fp)

    # 只需要对大小相同的文件计算MD5
    files_to_hash = []
    for size, paths in size_map.items():
        if len(paths) > 1:  # 大小相同才需要计算MD5
            files_to_hash.extend([(p, size) for p in paths])
        else:
            # 唯一大小的文件，直接分配唯一MD5（用路径+大小作为key）
            unique_key = f"unique_{paths[0]}_{size}"
            md5_map[unique_key] = [(paths[0], size)]

    if not files_to_hash:
        if verbose:
            print("No potential duplicates found (all files have unique sizes)")
    else:
        # 计算需要计算MD5的文件的哈希
        if workers > 1:
            # 并行处理
            if TQDM_AVAILABLE and verbose:
                pbar = tqdm(total=len(files_to_hash), desc="Calculating MD5", unit="files")
            else:
                pbar = None

            def update_progress():
                if pbar:
                    pbar.update(1)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_file = {
                    executor.submit(calculate_md5_wrapper, fi, update_progress): fi
                    for fi in files_to_hash
                }

                for future in as_completed(future_to_file):
                    md5_hash, file_path, file_size = future.result()
                    if md5_hash:
                        if md5_hash not in md5_map:
                            md5_map[md5_hash] = []
                        md5_map[md5_hash].append((file_path, file_size))

            if pbar:
                pbar.close()
        else:
            # 串行处理
            if TQDM_AVAILABLE and verbose:
                pbar = tqdm(files_to_hash, desc="Calculating MD5", unit="files")
            else:
                pbar = files_to_hash

            for i, (file_path, file_size) in enumerate(pbar):
                if TQDM_AVAILABLE and verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix_str(f"{file_path.name[:30]}")

                md5_hash = calculate_md5(file_path)
                if md5_hash:
                    if md5_hash not in md5_map:
                        md5_map[md5_hash] = []
                    md5_map[md5_hash].append((file_path, file_size))

                if not TQDM_AVAILABLE and verbose and (i + 1) % 100 == 0:
                    print(f"  Progress: {i + 1}/{len(files_to_hash)} files processed...")

            if TQDM_AVAILABLE and verbose and isinstance(pbar, tqdm):
                pbar.close()

    # 去重：保留每个 MD5 组的第一个文件，删除其余
    kept = 0
    deleted = 0
    deleted_dirs: Set[Path] = set()

    # 统计需要处理的重复组
    duplicate_groups = [(k, v) for k, v in md5_map.items() if len(v) > 1]
    total_duplicates = sum(len(v) - 1 for _, v in duplicate_groups)

    if TQDM_AVAILABLE and verbose and duplicate_groups:
        del_pbar = tqdm(total=total_duplicates, desc="Removing duplicates", unit="files")
    else:
        del_pbar = None

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

                if del_pbar:
                    del_pbar.update(1)

            if verbose and not del_pbar:
                print(f"  Deleted {len(file_list) - 1} duplicate(s)")
            kept += 1
        else:
            kept += 1

    if del_pbar:
        del_pbar.close()

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

        if TQDM_AVAILABLE and verbose and all_dirs:
            dir_pbar = tqdm(all_dirs, desc="Checking empty dirs", unit="dirs")
        else:
            dir_pbar = all_dirs

        for dir_path in dir_pbar:
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

        if isinstance(dir_pbar, tqdm):
            dir_pbar.close()

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

  # 静默模式（无进度条）
  python dedup_by_md5.py --quiet data/images

  # 并行处理（4线程，适合SSD）
  python dedup_by_md5.py -j 4 data/images

性能建议:
  - 机械硬盘: 保持默认 -j 1（串行）
  - SSD: 可设置 -j 4 到 -j 8
  - NVMe: 可设置 -j 8 到 -j 16

依赖:
  安装 tqdm 可获得进度条显示: pip install tqdm
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

    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=1,
        help="并行计算 MD5 的工作线程数 (默认: 1，建议根据磁盘 I/O 能力设置，如 SSD 可设 4-8)",
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
            workers=args.workers,
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
