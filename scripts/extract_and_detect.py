#!/usr/bin/env python3
"""抽取视频特定帧进行检测分析.

支持按帧号或时间戳抽取，可用于调试特定帧的检测效果.
"""

import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, "src")
from fall_detection.pipeline import FallDetectionPipeline
from fall_detection.utils import draw_results


def parse_frame_spec(spec: str, total_frames: int, fps: float) -> list:
    """解析帧规格字符串，返回帧号列表.

    支持的格式:
    - 单帧: "100"
    - 多帧: "100,200,300"
    - 范围: "100-200" 或 "100-200:10" (步长10)
    - 时间点: "10.5s" 或 "1:30" (1分30秒)
    - 百分比: "50%"
    """
    frames = set()

    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue

        # 百分比格式: 50%
        if part.endswith('%'):
            pct = float(part[:-1]) / 100.0
            frames.add(int(pct * total_frames))

        # 时间格式: 10.5s
        elif part.endswith('s'):
            seconds = float(part[:-1])
            frames.add(int(seconds * fps))

        # 范围格式: 100-200 或 100-200:10 (包含-)
        elif '-' in part and not ':' in part.split('-')[0]:
            range_parts = part.split('-')
            start = int(range_parts[0])

            if ':' in range_parts[1]:
                end, step = map(int, range_parts[1].split(':'))
            else:
                end = int(range_parts[1])
                step = 1

            for f in range(start, min(end + 1, total_frames), step):
                frames.add(f)

        # 时间格式: 1:30 或 1:30:00 (只包含:)
        elif ':' in part:
            parts = part.split(':')
            if len(parts) == 2:
                minutes, seconds = map(int, parts)
                frames.add(int((minutes * 60 + seconds) * fps))
            elif len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                frames.add(int((hours * 3600 + minutes * 60 + seconds) * fps))

        # 单帧格式: 100
        else:
            try:
                frames.add(int(part))
            except ValueError:
                print(f"Warning: Cannot parse '{part}', skipping")

    # 过滤无效帧号
    valid_frames = [f for f in sorted(frames) if 0 <= f < total_frames]
    return valid_frames


def extract_frames(video_path: str, frame_indices: list) -> list:
    """从视频中抽取指定帧."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    frames = []
    current_frame = 0
    frame_idx_set = set(frame_indices)
    max_frame = max(frame_indices) if frame_indices else 0

    while current_frame <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in frame_idx_set:
            frames.append((current_frame, frame.copy()))

        current_frame += 1

    cap.release()
    return frames


def detect_and_visualize(pipeline, frames: list, output_dir: str, save_original: bool = False):
    """对抽取的帧进行检测并保存结果."""
    os.makedirs(output_dir, exist_ok=True)

    results_summary = []

    for frame_idx, frame in frames:
        print(f"\nProcessing frame {frame_idx}...")

        # 运行检测
        results = pipeline.process_frame(frame)

        # 绘制结果
        vis_frame = draw_results(
            frame.copy(),
            results["tracks"],
            results["track_kpts"],
            results["track_scores"],
            results["track_falling"],
        )

        # 绘制所有原始检测框（蓝色），方便查看哪些框没有匹配到 track
        detections = results.get("detections", [])
        if detections:
            for i, bbox in enumerate(detections):
                x1, y1, x2, y2 = map(int, bbox)
                # 蓝色框表示 YOLO 检测框
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis_frame, f"D{i+1}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 添加帧信息
        h, w = frame.shape[:2]
        is_detection = results.get("is_detection_frame", True)
        phase = "DETECTION" if is_detection else "TRACKING"
        cv2.putText(vis_frame, f"Frame: {frame_idx} ({phase})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 检测统计
        num_tracks = len(results.get("tracks", []))
        num_falling = sum(results.get("track_falling", {}).values())
        cv2.putText(vis_frame, f"YOLO: {len(detections)}, Tracks: {num_tracks}, Falling: {num_falling}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 保存可视化结果
        output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}_detected.jpg")
        cv2.imwrite(output_path, vis_frame)
        print(f"  Saved: {output_path}")

        # 保存原图（可选）
        if save_original:
            orig_path = os.path.join(output_dir, f"frame_{frame_idx:06d}_original.jpg")
            cv2.imwrite(orig_path, frame)

        # 收集详细信息
        frame_info = {
            "frame_idx": frame_idx,
            "phase": phase,
            "detections": results.get("detections", []),
            "num_tracks": num_tracks,
            "tracks": [],
            "falling": [],
        }

        for track in results.get("tracks", []):
            tid = track.track_id
            bbox = track.to_tlbr().tolist()
            track_info = {
                "id": tid,
                "bbox": bbox,
                "state": getattr(track, 'state', 'unknown'),
            }
            frame_info["tracks"].append(track_info)

            if results.get("track_falling", {}).get(tid, False):
                frame_info["falling"].append(tid)

        results_summary.append(frame_info)

        # 打印详细信息
        print(f"  Phase: {phase}")
        if is_detection:
            detections = results.get('detections', [])
            print(f"  Detections (YOLO): {len(detections)}")
            for i, bbox in enumerate(detections):
                print(f"    [Det {i+1}] BBox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        print(f"  Tracks: {num_tracks}")
        for track in results.get("tracks", []):
            tid = track.track_id
            bbox = track.to_tlbr()
            state = getattr(track, 'state', 'unknown')
            hits = getattr(track, 'hits', 0)
            time_since = getattr(track, 'time_since_update', 0)
            is_falling = results.get("track_falling", {}).get(tid, False)
            status = "FALLING!" if is_falling else "normal"
            print(f"    Track {tid}: BBox=[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}] state={state},hits={hits},miss={time_since}f {status}")

    return results_summary


def generate_report(results: list, output_path: str):
    """生成检测报告."""
    with open(output_path, 'w') as f:
        f.write("# Frame Detection Report\n\n")
        f.write("| Frame | Phase | Detections | Tracks | Falling |\n")
        f.write("|-------|-------|------------|--------|----------|\n")

        for info in results:
            falling_str = ", ".join(map(str, info["falling"])) if info["falling"] else "-"
            f.write(f"| {info['frame_idx']} | {info['phase']} | {len(info['detections'])} | "
                    f"{info['num_tracks']} | {falling_str} |\n")

        f.write("\n## Detailed Information\n\n")
        for info in results:
            f.write(f"### Frame {info['frame_idx']}\n")
            f.write(f"- Phase: {info['phase']}\n")
            f.write(f"- Detections: {len(info['detections'])}\n")
            f.write(f"- Tracks: {info['num_tracks']}\n")
            f.write(f"- Falling: {info['falling']}\n")

            if info['detections']:
                f.write("\n**Detection Boxes:**\n")
                for i, bbox in enumerate(info['detections']):
                    f.write(f"  - Det {i+1}: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]\n")

            if info['tracks']:
                f.write("\n**Track Information:**\n")
                for track in info['tracks']:
                    f.write(f"  - Track {track['id']}: BBox={track['bbox']}, State={track['state']}\n")
            f.write("\n")

    print(f"\nReport saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract specific frames from video and run detection"
    )
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--frames", "-f", required=True,
                        help="Frame specification (e.g., '100', '100,200,300', '0-100:10', '10.5s', '1:30')")
    parser.add_argument("--config", "-c", default="configs/default.yaml",
                        help="Config file path")
    parser.add_argument("--output", "-o", default="output/extracted_frames",
                        help="Output directory")
    parser.add_argument("--save-original", action="store_true",
                        help="Save original frames without overlay")
    parser.add_argument("--report", "-r", action="store_true",
                        help="Generate markdown report")
    parser.add_argument("--list", "-l", action="store_true",
                        help="Only list frames without processing")

    args = parser.parse_args()

    # 打开视频获取信息
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video Info:")
    print(f"  Path: {args.video}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f}s ({int(duration//60)}:{int(duration%60):02d})")

    # 解析帧规格
    frame_indices = parse_frame_spec(args.frames, total_frames, fps)

    if not frame_indices:
        print("\nError: No valid frames specified")
        return 1

    print(f"\nWill process {len(frame_indices)} frames:")
    print(f"  {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")

    if args.list:
        print("\nFrame list (timestamp):")
        for f in frame_indices:
            ts = f / fps
            print(f"  Frame {f:6d} @ {ts:8.3f}s ({int(ts//60)}:{int(ts%60):02d}.{int((ts%1)*1000):03d})")
        return 0

    cap.release()

    # 初始化 pipeline
    print("\nInitializing pipeline...")
    pipeline = FallDetectionPipeline(args.config)

    # 抽取帧
    print(f"\nExtracting {len(frame_indices)} frames...")
    frames = extract_frames(args.video, frame_indices)
    print(f"Extracted {len(frames)} frames")

    # 检测并可视化
    print("\nRunning detection...")
    results = detect_and_visualize(
        pipeline, frames, args.output, args.save_original
    )

    # 生成报告
    if args.report:
        report_path = os.path.join(args.output, "report.md")
        generate_report(results, report_path)

    print(f"\n{'='*60}")
    print(f"Done! Processed {len(results)} frames.")
    print(f"Output directory: {args.output}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
