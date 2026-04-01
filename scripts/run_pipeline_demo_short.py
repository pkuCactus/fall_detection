#!/usr/bin/env python3
"""Pipeline演示脚本（短版本，只处理前N帧）."""

import argparse
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from fall_detection.pipeline import FallDetectionPipeline
from fall_detection.utils import draw_results


def main():
    parser = argparse.ArgumentParser(description="Fall Detection Pipeline Demo (Short)")
    parser.add_argument("--video", default="data/fall_test.mp4",
                        help="Video path")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config file path")
    parser.add_argument("--output", default="data/pipeline_output_short.mp4",
                        help="Output video path")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Maximum frames to process")
    args = parser.parse_args()

    print("Initializing pipeline...")
    pipeline = FallDetectionPipeline(args.config)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")
    print(f"Will process: {min(args.max_frames, total_frames)} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    frame_idx = 0
    detections_count = 0

    try:
        while frame_idx < args.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            results = pipeline.process_frame(frame)

            # 统计检测到的目标
            if results.get("tracks"):
                detections_count += len(results["tracks"])

            # 绘制结果
            frame = draw_results(
                frame,
                results["tracks"],
                results["track_kpts"],
                results["track_scores"],
                results["track_falling"],
            )

            # 添加帧信息
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames...")

    finally:
        cap.release()
        writer.release()
        print(f"\nDone! Processed {frame_idx} frames, detected {detections_count} track instances")
        print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
