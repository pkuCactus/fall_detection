#!/usr/bin/env python3
"""Pipeline演示脚本 - 展示完整的跌倒检测流程."""

import argparse
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from fall_detection.pipeline import FallDetectionPipeline
from fall_detection.utils import draw_results


def main():
    parser = argparse.ArgumentParser(description="Fall Detection Pipeline Demo")
    parser.add_argument("--video", default="data/sample.mp4",
                        help="Video path (or 0 for webcam)")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config file path")
    parser.add_argument("--output", default=None,
                        help="Output video path")
    parser.add_argument("--save-frames", default=None,
                        help="Directory to save frames")
    args = parser.parse_args()

    print("Initializing pipeline...")
    pipeline = FallDetectionPipeline(args.config)

    video_path = 0 if args.video == "0" else args.video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Warning: video not found, generating blank test frames.")
        # 生成测试序列
        for i in range(100):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            results = pipeline.process_frame(frame)
            frame = draw_results(
                frame,
                results["tracks"],
                results["track_kpts"],
                results["track_scores"],
                results["track_falling"],
            )
            cv2.imshow("Pipeline Demo", frame)
            if cv2.waitKey(100) == 27:
                break
        cv2.destroyAllWindows()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / fps)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    if args.save_frames:
        import os
        os.makedirs(args.save_frames, exist_ok=True)

    print("\nControls:")
    print("  ESC - quit")
    print("  p - pause/resume")
    print("  s - save current frame")

    paused = False
    frame_idx = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break

            results = pipeline.process_frame(frame)
            frame = draw_results(
                frame,
                results["tracks"],
                results["track_kpts"],
                results["track_scores"],
                results["track_falling"],
            )

            # 添加帧信息
            cv2.putText(frame, f"Frame: {frame_idx}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            frame_idx += 1
        else:
            cv2.putText(frame, "PAUSED", (w//2 - 60, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow("Pipeline Demo", frame)

        if writer:
            writer.write(frame)

        if args.save_frames:
            fname = f"{args.save_frames}/frame_{frame_idx:06d}.jpg"
            cv2.imwrite(fname, frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('s'):
            fname = f"pipeline_frame_{frame_idx:04d}.png"
            cv2.imwrite(fname, frame)
            print(f"Saved: {fname}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
