#!/usr/bin/env python3
"""YOLO-World 多姿态跌倒检测 Pipeline 演示脚本."""

import argparse
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, "src")
from fall_detection.pipeline.yolo_world_pipeline import YOLOWorldFallPipeline
from fall_detection.utils.visualization import draw_results


def main():
    parser = argparse.ArgumentParser(description="YOLO-World Fall Detection Pipeline Demo")
    parser.add_argument("--video", default="data/sample.mp4",
                        help="Video path (or 0 for webcam)")
    parser.add_argument("--config", default="configs/pipeline/yolo_world.yaml",
                        help="Config file path")
    parser.add_argument("--output", default=None,
                        help="Output video path")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI display")
    parser.add_argument("--device", default=None,
                        help="Device to use (e.g., 'cuda', 'cpu'). Auto-detect if not specified.")
    args = parser.parse_args()

    print("Initializing YOLO-World fall pipeline...")
    pipeline = YOLOWorldFallPipeline(args.config, device=args.device)
    print(f"  Using device: {args.device if args.device else 'auto'}")
    print(f"  Detector input size: {pipeline.detector.input_size}x{pipeline.detector.input_size}")
    print(f"  Skip frames: {pipeline.skip_frames}, FPS: {pipeline.fps}")

    video_path = 0 if args.video == "0" else args.video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Warning: video not found, generating blank test frames.")
        for i in range(100):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            results = pipeline.process_frame(frame)
            frame = draw_results(
                frame,
                results["tracks"],
                results["track_kpts"],
                results["track_scores"],
                results["track_falling"],
                results.get("fusion_histories", {}),
            )
            if not args.headless:
                cv2.imshow("YOLO-World Fall Demo", frame)
                if cv2.waitKey(100) == 27:
                    break
        if not args.headless:
            cv2.destroyAllWindows()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / fps)

    writer = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        print(f"Output video: {args.output}")

    if not args.headless:
        print("\nControls: ESC - quit")

    frame_idx = 0
    new_alarm_count = 0
    active_alarms = set()

    try:
        while True:
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
                results.get("fusion_histories", {}),
            )

            # 帧信息
            track_scores = results.get("track_scores", {})
            info_bar_height = 25 * len(track_scores) + 40 if track_scores else 50
            cv2.putText(frame, f"Frame: {frame_idx}", (10, info_bar_height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Alarms: {new_alarm_count}", (10, info_bar_height + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            for tid in results.get("new_alarms", []):
                if tid not in active_alarms:
                    new_alarm_count += 1
                    active_alarms.add(tid)
                    print(f"*** ALARM: New fall detected for Track {tid} at frame {frame_idx} ***")

            current_falling = {tid for tid, falling in results.get("track_falling", {}).items() if falling}
            active_alarms &= current_falling

            if writer:
                writer.write(frame)

            if not args.headless:
                cv2.imshow("YOLO-World Fall Demo", frame)
                if cv2.waitKey(delay) & 0xFF == 27:
                    break
            else:
                if frame_idx % 30 == 0:
                    active_tracks = len(results.get("tracks", []))
                    is_falling = any(results.get("track_falling", {}).values())
                    print(f"Frame {frame_idx}: {active_tracks} tracks, fall={is_falling}")

            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"\nDone. Processed {frame_idx} frames. New alarm events: {new_alarm_count}")


if __name__ == "__main__":
    main()
