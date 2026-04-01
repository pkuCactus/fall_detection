#!/usr/bin/env python3
"""Tracker演示脚本 - 展示ByteTrack-lite跟踪效果."""

import argparse
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from fall_detection.detector import PersonDetector
from fall_detection.tracker import ByteTrackLite, Detection


def draw_tracks(frame, tracks, track_history=None):
    """绘制跟踪结果."""
    h, w = frame.shape[:2]

    for track in tracks:
        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())

        # 不同ID用不同颜色
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
        ]
        color = colors[tid % len(colors)]

        # 画bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 画ID和状态
        state_text = track.state if hasattr(track, 'state') else 'active'
        label = f"ID:{tid} {state_text}"
        cv2.putText(frame, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 画中心点轨迹
        if track_history and tid in track_history:
            centers = track_history[tid]
            if len(centers) > 1:
                for i in range(1, len(centers)):
                    pt1 = tuple(map(int, centers[i - 1]))
                    pt2 = tuple(map(int, centers[i]))
                    cv2.line(frame, pt1, pt2, color, 2)

    # 显示统计信息
    info_text = [
        f"Active Tracks: {len(tracks)}",
        f"Press 'q' to quit, 'p' to pause"
    ]
    y_offset = 30
    for text in info_text:
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    return frame


def main():
    parser = argparse.ArgumentParser(description="Tracker Demo")
    parser.add_argument("--video", default="data/sample.mp4",
                        help="Video path (or 0 for webcam)")
    parser.add_argument("--output", default=None,
                        help="Output video path")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Detection confidence threshold")
    parser.add_argument("--track-thresh", type=float, default=0.5,
                        help="Tracking threshold")
    parser.add_argument("--match-thresh", type=float, default=0.8,
                        help="Matching IoU threshold")
    parser.add_argument("--max-age", type=int, default=30,
                        help="Max lost frames before deleting track")
    args = parser.parse_args()

    # 初始化检测器和跟踪器
    print("Initializing detector (YOLOv8n)...")
    detector = PersonDetector(model_name="yolov8n")

    print("Initializing tracker (ByteTrack-lite)...")
    tracker = ByteTrackLite(
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        max_age=args.max_age
    )

    # 打开视频
    video_path = 0 if args.video == "0" else args.video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        print("Generating synthetic test video...")
        cap = None

    # 准备输出
    writer = None
    if args.output and cap:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    print("\nControls:")
    print("  q - quit")
    print("  p - pause/resume")
    print("  s - save current frame")

    paused = False
    frame_idx = 0
    track_history = {}  # tid -> list of (cx, cy)

    while True:
        if not paused:
            if cap:
                ret, frame = cap.read()
                if not ret:
                    print("End of video")
                    break
            else:
                # 合成测试帧：移动的人形
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
                t = frame_idx % 200
                cx = 100 + t * 2
                cy = 240
                w, h = 60, 120
                cv2.rectangle(frame, (cx - w//2, cy - h//2),
                             (cx + w//2, cy + h//2), (100, 200, 100), -1)

            frame_idx += 1

            # 检测
            dets = detector(frame, conf_thresh=args.conf)
            detections = [Detection(d["bbox"], d["conf"]) for d in dets]

            # 跟踪
            tracks = tracker.update(detections)

            # 更新轨迹历史
            for track in tracks:
                tid = track.track_id
                tlwh = track.to_tlwh()
                cx = tlwh[0] + tlwh[2] / 2.0
                cy = tlwh[1] + tlwh[3] / 2.0
                if tid not in track_history:
                    track_history[tid] = []
                track_history[tid].append((cx, cy))
                # 限制历史长度
                if len(track_history[tid]) > 50:
                    track_history[tid].pop(0)

            # 绘制
            vis = draw_tracks(frame.copy(), tracks, track_history)

            # 添加检测框信息
            y = 100
            for i, d in enumerate(detections):
                text = f"Det{i}: conf={d.conf:.2f}"
                cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y += 20

        else:
            vis = frame.copy()
            cv2.putText(vis, "PAUSED", (vis.shape[1]//2 - 50, vis.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow("Tracker Demo", vis)

        if writer:
            writer.write(vis)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):
            fname = f"tracker_frame_{frame_idx:04d}.png"
            cv2.imwrite(fname, vis)
            print(f"Saved: {fname}")

    if cap:
        cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
