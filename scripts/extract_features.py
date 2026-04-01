import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from fall_detection.pipeline import FallDetectionPipeline


def parse_label(label_path, video_name, frame_idx):
    if not os.path.exists(label_path):
        return 0
    with open(label_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    info = labels.get(video_name, {})
    label = info.get("label", 0)
    segments = info.get("frames", [])
    if len(segments) == 2 and isinstance(segments[0], int):
        start, end = segments
        return label if start <= frame_idx <= end else 0
    return label


def extract(video_path, pipeline, label, out_dir, sample_fps=5):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip = max(1, int(round(video_fps / sample_fps)))
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        results = pipeline.process_frame(frame)
        for track in results.get("tracks", []):
            tid = track.track_id
            kpts = results["track_kpts"].get(tid)
            if kpts is None:
                continue
            motion = pipeline._extract_motion(
                tid,
                kpts,
                track.to_tlbr().tolist(),
                {"centers": list(pipeline._track_history[tid])},
            )
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = cv2.resize(frame[y1:y2, x1:x2], (96, 96))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            file_name = f"{base_name}_f{frame_idx}_t{tid}.npz"
            np.savez(
                os.path.join(out_dir, file_name),
                roi=roi,
                kpts=kpts.astype(np.float32),
                motion=motion.astype(np.float32),
                label=np.int64(label),
            )
            saved += 1
        frame_idx += 1
    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract classifier features from videos")
    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--label-file", default="data/labels.json")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="train/cache")
    parser.add_argument("--sample-fps", type=int, default=5)
    args = parser.parse_args()

    pipeline = FallDetectionPipeline(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    video_files = []
    if os.path.isdir(args.video_dir):
        video_files = [
            os.path.join(args.video_dir, f)
            for f in os.listdir(args.video_dir)
            if f.endswith((".mp4", ".avi", ".mov"))
        ]

    total_saved = 0
    for vpath in video_files:
        name = os.path.basename(vpath)
        label = 0
        if os.path.exists(args.label_file):
            label = parse_label(args.label_file, name, 0)
        n = extract(vpath, pipeline, label, args.out_dir, args.sample_fps)
        total_saved += n
        print(f"Processed {name}: saved {n} samples")

    print(f"Total saved: {total_saved}")


if __name__ == "__main__":
    main()
