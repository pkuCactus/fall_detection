import argparse
import json
import itertools
import os
from collections import defaultdict

import cv2
import numpy as np

import sys
sys.path.insert(0, "src")
from fall_detection.core.tracker import ByteTrackLite, Detection


def evaluate_tracker(video_path, detector_fn, cfg):
    cap = cv2.VideoCapture(video_path)
    tracker = ByteTrackLite(
        track_thresh=cfg["track_thresh"],
        match_thresh=cfg["match_thresh"],
        max_age=cfg["max_age"],
        min_hits=cfg["min_hits"],
    )
    id_history = defaultdict(int)
    switches = 0
    prev_active_ids = set()
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        dets = detector_fn(frame)
        detections = [Detection(d["bbox"], d["conf"]) for d in dets]
        active = tracker.update(detections)
        active_ids = {t.track_id for t in active}
        for tid in active_ids:
            id_history[tid] += 1
        if prev_active_ids and not prev_active_ids.intersection(active_ids):
            switches += 1
        prev_active_ids = active_ids
    cap.release()

    avg_life = sum(id_history.values()) / max(1, len(id_history))
    return {
        "frames": frames,
        "avg_life": avg_life,
        "id_switches": switches,
        "score": avg_life - 10 * switches,
    }


def main():
    parser = argparse.ArgumentParser(description="Tune ByteTrack-lite parameters")
    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--output", default="train/tracker/tune_result.json")
    args = parser.parse_args()

    h, w = 480, 640
    mock_dets = [{"bbox": [w * 0.3, h * 0.2, w * 0.7, h * 0.9], "conf": 0.9}]
    detector_fn = lambda frame: mock_dets

    grid = {
        "track_thresh": [0.4, 0.5, 0.6],
        "match_thresh": [0.7, 0.8, 0.9],
        "max_age": [20, 30, 40],
        "min_hits": [2, 3],
    }
    keys = list(grid.keys())
    best_score = -1e9
    best_cfg = None
    results = []
    video_files = [
        os.path.join(args.video_dir, f)
        for f in os.listdir(args.video_dir)
        if f.endswith((".mp4", ".avi"))
    ] if os.path.isdir(args.video_dir) else []

    if not video_files:
        video_files = [None]

    for values in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, values))
        scores = []
        for vf in video_files:
            if vf is None:
                cap_info = {"frames": 50}
                tracker = ByteTrackLite(**cfg)
                id_history = defaultdict(int)
                for _ in range(cap_info["frames"]):
                    dets = [Detection(mock_dets[0]["bbox"], mock_dets[0]["conf"])]
                    active = tracker.update(dets)
                    for t in active:
                        id_history[t.track_id] += 1
                avg_life = sum(id_history.values()) / max(1, len(id_history))
                scores.append({"avg_life": avg_life, "id_switches": 0, "score": avg_life})
            else:
                scores.append(evaluate_tracker(vf, detector_fn, cfg))
        avg_score = sum(s["score"] for s in scores) / len(scores)
        cfg_score = {**cfg, "avg_score": avg_score}
        results.append(cfg_score)
        if avg_score > best_score:
            best_score = avg_score
            best_cfg = cfg
        print(cfg_score)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"best": best_cfg, "results": results}, f, indent=2)
    print(f"Best config: {best_cfg} saved to {args.output}")


if __name__ == "__main__":
    main()
