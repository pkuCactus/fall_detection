import argparse
import json
import os
import sys
import tempfile
import itertools

import cv2
import numpy as np
import yaml

sys.path.insert(0, "src")
from fall_detection.pipeline import FallDetectionPipeline


def build_pipeline(base_config_path, cfg, mock_detector):
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.setdefault("rules", {})["trigger_thresh"] = cfg["trigger_thresh"]
    config.setdefault("fusion", {})["alarm_thresh"] = cfg["alarm_thresh"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        tmp_path = f.name
    pipeline = FallDetectionPipeline(tmp_path)
    os.unlink(tmp_path)
    if mock_detector:
        pipeline.detector = lambda img, conf_thresh=0.3: [
            {"bbox": [100.0, 100.0, 200.0, 400.0], "conf": 0.9, "class_id": 0}
        ]
    return pipeline


def evaluate_video(pipeline, video_path, gt_segments, cfg):
    """gt_segments: List[(start_frame, end_frame)] or []."""
    cap = cv2.VideoCapture(video_path)
    fall_frames = 0
    total_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        results = pipeline.process_frame(frame)
        if any(results.get("track_falling", {}).values()):
            fall_frames += 1
    cap.release()

    predicted = fall_frames > 0
    actual = len(gt_segments) > 0
    tp = int(predicted and actual)
    fp = int(predicted and not actual)
    fn = int(not predicted and actual)
    return {"tp": tp, "fp": fp, "fn": fn, "fall_frames": fall_frames, "total_frames": total_frames}


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline evaluation and threshold search")
    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--gt-file", default="data/event_gt.json")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="train/eval/eval_result.json")
    parser.add_argument("--mock-detector", action="store_true", help="Use fixed mock detector for speed")
    args = parser.parse_args()

    gt_data = {}
    if os.path.exists(args.gt_file):
        with open(args.gt_file, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

    video_files = []
    if os.path.isdir(args.video_dir):
        video_files = sorted([
            os.path.join(args.video_dir, f)
            for f in os.listdir(args.video_dir)
            if f.endswith((".mp4", ".avi", ".mov"))
        ])

    if not video_files:
        video_files = [None]

    grid = {
        "trigger_thresh": [0.5, 0.6, 0.7],
        "alarm_thresh": [0.6, 0.7, 0.8],
    }

    best_score = -1e9
    best_cfg = None
    all_results = []
    for trigger_thresh, alarm_thresh in itertools.product(*grid.values()):
        cfg = {"trigger_thresh": trigger_thresh, "alarm_thresh": alarm_thresh}
        pipeline = build_pipeline(args.config, cfg, args.mock_detector)
        tps = fps = fns = 0
        for vpath in video_files:
            if vpath is None:
                continue
            name = os.path.basename(vpath)
            segments = gt_data.get(name, [])
            res = evaluate_video(pipeline, vpath, segments, cfg)
            tps += res["tp"]
            fps += res["fp"]
            fns += res["fn"]
        precision = tps / max(1, tps + fps)
        recall = tps / max(1, tps + fns)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        row = {**cfg, "precision": precision, "recall": recall, "f1": f1, "tp": tps, "fp": fps, "fn": fns}
        all_results.append(row)
        if f1 > best_score:
            best_score = f1
            best_cfg = cfg
        print(row)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"best": best_cfg, "results": all_results}, f, indent=2)
    print(f"Best config saved to {args.output}: {best_cfg}")


if __name__ == "__main__":
    main()
