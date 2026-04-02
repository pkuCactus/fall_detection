# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Edge-AI fall detection system targeting the HiSilicon 3516C platform (0.5T INT8, 15M DDR, 30M Flash). It is a pure vision pipeline: person detection -> tracking -> pose estimation -> rule engine -> optional light classifier -> temporal fusion with a state machine.

## Common commands

- Install dependencies: `pip install -r requirements.txt`
- Run all tests: `bash scripts/shell/run_tests.sh`
- Run a single test file: `PYTHONPATH=src pytest tests/test_pipeline.py -v`
- Run pipeline demo on a video: `python scripts/run_pipeline_demo.py --video data/sample.mp4 --output output.mp4`
- Benchmark speed: `python scripts/benchmark_speed.py --video data/videos/test.mp4 --num-frames 100`
- End-to-end evaluation with threshold grid search: `bash scripts/shell/run_evaluate_pipeline.sh --video-dir data/videos --gt-file data/event_gt.json`
- Training stages (convenience wrappers in `scripts/shell/`):
  - Detector: `bash scripts/shell/run_train_detector.sh`
  - Pose estimator: `bash scripts/shell/run_train_pose.sh`
  - Tracker tuning: `bash scripts/shell/run_tune_tracker.sh --video-dir data/videos`
  - Feature extraction (for classifier): `bash scripts/shell/run_extract_features.sh --video-dir data/videos --label-file data/labels.json --out-dir train/cache`
  - Classifier (supports DDP): `bash scripts/shell/run_train_classifier.sh --ngpus 2 --batch-size 16`
  - Full pipeline: `NGPUS=2 bash scripts/shell/run_all_training.sh`

Tests and most scripts require `PYTHONPATH=src` because the package root is `src/fall_detection/` and imports use `from fall_detection.x import ...`.

## High-level architecture

### Pipeline frame flow (`src/fall_detection/pipeline.py`)

`FallDetectionPipeline` processes every frame but only runs the heavy models on detection frames:
- Detection is triggered every `skip_frames + 1` frames (default `skip_frames: 2`, so every 3rd frame).
- On detection frames: YOLOv8n person detector -> ByteTrack-lite tracker update -> YOLOv8n-pose per-track ROI inference -> RuleEngine -> (optional) FallClassifier -> FusionDecision.
- On skip frames: tracker predicts existing tracks without detection, rules and fusion still run using cached keypoints and cached classifier scores from the last detection frame.

This skip-frame design is central to meeting the edge latency budget (~34ms total, ≤15M DDR).

### Tracker (`src/fall_detection/tracker.py`)

`ByteTrackLite` is a stripped ByteTrack using only IoU + Kalman filter, no ReID. It matches detections in two stages: high-confidence detections first, then low-confidence ones. Track states are `tentative` (until `min_hits`) and `confirmed`.

### Rule engine (`src/fall_detection/rules.py`)

`RuleEngine.evaluate(kpts, bbox, history)` returns a score in `[0, 1]` and four boolean flags:
- **A**: height compression ratio (`h_ratio`) + ground contact points.
- **B**: ground ROI (if `ground_roi` is set) or fallback bottom-of-bbox check.
- **C**: motion-to-static transition using center displacement over a time window.
- **D**: rapid vertical descent (`vy`) combined with low height ratio.

Score = average of the four flags. If `score >= trigger_thresh`, the pipeline proceeds to the classifier to save compute.

### Classifier (`src/fall_detection/classifier.py`)

`FallClassifier` is a tiny 3-branch network (image ROI 96x96, 17 keypoints, 8-d motion) fusing into a single sigmoid output. It loads weights from `train/classifier/best.pt` if present. Motion features are extracted in the pipeline (`_extract_motion`) and include velocities, accelerations, bbox dimensions, `h_ratio`, and `n_ground`.

### Fusion and state machine (`src/fall_detection/fusion.py`)

`FusionDecision` is not a simple weighted sum; it drives a `FallState` state machine:
- `NORMAL -> SUSPECTED -> FALLING -> ALARM_SENT -> RECOVERING -> NORMAL`
- An alarm is fired once on entering `FALLING` (checked via `should_alarm()`).
- `cooldown_seconds` prevents rapid re-alarming; `recovery_seconds` prevents state oscillation after a fall ends.
- The smoothed temporal score uses a sliding window of classifier scores.

### Pose estimation (`src/fall_detection/pose_estimator.py`)

`PoseEstimator` wraps YOLOv8n-pose and returns 17 COCO keypoints per bbox. It runs full-frame inference and greedily matches returned pose boxes to the input bboxes via IoU (threshold 0.1). Unmatched bboxes receive zeroed keypoints.

### Config

All thresholds and training paths are in `configs/default.yaml`. Key sections: `detector`, `tracker`, `rules`, `fusion`, `pipeline`, `training`. Evaluation scripts often create temporary YAMLs to grid-search `trigger_thresh` and `alarm_thresh`.

## Important conventions

- **PYTHONPATH**: almost every script/test needs `PYTHONPATH=src` because the import namespace is `fall_detection`.
- **ROI input to classifier**: the pipeline resizes the track bbox to 96x96, converts BGR->RGB, transposes to `(3, 96, 96)`, and normalizes to `[0, 1]`. The classifier also accepts numpy arrays and adds batch dimensions internally.
- **Skipping frames requires caching**: when modifying `process_frame`, remember that skip frames reuse `_last_track_kpts` and `_last_cls_scores`; forgetting to update or clear these on track death can cause stale data.
- **Pose and detector coupling**: pose estimation uses a full-frame YOLOv8n-pose run and matches by IoU, rather than cropping ROIs and running pose on each crop. This keeps the pose model load minimal but means occlusion or nearby people can occasionally associate incorrectly.
