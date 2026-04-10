# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Style
- Write highly compact code.
- Strictly avoid unnecssary blank lines between statements.
- Max line length is 120 characters to prevent early line breaks.
- Use one-liners (comprehensions, ternary operators) where appropriate.

## Project overview

Edge-AI fall detection system targeting the HiSilicon 3516C platform (0.5T INT8, 15M DDR, 30M Flash). It is a pure vision pipeline: person detection -> tracking -> pose estimation -> rule engine -> optional light classifier -> temporal fusion with a state machine.

## Directory Structure

```
fall_detection/
├── configs/                    # Configuration files
│   ├── default.yaml           # System configuration
│   └── training/              # Training configurations
├── src/fall_detection/        # Core inference modules
│   ├── core/                  # Core components
│   │   ├── detector.py        # YOLOv8 person detector
│   │   ├── tracker.py         # ByteTrack-lite tracker
│   │   ├── pose_estimator.py  # YOLOv8-pose estimator
│   │   ├── rules.py           # Rule engine
│   │   └── fusion.py          # Fusion decision & state machine
│   ├── models/                # Model definitions
│   │   ├── classifier.py      # 3-branch fusion classifier
│   │   └── simple_classifier.py  # Simple image-only classifier
│   ├── pipeline/              # Pipeline modules
│   │   └── pipeline.py        # End-to-end pipeline
│   ├── data/                  # Data processing
│   │   ├── augmentation.py    # Data augmentation
│   │   └── datasets.py        # Dataset classes
│   └── utils/                 # Utilities
│       ├── visualization.py   # Visualization tools
│       ├── export.py          # Model export tools
│       ├── scheduler.py       # Warmup scheduler
│       └── common.py          # Common utilities
├── scripts/                   # All executable scripts
│   ├── train/                 # Training scripts
│   │   ├── train_detector.py
│   │   ├── train_pose.py
│   │   ├── train_classifier.py
│   │   ├── train_simple_classifier.py
│   │   ├── train_yoloworld.py
│   │   ├── export_yoloworld.py
│   │   ├── extract_features.py
│   │   └── validate_yoloworld.py
│   ├── eval/                  # Evaluation scripts
│   │   ├── evaluate_pipeline.py
│   │   ├── benchmark_speed.py
│   │   └── tune_tracker.py
│   ├── tools/                 # Data processing tools
│   │   ├── convert_voc_to_yolo.py
│   │   ├── extract_and_detect.py
│   │   ├── split_dataset.py
│   │   └── generate_requirements.py
│   ├── demo/                  # Demo scripts
│   │   ├── run_pipeline_demo.py
│   │   ├── demo_tracker.py
│   │   └── video_to_frames.py
│   └── shell/                 # Shell convenience wrappers
├── tests/                     # Test cases
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── data/                      # Data directory
├── models/                    # Pretrained models
│   └── pretrained/
└── outputs/                   # Training outputs
```

## Common commands

- Install dependencies: `pip install -r requirements.txt`
- Run all tests: `bash scripts/shell/run_tests.sh`
- Run a single test file: `PYTHONPATH=src pytest tests/unit/test_pipeline.py -v`
- Run pipeline demo on a video: `python scripts/demo/run_pipeline_demo.py --video data/sample.mp4 --output output.mp4`
- Benchmark speed: `python scripts/eval/benchmark_speed.py --video data/videos/test.mp4 --num-frames 100`
- End-to-end evaluation with threshold grid search: `bash scripts/shell/run_evaluate_pipeline.sh --video-dir data/videos --gt-file data/event_gt.json`

### Training stages (convenience wrappers in `scripts/shell/`):

- Detector: `bash scripts/shell/run_train_detector.sh`
- Pose estimator: `bash scripts/shell/run_train_pose.sh`
- Tracker tuning: `bash scripts/shell/run_tune_tracker.sh --video-dir data/videos`
- Feature extraction (for classifier): `bash scripts/shell/run_extract_features.sh --video-dir data/videos --label-file data/labels.json --out-dir outputs/cache`
- Fusion classifier (supports DDP): `bash scripts/shell/run_train_classifier.sh --ngpus 2 --batch-size 16`
- Simple image classifier: `bash scripts/shell/run_train_simple_classifier.sh --config configs/training/simple_classifier.yaml --ngpus 2`

Tests and most scripts require `PYTHONPATH=src` because the package root is `src/fall_detection/` and imports use `from fall_detection.x import ...`.

## Import Conventions

### Core Components
```python
from fall_detection.core import PersonDetector, ByteTrackLite, Detection, Track
from fall_detection.core import PoseEstimator, RuleEngine, FusionDecision, FallState
```

### Models
```python
from fall_detection.models import FallClassifier, SimpleFallClassifier
```

### Pipeline
```python
from fall_detection.pipeline import FallDetectionPipeline
```

### Utils
```python
from fall_detection.utils import draw_results, load_config, save_config, WarmupScheduler
from fall_detection.utils.export import export_classifier_onnx, export_simple_classifier_onnx
```

## High-level architecture

### Pipeline frame flow (`src/fall_detection/pipeline/pipeline.py`)

`FallDetectionPipeline` processes every frame but only runs the heavy models on detection frames:
- Detection is triggered every `skip_frames + 1` frames (default `skip_frames: 2`, so every 3rd frame).
- On detection frames: YOLOv8n person detector -> ByteTrack-lite tracker update -> YOLOv8n-pose per-track ROI inference -> RuleEngine -> (optional) FallClassifier -> FusionDecision.
- On skip frames: tracker predicts existing tracks without detection, rules and fusion still run using cached keypoints and cached classifier scores from the last detection frame.

This skip-frame design is central to meeting the edge latency budget (~34ms total, ≤15M DDR).

### Tracker (`src/fall_detection/core/tracker.py`)

`ByteTrackLite` is a stripped ByteTrack using only IoU + Kalman filter, no ReID. It matches detections in two stages: high-confidence detections first, then low-confidence ones. Track states are `tentative` (until `min_hits`) and `confirmed`.

### Rule engine (`src/fall_detection/core/rules.py`)

`RuleEngine.evaluate(kpts, bbox, history)` returns a score in `[0, 1]` and four boolean flags:
- **A**: height compression ratio (`h_ratio`) + ground contact points.
- **B**: ground ROI (if `ground_roi` is set) or fallback bottom-of-bbox check.
- **C**: motion-to-static transition using center displacement over a time window.
- **D**: rapid vertical descent (`vy`) combined with low height ratio.

Score = average of the four flags. If `score >= trigger_thresh`, the pipeline proceeds to the classifier to save compute.

### Classifier (`src/fall_detection/models/classifier.py`)

`FallClassifier` is a tiny 3-branch network (image ROI 96x96, 17 keypoints, 8-d motion) fusing into a single sigmoid output. It loads weights from `outputs/classifier/best.pt` if present. Motion features are extracted in the pipeline (`_extract_motion`) and include velocities, accelerations, bbox dimensions, `h_ratio`, and `n_ground`.

### Simple Classifier (`src/fall_detection/models/simple_classifier.py`)

`SimpleFallClassifier` is a lightweight single-branch image-only classifier for edge deployment. It takes 96x96 RGB images and outputs 2-class logits. Supports DDP training with data augmentation.

### Fusion and state machine (`src/fall_detection/core/fusion.py`)

`FusionDecision` is not a simple weighted sum; it drives a `FallState` state machine:
- `NORMAL -> SUSPECTED -> FALLING -> ALARM_SENT -> RECOVERING -> NORMAL`
- An alarm is fired once on entering `FALLING` (checked via `should_alarm()`).
- `cooldown_seconds` prevents rapid re-alarming; `recovery_seconds` prevents state oscillation after a fall ends.
- The smoothed temporal score uses a sliding window of classifier scores.

### Pose estimation (`src/fall_detection/core/pose_estimator.py`)

`PoseEstimator` wraps YOLOv8n-pose and returns 17 COCO keypoints per bbox. It runs full-frame inference and greedily matches returned pose boxes to the input bboxes via IoU (threshold 0.1). Unmatched bboxes receive zeroed keypoints.

## Config

All thresholds and training paths are in `configs/default.yaml`. Key sections: `detector`, `tracker`, `rules`, `fusion`, `pipeline`, `training`, `classifier`. Evaluation scripts often create temporary YAMLs to grid-search `trigger_thresh` and `alarm_thresh`.

### Training Configurations

- `configs/training/simple_classifier.yaml`: Simple image classifier training config
- `configs/training/simple_classifier_voc.yaml`: Simple classifier config for VOC format
- `configs/training/yoloworld.yaml`: YOLOWorld training config

## Important conventions

- **PYTHONPATH**: almost every script/test needs `PYTHONPATH=src` because the import namespace is `fall_detection`.
- **ROI input to classifier**: the pipeline resizes the track bbox to 96x96, converts BGR->RGB, transposes to `(3, 96, 96)`, and normalizes to `[0, 1]`. The classifier also accepts numpy arrays and adds batch dimensions internally.
- **Skipping frames requires caching**: when modifying `process_frame`, remember that skip frames reuse `_last_track_kpts` and `_last_cls_scores`; forgetting to update or clear these on track death can cause stale data.
- **Pose and detector coupling**: pose estimation uses a full-frame YOLOv8n-pose run and matches by IoU, rather than cropping ROIs and running pose on each crop. This keeps the pose model load minimal but means occlusion or nearby people can occasionally associate incorrectly.
- **Outputs directory**: training outputs go to `outputs/` (previously `train/`). Each module has its own subdirectory: `outputs/detector/`, `outputs/pose/`, `outputs/classifier/`, `outputs/simple_classifier/`, `outputs/tracker/`, `outputs/cache/`, `outputs/eval/`.

## Testing Guidelines (TDD)

### Running Tests

```bash
# Install test dependencies
pip install -r requirements/test.txt

# Run all tests with coverage
PYTHONPATH=src python -m pytest tests/ -v --cov=src/fall_detection --cov-report=term-missing

# Run specific test file
PYTHONPATH=src pytest tests/unit/test_rules.py -v

# Run with coverage threshold (90%)
PYTHONPATH=src pytest tests/ --cov=src/fall_detection --cov-fail-under=90

# Run only unit tests
PYTHONPATH=src pytest tests/unit/ -v

# Run only integration tests
PYTHONPATH=src pytest tests/integration/ -v
```

### Writing Tests

- All new code must have tests written first (TDD)
- Use `pytest` framework with fixtures
- Mock external dependencies (YOLO models, CUDA, file I/O)
- Use `tmp_path` fixture for temporary files
- Use `mocker` fixture from pytest-mock for patching

### Test Organization

```
tests/
├── unit/              # Unit tests for individual components
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_pose_estimator.py
│   ├── test_rules.py
│   ├── test_fusion.py
│   ├── test_classifier.py
│   ├── test_simple_classifier.py
│   ├── test_pipeline.py
│   ├── test_augmentation.py
│   ├── test_datasets.py
│   ├── test_utils_geometry.py
│   ├── test_utils_common.py
│   ├── test_utils_export.py
│   ├── test_utils_visualization.py
│   └── test_scheduler.py
├── integration/       # Integration tests
│   ├── test_training_detector.py
│   ├── test_training_pose.py
│   ├── test_training_classifier.py
│   ├── test_training_simple_classifier.py
│   ├── test_training_yoloworld.py
│   └── test_extract_features.py
└── conftest.py        # Shared fixtures
```

### Test Coverage Requirements

- Line coverage: minimum 90%
- Branch coverage: minimum 85%
- Critical paths (pipeline, rules, fusion): 100%

### Mocking Guidelines

```python
# Mock YOLO model
def test_detector(mocker):
    mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
    mock_model = mocker.MagicMock()
    mock_yolo.return_value = mock_model
    # ... test code

# Mock CUDA
def test_classifier_cuda(mocker):
    mocker.patch('torch.cuda.is_available', return_value=False)
    # ... test code
```

### Coverage Summary

| Module | Coverage | Test File |
|--------|----------|-----------|
| core/detector.py | 95% | test_detector.py |
| core/tracker.py | 92% | test_tracker.py |
| core/pose_estimator.py | 93% | test_pose_estimator.py |
| core/rules.py | 96% | test_rules.py |
| core/fusion.py | 94% | test_fusion.py |
| models/classifier.py | 91% | test_classifier.py |
| models/simple_classifier.py | 92% | test_simple_classifier.py |
| data/augmentation.py | 90% | test_augmentation.py |
| data/datasets.py | 88% | test_datasets.py |
| utils/* | 90% | test_utils_*.py |
