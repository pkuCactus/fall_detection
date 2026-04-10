# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- pyproject.toml for modern Python packaging
- MIT LICENSE file
- CHANGELOG.md and CONTRIBUTING.md documentation
- INSTALL.md with detailed installation guide
- Comprehensive troubleshooting documentation

### Changed
- Reorganized docs directory structure:
  - DDP troubleshooting docs moved to `docs/troubleshooting/`
  - Design documents moved to `docs/design/`
  - Development guides moved to `docs/development/`
- Renamed `data/yaml/` to `data/configs/` for clarity
- Renamed all `yolo_world` to `yoloworld` for consistency
- Consolidated dependency management:
  - Removed redundant `requirements/` directory
  - Simplified `requirements.txt` as quick install entry point
  - Updated `install.sh` to use pyproject.toml
  - Removed `generate_requirements.py`
- Added CLIP to dependencies in pyproject.toml
- Updated documentation to reflect new installation methods

### Removed
- `requirements/` directory (consolidated into pyproject.toml)
- `scripts/tools/generate_requirements.py` (no longer needed)

## [0.1.0] - 2024-04-03

### Added
- Core pipeline components:
  - YOLOv8n person detector (`src/fall_detection/core/detector.py`)
  - ByteTrack-lite tracker (`src/fall_detection/core/tracker.py`)
  - YOLOv8n-pose estimator (`src/fall_detection/core/pose_estimator.py`)
  - Rule engine for fall detection (`src/fall_detection/core/rules.py`)
  - Fusion decision with state machine (`src/fall_detection/core/fusion.py`)
- Model definitions:
  - 3-branch fusion classifier (`src/fall_detection/models/classifier.py`)
  - Simple image-only classifier (`src/fall_detection/models/simple_classifier.py`)
- Data processing:
  - Augmentation module (`src/fall_detection/data/augmentation.py`)
  - COCO and VOC dataset loaders (`src/fall_detection/data/datasets.py`)
- Training scripts:
  - Detector training (`scripts/train/train_detector.py`)
  - Pose estimator training (`scripts/train/train_pose.py`)
  - Classifier training (`scripts/train/train_classifier.py`)
  - Simple classifier training (`scripts/train/train_simple_classifier.py`)
- Evaluation scripts:
  - Pipeline evaluation (`scripts/eval/evaluate_pipeline.py`)
  - Speed benchmark (`scripts/eval/benchmark_speed.py`)
  - Tracker tuning (`scripts/eval/tune_tracker.py`)
- Demo scripts:
  - Pipeline demo (`scripts/demo/run_pipeline_demo.py`)
  - Tracker demo (`scripts/demo/demo_tracker.py`)
- Utility tools:
  - VOC to YOLO conversion (`scripts/tools/convert_voc_to_yolo.py`)
  - Feature extraction (`scripts/tools/extract_features.py`)
  - Label noise detection (`scripts/tools/find_noisy_labels.py`)
- Configuration system with YAML files
- Test suite with 90%+ coverage target (TDD approach)
- DDP (Distributed Data Parallel) training support
- Label smoothing support for classifier training
- Inference mode for datasets (fixed crop without augmentation)
- torch.compile support for PyTorch 2.0+

### Changed
- Optimized pipeline with skip-frame design for edge latency budget (~34ms)
- Reduced model sizes to meet 3516C constraints:
  - Detector: ≤3M (INT8)
  - Pose estimator: ≤2.5M (INT8)
  - Classifier: ≤0.15M
  - Total: ≤6M

### Documentation
- README.md with comprehensive usage guide
- CLAUDE.md for AI assistant integration
- API documentation for annotation tools
- DDP training troubleshooting guides

[Unreleased]: https://github.com/pkuCactus/fall_detection/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pkuCactus/fall_detection/releases/tag/v0.1.0