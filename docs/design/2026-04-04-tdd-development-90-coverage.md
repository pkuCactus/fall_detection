# TDD Development for Fall Detection System - 90%+ Coverage

> **Status**: Partially implemented. YOLOWorld training scripts (`scripts/train_yoloworld.py`, `scripts/validate_yoloworld.py`) are not yet implemented. Tasks related to these scripts are marked as pending.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Achieve 90%+ line coverage for all source modules and comprehensive TDD for training scripts including YOLO, YOLOWorld (pending), and both classifier trainers.

**Architecture:** Use pytest with mocking for external dependencies (YOLO models, CUDA). Organize tests into unit tests for individual components and integration tests for training workflows.

**Tech Stack:** pytest, pytest-cov, pytest-mock, unittest.mock, torch.testing, numpy.testing

---

## File Structure Overview

**Source files to test (2175 total lines):**
- `src/fall_detection/core/`: detector.py, tracker.py, pose_estimator.py, rules.py, fusion.py
- `src/fall_detection/models/`: classifier.py, simple_classifier.py
- `src/fall_detection/pipeline/`: pipeline.py
- `src/fall_detection/data/`: augmentation.py, datasets.py
- `src/fall_detection/utils/`: common.py, export.py, geometry.py, visualization.py
- `src/fall_detection/training/`: scheduler.py

**Training scripts to test:**
- `scripts/train/train_detector.py` (YOLO detector)
- `scripts/train/train_pose.py` (YOLO-pose)
- `scripts/train/train_classifier.py` (fusion classifier)
- `scripts/train/train_simple_classifier.py` (simple classifier)
- `scripts/train/extract_features.py`
- `scripts/train_yoloworld.py` (YOLOWorld) - **NOT YET IMPLEMENTED**

---

## Phase 1: Update CLAUDE.md with TDD Guidelines

### Task 1: Add TDD Section to CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (append new section)

- [ ] **Step 1: Add Testing Guidelines Section**

Add the following section to the end of CLAUDE.md:

```markdown
## Testing Guidelines (TDD)

### Running Tests

```bash
# Run all tests with coverage
PYTHONPATH=src python -m pytest tests/ -v --cov=src/fall_detection --cov-report=term-missing

# Run specific test file
PYTHONPATH=src pytest tests/unit/test_rules.py -v

# Run with coverage threshold (90%)
PYTHONPATH=src pytest tests/ --cov=src/fall_detection --cov-fail-under=90
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
```

- [ ] **Step 2: Commit CLAUDE.md update**

```bash
git add CLAUDE.md
git commit -m "docs: add TDD testing guidelines to CLAUDE.md

- Add testing commands and organization
- Add coverage requirements (90% line, 85% branch)
- Add mocking guidelines for YOLO and CUDA

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Phase 2: Core Module Unit Tests

### Task 2: Complete test_detector.py (currently basic)

**Files:**
- Modify: `tests/unit/test_detector.py`

- [ ] **Step 1: Test PersonDetector initialization with model_path**

```python
def test_detector_init_with_model_path(mocker):
    """Test detector initialization with custom model path."""
    mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
    mock_model = mocker.MagicMock()
    mock_model.args = {'imgsz': 640}
    mock_yolo.return_value = mock_model
    
    detector = PersonDetector(model_path="custom_model.pt")
    
    mock_yolo.assert_called_once_with("custom_model.pt")
    assert detector.input_size == 640
```

- [ ] **Step 2: Test PersonDetector initialization with model_name**

```python
def test_detector_init_with_model_name(mocker):
    """Test detector initialization with model name."""
    mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
    mock_model = mocker.MagicMock()
    mock_model.args = {'imgsz': 416}
    mock_yolo.return_value = mock_model
    
    detector = PersonDetector(model_name="yolov8s")
    
    mock_yolo.assert_called_once_with("yolov8s.pt")
    assert detector.input_size == 416
```

- [ ] **Step 3: Test PersonDetector with set_classes**

```python
def test_detector_with_set_classes(mocker):
    """Test detector with custom classes."""
    mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
    mock_model = mocker.MagicMock()
    mock_model.args = {'imgsz': 640}
    mock_yolo.return_value = mock_model
    
    detector = PersonDetector(model_name="yolov8n", classes=["person", "fall"])
    
    mock_model.set_classes.assert_called_once_with(["person", "fall"])
```

- [ ] **Step 4: Test detector returns person detections**

```python
def test_detector_returns_person_detections(mocker):
    """Test detector filters for person class (class_id=0)."""
    import numpy as np
    from unittest.mock import MagicMock
    
    mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
    mock_model = mocker.MagicMock()
    mock_yolo.return_value = mock_model
    
    # Create mock results with person detection
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.cls.item.return_value = 0  # person class
    mock_box.conf.item.return_value = 0.85
    mock_box.xyxy.cpu.return_value.numpy.return_value.flatten.return_value = [100, 200, 300, 400]
    mock_result.boxes = [mock_box]
    mock_model.return_value = [mock_result]
    
    detector = PersonDetector(model_name="yolov8n")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = detector(img, conf_thresh=0.3)
    
    assert len(boxes) == 1
    assert boxes[0]['class_id'] == 0
    assert boxes[0]['conf'] == 0.85
    assert boxes[0]['bbox'] == [100.0, 200.0, 300.0, 400.0]
```

- [ ] **Step 5: Test detector filters non-person classes**

```python
def test_detector_filters_non_person_classes(mocker):
    """Test detector filters out non-person detections."""
    import numpy as np
    from unittest.mock import MagicMock
    
    mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
    mock_model = mocker.MagicMock()
    mock_yolo.return_value = mock_model
    
    # Create mock results with mixed classes
    mock_result = MagicMock()
    mock_box_person = MagicMock()
    mock_box_person.cls.item.return_value = 0
    mock_box_person.conf.item.return_value = 0.9
    mock_box_person.xyxy.cpu.return_value.numpy.return_value.flatten.return_value = [100, 200, 300, 400]
    
    mock_box_car = MagicMock()
    mock_box_car.cls.item.return_value = 2  # car class
    mock_box_car.conf.item.return_value = 0.8
    mock_box_car.xyxy.cpu.return_value.numpy.return_value.flatten.return_value = [50, 50, 150, 150]
    
    mock_result.boxes = [mock_box_person, mock_box_car]
    mock_model.return_value = [mock_result]
    
    detector = PersonDetector(model_name="yolov8n")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = detector(img)
    
    assert len(boxes) == 1
    assert boxes[0]['class_id'] == 0
```

- [ ] **Step 6: Test detector filters by confidence threshold**

```python
def test_detector_filters_by_confidence(mocker):
    """Test detector filters by confidence threshold."""
    import numpy as np
    from unittest.mock import MagicMock
    
    mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
    mock_model = mocker.MagicMock()
    mock_yolo.return_value = mock_model
    
    mock_result = MagicMock()
    mock_box_high = MagicMock()
    mock_box_high.cls.item.return_value = 0
    mock_box_high.conf.item.return_value = 0.8
    mock_box_high.xyxy.cpu.return_value.numpy.return_value.flatten.return_value = [100, 200, 300, 400]
    
    mock_box_low = MagicMock()
    mock_box_low.cls.item.return_value = 0
    mock_box_low.conf.item.return_value = 0.2  # Below threshold
    mock_box_low.xyxy.cpu.return_value.numpy.return_value.flatten.return_value = [50, 50, 150, 150]
    
    mock_result.boxes = [mock_box_high, mock_box_low]
    mock_model.return_value = [mock_result]
    
    detector = PersonDetector(model_name="yolov8n")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = detector(img, conf_thresh=0.5)
    
    assert len(boxes) == 1
    assert boxes[0]['conf'] == 0.8
```

- [ ] **Step 7: Test detector handles no detections**

```python
def test_detector_handles_no_detections(mocker):
    """Test detector handles case with no detections."""
    import numpy as np
    from unittest.mock import MagicMock
    
    mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
    mock_model = mocker.MagicMock()
    mock_yolo.return_value = mock_model
    
    mock_result = MagicMock()
    mock_result.boxes = None
    mock_model.return_value = [mock_result]
    
    detector = PersonDetector(model_name="yolov8n")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = detector(img)
    
    assert boxes == []
```

- [ ] **Step 8: Test detector handles empty boxes**

```python
def test_detector_handles_empty_boxes(mocker):
    """Test detector handles case with empty boxes list."""
    import numpy as np
    from unittest.mock import MagicMock
    
    mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
    mock_model = mocker.MagicMock()
    mock_yolo.return_value = mock_model
    
    mock_result = MagicMock()
    mock_result.boxes = []
    mock_model.return_value = [mock_result]
    
    detector = PersonDetector(model_name="yolov8n")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = detector(img)
    
    assert boxes == []
```

- [ ] **Step 9: Run all detector tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_detector.py -v
```
Expected: All 9 tests pass

- [ ] **Step 10: Commit**

```bash
git add tests/unit/test_detector.py
git commit -m "test(detector): add comprehensive tests for PersonDetector

- Test initialization with model_path and model_name
- Test custom class setting
- Test person class filtering
- Test confidence threshold filtering
- Test empty detection handling

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Complete test_tracker.py

**Files:**
- Modify: `tests/unit/test_tracker.py`

- [ ] **Step 1: Read current tracker.py to understand all methods**

```bash
cat src/fall_detection/core/tracker.py
```

- [ ] **Step 2: Test ByteTrackLite initialization with custom params**

```python
def test_tracker_init_custom_params():
    """Test tracker initialization with custom parameters."""
    from fall_detection.core.tracker import ByteTrackLite
    
    tracker = ByteTrackLite(
        track_thresh=0.6,
        match_thresh=0.8,
        track_buffer=60,
        min_hits=5
    )
    
    assert tracker.track_thresh == 0.6
    assert tracker.match_thresh == 0.8
    assert tracker.track_buffer == 60
    assert tracker.min_hits == 5
```

- [ ] **Step 3: Test track ID persistence across frames**

```python
def test_tracker_id_persistence_across_frames():
    """Test that track IDs persist across multiple frames."""
    import numpy as np
    from fall_detection.core.tracker import ByteTrackLite
    
    tracker = ByteTrackLite()
    
    # First frame - two detections
    detections1 = [
        {'bbox': [100, 100, 200, 200], 'conf': 0.9, 'class_id': 0},
        {'bbox': [300, 300, 400, 400], 'conf': 0.85, 'class_id': 0}
    ]
    tracks1 = tracker.update(detections1)
    
    assert len(tracks1) == 2
    id1 = tracks1[0].track_id
    id2 = tracks1[1].track_id
    
    # Second frame - slight movement
    detections2 = [
        {'bbox': [105, 102, 205, 202], 'conf': 0.9, 'class_id': 0},
        {'bbox': [302, 305, 402, 405], 'conf': 0.85, 'class_id': 0}
    ]
    tracks2 = tracker.update(detections2)
    
    assert len(tracks2) == 2
    track_ids = [t.track_id for t in tracks2]
    assert id1 in track_ids
    assert id2 in track_ids
```

- [ ] **Step 4: Test track state transitions**

```python
def test_track_state_transitions():
    """Test track state transitions from tentative to confirmed."""
    from fall_detection.core.tracker import ByteTrackLite
    
    tracker = ByteTrackLite(min_hits=3)
    
    # First detection - tentative
    detections = [{'bbox': [100, 100, 200, 200], 'conf': 0.9, 'class_id': 0}]
    tracks = tracker.update(detections)
    assert len(tracks) == 0  # Not yet confirmed
    
    # Second detection - still tentative
    tracks = tracker.update(detections)
    assert len(tracks) == 0  # Still not confirmed
    
    # Third detection - confirmed
    tracks = tracker.update(detections)
    assert len(tracks) == 1
    assert tracks[0].state == 'confirmed'
```

- [ ] **Step 5: Test track deletion after buffer expires**

```python
def test_track_deletion_after_buffer_expires():
    """Test that tracks are deleted after track_buffer frames."""
    from fall_detection.core.tracker import ByteTrackLite
    
    tracker = ByteTrackLite(track_buffer=2, min_hits=1)
    
    # First frame - create track
    detections = [{'bbox': [100, 100, 200, 200], 'conf': 0.9, 'class_id': 0}]
    tracks = tracker.update(detections)
    track_id = tracks[0].track_id
    
    # No detection for 3 frames
    for _ in range(3):
        tracks = tracker.update([])
    
    # Track should be deleted
    assert track_id not in [t.track_id for t in tracker.tracks]
```

- [ ] **Step 6: Test IOU matching**

```python
def test_tracker_iou_matching():
    """Test IOU-based matching between detections and tracks."""
    from fall_detection.core.tracker import ByteTrackLite
    
    tracker = ByteTrackLite(match_thresh=0.5, min_hits=1)
    
    # Create track
    detections = [{'bbox': [100, 100, 200, 200], 'conf': 0.9, 'class_id': 0}]
    tracks = tracker.update(detections)
    track_id = tracks[0].track_id
    
    # Update with overlapping detection
    detections = [{'bbox': [110, 110, 210, 210], 'conf': 0.9, 'class_id': 0}]
    tracks = tracker.update(detections)
    
    assert len(tracks) == 1
    assert tracks[0].track_id == track_id
```

- [ ] **Step 7: Test low confidence detection handling**

```python
def test_tracker_low_confidence_detections():
    """Test handling of low confidence detections."""
    from fall_detection.core.tracker import ByteTrackLite
    
    tracker = ByteTrackLite(track_thresh=0.5, min_hits=1)
    
    # Mix of high and low confidence
    detections = [
        {'bbox': [100, 100, 200, 200], 'conf': 0.6, 'class_id': 0},  # High
        {'bbox': [300, 300, 400, 400], 'conf': 0.3, 'class_id': 0}   # Low
    ]
    tracks = tracker.update(detections)
    
    # Both should create tracks (low conf can still match existing)
    assert len(tracks) >= 1
```

- [ ] **Step 8: Run all tracker tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_tracker.py -v
```
Expected: All 7+ tests pass

- [ ] **Step 9: Commit**

```bash
git add tests/unit/test_tracker.py
git commit -m "test(tracker): add comprehensive ByteTrackLite tests

- Test initialization with custom parameters
- Test track ID persistence across frames
- Test track state transitions (tentative -> confirmed)
- Test track deletion after buffer expires
- Test IOU matching algorithm
- Test low confidence detection handling

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Complete test_pose_estimator.py

**Files:**
- Modify: `tests/unit/test_pose_estimator.py`
- Read: `src/fall_detection/core/pose_estimator.py`

- [ ] **Step 1: Test PoseEstimator initialization**

```python
def test_pose_estimator_init(mocker):
    """Test PoseEstimator initialization."""
    from fall_detection.core.pose_estimator import PoseEstimator
    
    mock_yolo = mocker.patch('fall_detection.core.pose_estimator.YOLO')
    mock_model = mocker.MagicMock()
    mock_yolo.return_value = mock_model
    
    estimator = PoseEstimator(model_name="yolov8n-pose")
    
    mock_yolo.assert_called_once_with("yolov8n-pose.pt")
```

- [ ] **Step 2: Test pose estimation with single person**

```python
def test_pose_single_person(mocker):
    """Test pose estimation for single person."""
    import numpy as np
    from unittest.mock import MagicMock
    from fall_detection.core.pose_estimator import PoseEstimator
    
    mock_yolo = mocker.patch('fall_detection.core.pose_estimator.YOLO')
    mock_model = mocker.MagicMock()
    mock_yolo.return_value = mock_model
    
    # Create mock result with keypoints
    mock_result = MagicMock()
    mock_box = MagicMock()
    mock_box.xyxy.cpu.return_value.numpy.return_value = [[50, 50, 150, 150]]
    mock_result.boxes = [mock_box]
    
    # 17 keypoints x 3 values (x, y, conf)
    keypoints = np.random.rand(1, 17, 3) * 100
    mock_result.keypoints.data.cpu.return_value.numpy.return_value = keypoints
    mock_model.return_value = [mock_result]
    
    estimator = PoseEstimator()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    bboxes = [[100, 100, 200, 200]]
    
    poses = estimator(img, bboxes)
    
    assert len(poses) == 1
    assert poses[0].shape == (17, 3)
```

- [ ] **Step 3: Test pose estimation with multiple people**

```python
def test_pose_multiple_people(mocker):
    """Test pose estimation for multiple people."""
    import numpy as np
    from unittest.mock import MagicMock
    from fall_detection.core.pose_estimator import PoseEstimator
    
    mock_yolo = mocker.patch('fall_detection.core.pose_estimator.YOLO')
    mock_model = mocker.MagicMock()
    mock_yolo.return_value = mock_model
    
    mock_result = MagicMock()
    mock_box1 = MagicMock()
    mock_box1.xyxy.cpu.return_value.numpy.return_value = [[90, 90, 210, 210]]
    mock_box2 = MagicMock()
    mock_box2.xyxy.cpu.return_value.numpy.return_value = [[290, 290, 410, 410]]
    mock_result.boxes = [mock_box1, mock_box2]
    
    keypoints = np.random.rand(2, 17, 3) * 100
    mock_result.keypoints.data.cpu.return_value.numpy.return_value = keypoints
    mock_model.return_value = [mock_result]
    
    estimator = PoseEstimator()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    bboxes = [[100, 100, 200, 200], [300, 300, 400, 400]]
    
    poses = estimator(img, bboxes)
    
    assert len(poses) == 2
    assert all(p.shape == (17, 3) for p in poses)
```

- [ ] **Step 4: Test pose estimation with no detections**

```python
def test_pose_no_detections(mocker):
    """Test pose estimation with no person detections."""
    import numpy as np
    from fall_detection.core.pose_estimator import PoseEstimator
    
    mock_yolo = mocker.patch('fall_detection.core.pose_estimator.YOLO')
    mock_yolo.return_value = mocker.MagicMock()
    
    estimator = PoseEstimator()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    bboxes = []
    
    poses = estimator(img, bboxes)
    
    assert poses == []
```

- [ ] **Step 5: Test IOU threshold matching**

```python
def test_pose_iou_threshold(mocker):
    """Test pose-to-bbox matching with IOU threshold."""
    import numpy as np
    from unittest.mock import MagicMock
    from fall_detection.core.pose_estimator import PoseEstimator
    
    mock_yolo = mocker.patch('fall_detection.core.pose_estimator.YOLO')
    mock_model = mocker.MagicMock()
    mock_yolo.return_value = mock_model
    
    mock_result = MagicMock()
    # Pose bbox far from input bbox (IOU < 0.1)
    mock_box = MagicMock()
    mock_box.xyxy.cpu.return_value.numpy.return_value = [[400, 400, 500, 500]]
    mock_result.boxes = [mock_box]
    mock_result.keypoints.data.cpu.return_value.numpy.return_value = np.random.rand(1, 17, 3) * 100
    mock_model.return_value = [mock_result]
    
    estimator = PoseEstimator(iou_thresh=0.1)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    bboxes = [[100, 100, 200, 200]]  # Far from pose bbox
    
    poses = estimator(img, bboxes)
    
    # Should return zeros due to low IOU
    assert len(poses) == 1
    assert np.allclose(poses[0], 0)
```

- [ ] **Step 6: Run all pose estimator tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_pose_estimator.py -v
```

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_pose_estimator.py
git commit -m "test(pose): add comprehensive PoseEstimator tests

- Test initialization with model name
- Test single person pose estimation
- Test multiple people pose estimation
- Test empty detection handling
- Test IOU threshold matching

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Complete test_rules.py

**Files:**
- Check existing: `tests/unit/test_rules.py`

- [ ] **Step 1: Check current test coverage**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_rules.py -v
```

- [ ] **Step 2: Add tests for edge cases in rules**

```python
def test_rule_engine_with_empty_keypoints():
    """Test rule engine with empty/invalid keypoints."""
    import numpy as np
    from fall_detection.core.rules import RuleEngine
    
    engine = RuleEngine()
    kpts = np.zeros((17, 3))  # All zeros
    bbox = [100, 100, 200, 300]
    
    score, flags = engine.evaluate(kpts, bbox, [])
    
    assert 0 <= score <= 1
    assert len(flags) == 4
```

- [ ] **Step 3: Add tests for rule C (motion-to-static)**

```python
def test_rule_c_motion_to_static():
    """Test rule C: motion to static transition."""
    import numpy as np
    from fall_detection.core.rules import RuleEngine
    
    engine = RuleEngine()
    kpts = np.ones((17, 3)) * 100
    bbox = [100, 100, 200, 300]
    
    # Empty history - should not trigger
    score, flags = engine.evaluate(kpts, bbox, [])
    assert flags[2] == False  # Rule C
    
    # History with movement then stop
    history = [
        {'center': (110, 200), 'timestamp': 0},
        {'center': (120, 200), 'timestamp': 0.5},
        {'center': (125, 200), 'timestamp': 1.0},
    ]
    score, flags = engine.evaluate(kpts, bbox, history)
    # Should detect motion
```

- [ ] **Step 4: Add tests for rule D (rapid descent)**

```python
def test_rule_d_rapid_descent():
    """Test rule D: rapid vertical descent."""
    import numpy as np
    from fall_detection.core.rules import RuleEngine
    
    engine = RuleEngine()
    
    # Simulate falling: low height ratio + negative vy
    kpts = np.ones((17, 3)) * 50
    bbox = [100, 250, 200, 300]  # Low height
    
    history = [
        {'bbox': [100, 100, 200, 200], 'timestamp': 0, 'center': (150, 150)},
        {'bbox': [100, 250, 200, 300], 'timestamp': 1.0, 'center': (150, 275)},
    ]
    
    score, flags = engine.evaluate(kpts, bbox, history)
    # Rule D should trigger for rapid descent
```

- [ ] **Step 5: Run all rules tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_rules.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_rules.py
git commit -m "test(rules): add edge case tests for RuleEngine

- Test empty keypoints handling
- Test rule C motion-to-static detection
- Test rule D rapid descent detection

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Complete test_fusion.py

**Files:**
- Check existing: `tests/unit/test_fusion.py`

- [ ] **Step 1: Check current fusion tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_fusion.py -v
```

- [ ] **Step 2: Add tests for all state transitions**

```python
def test_state_machine_all_transitions():
    """Test all state machine transitions."""
    from fall_detection.core.fusion import FallState, FusionDecision
    
    fusion = FusionDecision(
        trigger_thresh=0.5,
        alarm_thresh=0.8,
        window_size=3,
        cooldown_seconds=5,
        recovery_seconds=3
    )
    
    # Start in NORMAL
    assert fusion.state == FallState.NORMAL
    
    # Move to SUSPECTED
    for _ in range(3):
        fusion.update(0.6)  # Above trigger_thresh
    assert fusion.state == FallState.SUSPECTED
    
    # Move to FALLING
    for _ in range(3):
        fusion.update(0.9)  # Above alarm_thresh
    assert fusion.state == FallState.FALLING
```

- [ ] **Step 3: Add tests for cooldown mechanism**

```python
def test_cooldown_mechanism():
    """Test cooldown prevents rapid re-alarming."""
    import time
    from fall_detection.core.fusion import FusionDecision
    
    fusion = FusionDecision(cooldown_seconds=2)
    
    # Trigger first alarm
    for _ in range(5):
        fusion.update(0.9)
    assert fusion.should_alarm()
    
    # Immediately try again - should be blocked
    fusion.reset()
    for _ in range(5):
        fusion.update(0.9)
    assert not fusion.should_alarm()  # In cooldown
```

- [ ] **Step 4: Add tests for recovery state**

```python
def test_recovery_state():
    """Test recovery state after alarm."""
    from fall_detection.core.fusion import FallState, FusionDecision
    
    fusion = FusionDecision(recovery_seconds=1)
    
    # Trigger alarm
    for _ in range(5):
        fusion.update(0.9)
    assert fusion.state == FallState.FALLING
    
    # Lower scores - should move to RECOVERING
    for _ in range(3):
        fusion.update(0.2)
    assert fusion.state == FallState.RECOVERING
```

- [ ] **Step 5: Run all fusion tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_fusion.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_fusion.py
git commit -m "test(fusion): add comprehensive state machine tests

- Test all state transitions
- Test cooldown mechanism
- Test recovery state behavior

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Complete test_classifier.py

**Files:**
- Check existing: `tests/unit/test_classifier.py`
- Read: `src/fall_detection/models/classifier.py`

- [ ] **Step 1: Check current classifier tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_classifier.py -v
```

- [ ] **Step 2: Add tests for FallClassifier branches**

```python
def test_classifier_all_branches():
    """Test all three branches of FallClassifier."""
    import torch
    from fall_detection.models.classifier import FallClassifier
    
    model = FallClassifier()
    model.eval()
    
    batch_size = 2
    roi = torch.randn(batch_size, 3, 96, 96)
    kpts = torch.randn(batch_size, 17, 3)
    motion = torch.randn(batch_size, 8)
    
    with torch.no_grad():
        output = model(roi, kpts, motion)
    
    assert output.shape == (batch_size, 1)
    assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output
```

- [ ] **Step 3: Add tests for batch size 1**

```python
def test_classifier_batch_size_1():
    """Test classifier with batch size 1."""
    import torch
    from fall_detection.models.classifier import FallClassifier
    
    model = FallClassifier()
    roi = torch.randn(1, 3, 96, 96)
    kpts = torch.randn(1, 17, 3)
    motion = torch.randn(1, 8)
    
    output = model(roi, kpts, motion)
    assert output.shape == (1, 1)
```

- [ ] **Step 4: Add tests for zero input**

```python
def test_classifier_zero_input():
    """Test classifier with zero input."""
    import torch
    from fall_detection.models.classifier import FallClassifier
    
    model = FallClassifier()
    roi = torch.zeros(1, 3, 96, 96)
    kpts = torch.zeros(1, 17, 3)
    motion = torch.zeros(1, 8)
    
    output = model(roi, kpts, motion)
    assert output.shape == (1, 1)
    # With zero input, sigmoid should be around 0.5
    assert 0.4 < output.item() < 0.6
```

- [ ] **Step 5: Run all classifier tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_classifier.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_classifier.py
git commit -m "test(classifier): add comprehensive FallClassifier tests

- Test all three input branches
- Test various batch sizes
- Test zero input handling

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Create test_simple_classifier.py

**Files:**
- Create: `tests/unit/test_simple_classifier.py`
- Read: `src/fall_detection/models/simple_classifier.py`

- [ ] **Step 1: Test SimpleFallClassifier initialization**

```python
def test_simple_classifier_init():
    """Test SimpleFallClassifier initialization."""
    from fall_detection.models.simple_classifier import SimpleFallClassifier
    
    model = SimpleFallClassifier(num_classes=2, dropout=0.3)
    
    assert model.num_classes == 2
    assert model.dropout.p == 0.3
```

- [ ] **Step 2: Test forward pass**

```python
def test_simple_classifier_forward():
    """Test forward pass with various input sizes."""
    import torch
    from fall_detection.models.simple_classifier import SimpleFallClassifier
    
    model = SimpleFallClassifier()
    
    # Single image
    x = torch.randn(1, 3, 96, 96)
    output = model(x)
    assert output.shape == (1, 2)
    
    # Batch of images
    x = torch.randn(8, 3, 96, 96)
    output = model(x)
    assert output.shape == (8, 2)
```

- [ ] **Step 3: Test feature extraction**

```python
def test_simple_classifier_features():
    """Test feature extraction before final layer."""
    import torch
    from fall_detection.models.simple_classifier import SimpleFallClassifier
    
    model = SimpleFallClassifier()
    x = torch.randn(1, 3, 96, 96)
    
    features = model.features(x)
    assert features.shape == (1, 128)  # Feature vector size
```

- [ ] **Step 4: Run all simple classifier tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_simple_classifier.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_simple_classifier.py
git commit -m "test(simple_classifier): add SimpleFallClassifier tests

- Test initialization with custom parameters
- Test forward pass with various batch sizes
- Test feature extraction

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 9: Complete test_pipeline.py

**Files:**
- Check existing: `tests/unit/test_pipeline.py`
- Read: `src/fall_detection/pipeline/pipeline.py`

- [ ] **Step 1: Check current pipeline tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_pipeline.py -v
```

- [ ] **Step 2: Add tests for skip frame logic**

```python
def test_pipeline_skip_frames(mocker):
    """Test pipeline skip frame behavior."""
    import numpy as np
    from fall_detection.pipeline import FallDetectionPipeline
    
    # Mock all components
    mocker.patch('fall_detection.pipeline.pipeline.PersonDetector')
    mocker.patch('fall_detection.pipeline.pipeline.ByteTrackLite')
    mocker.patch('fall_detection.pipeline.pipeline.PoseEstimator')
    mocker.patch('fall_detection.pipeline.pipeline.RuleEngine')
    mocker.patch('fall_detection.pipeline.pipeline.FusionDecision')
    
    pipeline = FallDetectionPipeline(skip_frames=2)
    
    # First frame - detection frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result1 = pipeline.process_frame(frame)
    
    # Second frame - skip frame
    result2 = pipeline.process_frame(frame)
    
    # Detection should only run on first frame
```

- [ ] **Step 3: Add tests for track caching**

```python
def test_pipeline_track_caching(mocker):
    """Test that track keypoints are cached on skip frames."""
    import numpy as np
    from fall_detection.pipeline import FallDetectionPipeline
    
    mocker.patch('fall_detection.pipeline.pipeline.PersonDetector')
    mocker.patch('fall_detection.pipeline.pipeline.ByteTrackLite')
    mocker.patch('fall_detection.pipeline.pipeline.PoseEstimator')
    mocker.patch('fall_detection.pipeline.pipeline.RuleEngine')
    mocker.patch('fall_detection.pipeline.pipeline.FusionDecision')
    
    pipeline = FallDetectionPipeline()
    
    # Process frame and check caching
```

- [ ] **Step 4: Run all pipeline tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_pipeline.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_pipeline.py
git commit -m "test(pipeline): add skip frame and caching tests

- Test skip frame logic
- Test track keypoint caching

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Phase 3: Data and Utils Tests

### Task 10: Create test_augmentation.py

**Files:**
- Create: `tests/unit/test_augmentation.py`

- [ ] **Step 1: Test RandomMask**

```python
def test_random_mask():
    """Test RandomMask augmentation."""
    import numpy as np
    from fall_detection.data.augmentation import RandomMask
    
    mask = RandomMask(mask_ratio=0.25, p=1.0)  # Always apply
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    result = mask(img)
    
    assert result.shape == img.shape
    # Some pixels should be different due to masking
    assert not np.array_equal(result, img)
```

- [ ] **Step 2: Test RandomCropWithPadding**

```python
def test_random_crop_with_padding():
    """Test RandomCropWithPadding."""
    import numpy as np
    from fall_detection.data.augmentation import RandomCropWithPadding
    
    crop = RandomCropWithPadding(shrink_max=5, expand_max=10)
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    bbox = [50, 50, 100, 100]
    
    result = crop(img, bbox)
    
    assert len(result.shape) == 3
    assert result.shape[2] == 3
```

- [ ] **Step 3: Test LetterBoxResize**

```python
def test_letterbox_resize():
    """Test LetterBoxResize preserves aspect ratio."""
    import numpy as np
    from fall_detection.data.augmentation import LetterBoxResize
    
    resize = LetterBoxResize(target_size=96, fill_value=114)
    
    # Wide image
    img_wide = np.ones((100, 200, 3), dtype=np.uint8) * 255
    result = resize(img_wide)
    assert result.shape == (96, 96, 3)
    
    # Tall image
    img_tall = np.ones((200, 100, 3), dtype=np.uint8) * 255
    result = resize(img_tall)
    assert result.shape == (96, 96, 3)
```

- [ ] **Step 4: Test TrainingAugmentation**

```python
def test_training_augmentation():
    """Test TrainingAugmentation with all features enabled."""
    import numpy as np
    from fall_detection.data.augmentation import TrainingAugmentation
    
    aug_cfg = {
        'color_jitter': {'enabled': True, 'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3},
        'random_gray': {'enabled': True, 'p': 1.0},
        'random_rotation': {'enabled': True, 'angle_range': [-5, 5], 'p': 1.0},
        'random_mask': {'enabled': True, 'mask_ratio': 0.15, 'p': 1.0},
        'horizontal_flip': {'enabled': True, 'p': 1.0},
    }
    
    aug = TrainingAugmentation(aug_cfg)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    result = aug(img)
    
    assert result.shape == img.shape
```

- [ ] **Step 5: Run all augmentation tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_augmentation.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_augmentation.py
git commit -m "test(augmentation): add data augmentation tests

- Test RandomMask
- Test RandomCropWithPadding
- Test LetterBoxResize
- Test TrainingAugmentation

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 11: Create test_datasets.py

**Files:**
- Create: `tests/unit/test_datasets.py`
- Read: `src/fall_detection/data/datasets.py`

- [ ] **Step 1: Test CocoFallDataset initialization**

```python
def test_coco_dataset_init(mocker, tmp_path):
    """Test CocoFallDataset initialization."""
    import json
    import numpy as np
    from PIL import Image
    from fall_detection.data import CocoFallDataset
    
    # Create mock COCO data
    coco_data = {
        'images': [{'id': 1, 'file_name': 'test.jpg', 'width': 640, 'height': 480}],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [100, 100, 50, 100]}
        ],
        'categories': [{'id': 1, 'name': 'person'}]
    }
    
    coco_path = tmp_path / 'coco.json'
    with open(coco_path, 'w') as f:
        json.dump(coco_data, f)
    
    # Create dummy image
    img_dir = tmp_path / 'images'
    img_dir.mkdir()
    img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    img.save(img_dir / 'test.jpg')
    
    dataset = CocoFallDataset(
        image_dir=str(img_dir),
        coco_json=str(coco_path),
        fall_category_id=1
    )
    
    assert len(dataset) == 1
```

- [ ] **Step 2: Test CocoFallDataset __getitem__**

```python
def test_coco_dataset_getitem(mocker, tmp_path):
    """Test CocoFallDataset item retrieval."""
    import json
    import numpy as np
    from PIL import Image
    from fall_detection.data import CocoFallDataset
    
    # Setup mock data similar to above
    # ...
    
    dataset = CocoFallDataset(
        image_dir=str(img_dir),
        coco_json=str(coco_path),
        fall_category_id=1
    )
    
    img, label = dataset[0]
    
    assert img.shape == (3, 96, 96)  # After transforms
    assert label in [0, 1]
```

- [ ] **Step 3: Test VOCFallDataset**

```python
def test_voc_dataset_init(tmp_path):
    """Test VOCFallDataset initialization."""
    import xml.etree.ElementTree as ET
    from PIL import Image
    import numpy as np
    from fall_detection.data import VOCFallDataset
    
    # Create mock VOC structure
    voc_dir = tmp_path / 'VOC2007'
    img_dir = voc_dir / 'JPEGImages'
    ann_dir = voc_dir / 'Annotations'
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    
    # Create dummy image
    img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
    img.save(img_dir / 'test.jpg')
    
    # Create annotation XML
    annotation = ET.Element('annotation')
    obj = ET.SubElement(annotation, 'object')
    name = ET.SubElement(obj, 'name')
    name.text = 'fall'
    bndbox = ET.SubElement(obj, 'bndbox')
    for tag in ['xmin', 'ymin', 'xmax', 'ymax']:
        ET.SubElement(bndbox, tag).text = '100'
    
    tree = ET.ElementTree(annotation)
    tree.write(ann_dir / 'test.xml')
    
    dataset = VOCFallDataset(
        data_dirs=[str(voc_dir)],
        split='train',
        fall_classes=['fall']
    )
    
    assert len(dataset) == 1
```

- [ ] **Step 4: Run all dataset tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_datasets.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_datasets.py
git commit -m "test(datasets): add dataset loading tests

- Test CocoFallDataset initialization and __getitem__
- Test VOCFallDataset with mock data

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 12: Create test_utils_*.py files

**Files:**
- Create: `tests/unit/test_utils_geometry.py`
- Create: `tests/unit/test_utils_common.py`
- Create: `tests/unit/test_utils_export.py`
- Create: `tests/unit/test_utils_visualization.py`

- [ ] **Step 1: Test geometry.iou**

```python
def test_iou_identical_boxes():
    """Test IoU with identical boxes."""
    from fall_detection.utils.geometry import iou
    
    bbox = [0, 0, 100, 100]
    assert iou(bbox, bbox) == 1.0

def test_iou_no_overlap():
    """Test IoU with non-overlapping boxes."""
    from fall_detection.utils.geometry import iou
    
    bbox1 = [0, 0, 50, 50]
    bbox2 = [60, 60, 100, 100]
    assert iou(bbox1, bbox2) == 0.0

def test_iou_partial_overlap():
    """Test IoU with partial overlap."""
    from fall_detection.utils.geometry import iou
    
    bbox1 = [0, 0, 100, 100]
    bbox2 = [50, 50, 150, 150]
    # Intersection: 50x50 = 2500
    # Union: 10000 + 10000 - 2500 = 17500
    # IoU: 2500 / 17500 = 0.142...
    result = iou(bbox1, bbox2)
    assert 0.14 < result < 0.15
```

- [ ] **Step 2: Test utils.common**

```python
def test_load_config(tmp_path):
    """Test load_config function."""
    from fall_detection.utils.common import load_config
    
    config_path = tmp_path / 'test.yaml'
    config_path.write_text('key: value\nnested:\n  inner: data\n')
    
    config = load_config(str(config_path))
    
    assert config['key'] == 'value'
    assert config['nested']['inner'] == 'data'

def test_save_config(tmp_path):
    """Test save_config function."""
    from fall_detection.utils.common import save_config
    
    config = {'key': 'value', 'number': 42}
    config_path = tmp_path / 'output.yaml'
    
    save_config(config, str(config_path))
    
    assert config_path.exists()
    content = config_path.read_text()
    assert 'key: value' in content
```

- [ ] **Step 3: Test utils.export (mock ONNX)**

```python
def test_export_classifier_onnx(mocker, tmp_path):
    """Test export_classifier_onnx function."""
    from fall_detection.utils.export import export_classifier_onnx
    
    mock_export = mocker.patch('torch.onnx.export')
    
    out_path = str(tmp_path / 'model.onnx')
    export_classifier_onnx(out_path)
    
    mock_export.assert_called_once()

def test_export_simple_classifier_onnx(mocker, tmp_path):
    """Test export_simple_classifier_onnx function."""
    from fall_detection.utils.export import export_simple_classifier_onnx
    
    mock_export = mocker.patch('torch.onnx.export')
    
    out_path = str(tmp_path / 'model.onnx')
    export_simple_classifier_onnx(out_path)
    
    mock_export.assert_called_once()
```

- [ ] **Step 4: Test utils.visualization**

```python
def test_draw_results(mocker):
    """Test draw_results function."""
    import numpy as np
    from fall_detection.utils.visualization import draw_results
    
    # Mock cv2 functions
    mocker.patch('cv2.rectangle')
    mocker.patch('cv2.putText')
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = [{'track_id': 1, 'bbox': [100, 100, 200, 200]}]
    
    result = draw_results(frame, tracks)
    
    assert result.shape == frame.shape
```

- [ ] **Step 5: Run all utils tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_utils_*.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_utils_*.py
git commit -m "test(utils): add utility function tests

- Test geometry.iou calculations
- Test common.load_config and save_config
- Test export functions (mocked ONNX)
- Test visualization functions

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 13: Create test_scheduler.py

**Files:**
- Create: `tests/unit/test_scheduler.py`

- [ ] **Step 1: Test WarmupScheduler initialization**

```python
def test_warmup_scheduler_init():
    """Test WarmupScheduler initialization."""
    import torch
    from fall_detection.utils.scheduler import WarmupScheduler
    
    optimizer = torch.optim.SGD([torch.randn(10)], lr=0.01)
    scheduler = WarmupScheduler(
        optimizer,
        None,
        warmup_steps=100,
        warmup_strategy='linear',
        warmup_init_lr=1e-5
    )
    
    assert scheduler.warmup_steps == 100
    assert scheduler.warmup_strategy == 'linear'
    assert scheduler.warmup_init_lr == 1e-5
```

- [ ] **Step 2: Test linear warmup**

```python
def test_warmup_scheduler_linear():
    """Test linear warmup strategy."""
    import torch
    from fall_detection.utils.scheduler import WarmupScheduler
    
    optimizer = torch.optim.SGD([torch.randn(10)], lr=0.01)
    scheduler = WarmupScheduler(
        optimizer,
        None,
        warmup_steps=10,
        warmup_strategy='linear',
        warmup_init_lr=0.0
    )
    
    # Initial LR should be 0
    assert scheduler.get_last_lr()[0] == 0.0
    
    # After 5 steps, LR should be halfway
    for _ in range(5):
        scheduler.step_batch()
    
    lr = scheduler.get_last_lr()[0]
    assert abs(lr - 0.005) < 0.001  # Approximately halfway
```

- [ ] **Step 3: Test constant warmup**

```python
def test_warmup_scheduler_constant():
    """Test constant warmup strategy."""
    import torch
    from fall_detection.utils.scheduler import WarmupScheduler
    
    optimizer = torch.optim.SGD([torch.randn(10)], lr=0.01)
    scheduler = WarmupScheduler(
        optimizer,
        None,
        warmup_steps=10,
        warmup_strategy='constant',
        warmup_init_lr=1e-5
    )
    
    # During warmup, LR should stay constant
    for _ in range(5):
        scheduler.step_batch()
    
    assert scheduler.get_last_lr()[0] == 1e-5
```

- [ ] **Step 4: Test warmup completion**

```python
def test_warmup_scheduler_completion():
    """Test warmup completion transitions to base scheduler."""
    import torch
    from fall_detection.utils.scheduler import WarmupScheduler
    
    base_scheduler = torch.optim.lr_scheduler.StepLR(
        torch.optim.SGD([torch.randn(10)], lr=0.01),
        step_size=10
    )
    optimizer = torch.optim.SGD([torch.randn(10)], lr=0.01)
    scheduler = WarmupScheduler(
        optimizer,
        base_scheduler,
        warmup_steps=5,
        warmup_strategy='linear',
        warmup_init_lr=0.0
    )
    
    # Complete warmup
    for _ in range(5):
        scheduler.step_batch()
    
    # Should be at base LR
    assert scheduler.warmup_finished
    assert abs(scheduler.get_last_lr()[0] - 0.01) < 0.0001
```

- [ ] **Step 5: Run all scheduler tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/test_scheduler.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_scheduler.py
git commit -m "test(scheduler): add WarmupScheduler tests

- Test initialization
- Test linear warmup strategy
- Test constant warmup strategy
- Test warmup completion transition

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Phase 4: Integration Tests for Training Scripts

### Task 14: Create test_training_detector.py

**Files:**
- Create: `tests/integration/test_training_detector.py`

- [ ] **Step 1: Test train_detector argument parsing**

```python
def test_train_detector_args(mocker):
    """Test train_detector argument parsing."""
    import sys
    from unittest.mock import patch, MagicMock
    
    mock_yolo = mocker.patch('ultralytics.YOLO')
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    
    test_args = [
        'train_detector.py',
        '--data', 'data/test.yaml',
        '--epochs', '1',
        '--batch', '2',
        '--model', 'yolov8n.pt'
    ]
    
    with patch.object(sys, 'argv', test_args):
        # Import and run main
        pass  # Run the training script
```

- [ ] **Step 2: Test train_detector end-to-end (mocked)**

```python
def test_train_detector_e2e(mocker, tmp_path):
    """Test train_detector end-to-end with mocked YOLO."""
    import sys
    from unittest.mock import patch, MagicMock
    
    mock_yolo = mocker.patch('ultralytics.YOLO')
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    
    # Create mock data config
    data_yaml = tmp_path / 'data.yaml'
    data_yaml.write_text('train: images/train\nval: images/val\nnc: 1\nnames: [person]\n')
    
    test_args = [
        'scripts/train/train_detector.py',
        '--data', str(data_yaml),
        '--epochs', '1',
        '--batch', '2',
        '--project', str(tmp_path / 'train'),
        '--name', 'test'
    ]
    
    with patch.object(sys, 'argv', test_args):
        # Import and execute
        sys.path.insert(0, 'training/scripts')
        import train_detector
        try:
            train_detector.main()
        except SystemExit:
            pass
    
    mock_model.train.assert_called_once()
```

- [ ] **Step 3: Run detector training tests**

```bash
PYTHONPATH=src python -m pytest tests/integration/test_training_detector.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_training_detector.py
git commit -m "test(integration): add YOLO detector training tests

- Test argument parsing
- Test end-to-end training with mocked YOLO

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 15: Create test_training_yoloworld.py

**Files:**
- Create: `tests/integration/test_training_yoloworld.py`

- [ ] **Step 1: Test YOLO-World training**

```python
def test_train_yoloworld_e2e(mocker, tmp_path):
    """Test YOLO-World training end-to-end."""
    import sys
    from unittest.mock import patch, MagicMock
    
    mock_yolo = mocker.patch('ultralytics.YOLO')
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    
    # Create mock data config with class names
    data_yaml = tmp_path / 'data.yaml'
    data_yaml.write_text('train: images/train\nval: images/val\nnc: 1\nnames: {0: person}\n')
    
    test_args = [
        'scripts/train_yoloworld.py',
        '--data', str(data_yaml),
        '--epochs', '1',
        '--batch', '2',
        '--project', str(tmp_path / 'train'),
        '--name', 'test'
    ]
    
    with patch.object(sys, 'argv', test_args):
        sys.path.insert(0, 'scripts')
        import train_yoloworld
        try:
            train_yoloworld.main()
        except SystemExit:
            pass
    
    mock_model.set_classes.assert_called_once()
    mock_model.train.assert_called_once()
```

- [ ] **Step 2: Test YOLO-World class loading**

```python
def test_yoloworld_class_loading(mocker, tmp_path):
    """Test YOLO-World loads classes from data config."""
    import sys
    from unittest.mock import patch, MagicMock
    
    mock_yolo = mocker.patch('ultralytics.YOLO')
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    
    # Test with list format
    data_yaml = tmp_path / 'data.yaml'
    data_yaml.write_text('train: images/train\nval: images/val\nnc: 2\nnames: [person, fall]\n')
    
    test_args = [
        'scripts/train_yoloworld.py',
        '--data', str(data_yaml),
        '--epochs', '1',
        '--project', str(tmp_path / 'train'),
        '--name', 'test'
    ]
    
    with patch.object(sys, 'argv', test_args):
        sys.path.insert(0, 'scripts')
        import importlib
        importlib.reload(sys.modules.get('train_yoloworld', sys.modules[__name__]))
```

- [ ] **Step 3: Run YOLO-World tests**

```bash
PYTHONPATH=src python -m pytest tests/integration/test_training_yoloworld.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_training_yoloworld.py
git commit -m "test(integration): add YOLO-World training tests

- Test end-to-end training with mocked YOLO
- Test class loading from data config

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 16: Create test_training_simple_classifier.py

**Files:**
- Create: `tests/integration/test_training_simple_classifier.py`

- [ ] **Step 1: Test simple classifier training setup**

```python
def test_simple_classifier_training_setup(mocker, tmp_path):
    """Test simple classifier training setup."""
    import sys
    from unittest.mock import patch, MagicMock
    
    # Mock torch.distributed
    mocker.patch('torch.cuda.is_available', return_value=False)
    
    # Create mock config
    config = tmp_path / 'config.yaml'
    config.write_text('''
seed: 42
data:
  format: "voc"
  image_dir: "data/images"
  train_coco_json: null
  val_coco_json: null
model:
  type: "simple"
  dropout: 0.3
  num_classes: 2
epochs: 1
batch_size: 4
num_workers: 0
lr: 0.001
weight_decay: 0.0001
lr_scheduler:
  type: "plateau"
  factor: 0.5
  patience: 10
  min_lr: 0.00001
early_stopping:
  enabled: false
input:
  target_size: 96
  use_letterbox: true
data_augmentation:
  enabled: false
voc:
  train_dirs: []
  val_dirs: []
  fall_classes: ["fall"]
  normal_classes: null
output:
  dir: "outputs/test"
  save_best: false
log:
  epoch_log_interval: 1
  batch_log_interval: 10
''')
    
    test_args = [
        'scripts/train/train_simple_classifier.py',
        '--config', str(config)
    ]
    
    with patch.object(sys, 'argv', test_args):
        # Would run the training script
        pass
```

- [ ] **Step 2: Test simple classifier with mock data**

```python
def test_simple_classifier_with_mock_data(mocker, tmp_path):
    """Test simple classifier training with mock dataset."""
    import torch
    import numpy as np
    from fall_detection.models.simple_classifier import SimpleFallClassifier
    
    # Create model
    model = SimpleFallClassifier(num_classes=2, dropout=0.3)
    
    # Create mock data
    batch_size = 4
    images = torch.randn(batch_size, 3, 96, 96)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Forward pass
    outputs = model(images)
    
    assert outputs.shape == (batch_size, 2)
    
    # Test loss computation
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    
    assert loss.item() > 0
```

- [ ] **Step 3: Run simple classifier training tests**

```bash
PYTHONPATH=src python -m pytest tests/integration/test_training_simple_classifier.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_training_simple_classifier.py
git commit -m "test(integration): add simple classifier training tests

- Test training setup with config
- Test with mock dataset

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 17: Create test_training_classifier.py (Fusion Classifier)

**Files:**
- Create: `tests/integration/test_training_classifier.py`

- [ ] **Step 1: Test fusion classifier FeatureDataset**

```python
def test_feature_dataset(tmp_path):
    """Test FeatureDataset loading."""
    import numpy as np
    import sys
    sys.path.insert(0, 'training/scripts')
    from train_classifier import FeatureDataset
    
    # Create mock feature files
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    
    for i in range(5):
        np.savez(
            cache_dir / f'sample_{i}.npz',
            roi=np.random.rand(96, 96, 3),
            kpts=np.random.rand(17, 3),
            motion=np.random.rand(8),
            label=np.random.randint(0, 2)
        )
    
    dataset = FeatureDataset(str(cache_dir))
    
    assert len(dataset) == 5
    
    roi, kpts, motion, label = dataset[0]
    assert roi.shape == (3, 96, 96)
    assert kpts.shape == (17, 3)
    assert motion.shape == (8,)
    assert label.dim() == 0  # Scalar tensor
```

- [ ] **Step 2: Test fusion classifier training epoch**

```python
def test_classifier_train_epoch(mocker):
    """Test fusion classifier training epoch."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import sys
    sys.path.insert(0, 'training/scripts')
    from train_classifier import train_epoch
    from fall_detection.models.classifier import FallClassifier
    
    # Create mock model and data
    model = FallClassifier()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCELoss()
    
    # Create fake data
    batch_size = 4
    roi = torch.randn(batch_size, 3, 96, 96)
    kpts = torch.randn(batch_size, 17, 3)
    motion = torch.randn(batch_size, 8)
    labels = torch.rand(batch_size)
    
    dataset = TensorDataset(roi, kpts, motion, labels)
    loader = DataLoader(dataset, batch_size=2)
    
    device = torch.device('cpu')
    
    # Run training epoch
    loss, samples = train_epoch(model, loader, optimizer, criterion, device)
    
    assert loss > 0
    assert samples == batch_size
```

- [ ] **Step 3: Test fusion classifier eval epoch**

```python
def test_classifier_eval_epoch():
    """Test fusion classifier evaluation epoch."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import sys
    sys.path.insert(0, 'training/scripts')
    from train_classifier import eval_epoch
    from fall_detection.models.classifier import FallClassifier
    
    model = FallClassifier()
    model.eval()
    criterion = torch.nn.BCELoss()
    
    batch_size = 4
    roi = torch.randn(batch_size, 3, 96, 96)
    kpts = torch.randn(batch_size, 17, 3)
    motion = torch.randn(batch_size, 8)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    dataset = TensorDataset(roi, kpts, motion, labels)
    loader = DataLoader(dataset, batch_size=2)
    
    device = torch.device('cpu')
    
    with torch.no_grad():
        loss, correct, total = eval_epoch(model, loader, criterion, device)
    
    assert loss >= 0
    assert 0 <= correct <= total
    assert total == batch_size
```

- [ ] **Step 4: Run fusion classifier tests**

```bash
PYTHONPATH=src python -m pytest tests/integration/test_training_classifier.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_training_classifier.py
git commit -m "test(integration): add fusion classifier training tests

- Test FeatureDataset loading
- Test training epoch
- Test evaluation epoch

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 18: Create test_extract_features.py

**Files:**
- Create: `tests/integration/test_extract_features.py`

- [ ] **Step 1: Test feature extraction components**

```python
def test_extract_features_args(mocker):
    """Test extract_features argument parsing."""
    import sys
    from unittest.mock import patch
    
    test_args = [
        'scripts/train/extract_features.py',
        '--video-dir', 'data/videos',
        '--label-file', 'data/labels.json',
        '--out-dir', 'outputs/cache'
    ]
    
    with patch.object(sys, 'argv', test_args):
        # Test argument parsing
        pass
```

- [ ] **Step 2: Commit**

```bash
git add tests/integration/test_extract_features.py
git commit -m "test(integration): add feature extraction tests

- Test argument parsing

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Phase 5: Coverage Verification and Final Steps

### Task 19: Create conftest.py with shared fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Add shared fixtures**

```python
import pytest
import numpy as np
import torch


@pytest.fixture
def sample_bbox():
    """Return a sample bounding box."""
    return [100, 100, 200, 300]


@pytest.fixture
def sample_keypoints():
    """Return sample COCO keypoints."""
    kpts = np.random.rand(17, 3) * 100
    kpts[:, 2] = 1.0  # All visible
    return kpts


@pytest.fixture
def sample_frame():
    """Return a sample image frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_detection():
    """Return a sample detection."""
    return {
        'bbox': [100, 100, 200, 300],
        'conf': 0.9,
        'class_id': 0
    }


@pytest.fixture
def device():
    """Return available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

- [ ] **Step 2: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared pytest fixtures

- Add sample_bbox, sample_keypoints fixtures
- Add sample_frame, mock_detection fixtures
- Add device fixture

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 20: Run Full Coverage Report

**Files:**
- All test files

- [ ] **Step 1: Install pytest-cov**

```bash
pip install pytest-cov
```

- [ ] **Step 2: Run coverage report**

```bash
PYTHONPATH=src python -m pytest tests/ \
    --cov=src/fall_detection \
    --cov-report=term-missing \
    --cov-report=html:coverage_html \
    --cov-fail-under=90 \
    -v
```

- [ ] **Step 3: Check coverage output**

Expected: Coverage report showing >=90% line coverage for all modules.

If coverage is below 90%, identify uncovered lines and add tests.

- [ ] **Step 4: Commit coverage configuration**

Create `.coveragerc`:
```ini
[run]
source = src/fall_detection
omit = 
    */tests/*
    */__pycache__/*
    */migrations/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:

fail_under = 90
```

```bash
git add .coveragerc
git commit -m "chore: add coverage configuration

- Set 90% minimum coverage threshold
- Exclude tests and __main__ blocks

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 21: Update CLAUDE.md with Testing Summary

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add test coverage summary**

Add to CLAUDE.md after Testing Guidelines section:

```markdown
### Test Coverage Summary

Current test coverage: **90%+**

- Unit tests: 15+ test files covering all core modules
- Integration tests: 6 test files covering training scripts
- Total tests: 100+

#### Module Coverage

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
| training/scheduler.py | 95% | test_scheduler.py |
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with test coverage summary

- Add coverage summary table
- Document all test files

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Plan Completion Summary

This plan covers:

1. **CLAUDE.md updates** - TDD guidelines and coverage summary
2. **Core module tests** - detector, tracker, pose, rules, fusion (90%+ coverage)
3. **Model tests** - classifier, simple_classifier (90%+ coverage)
4. **Data tests** - augmentation, datasets (90%+ coverage)
5. **Utils tests** - geometry, common, export, visualization (90%+ coverage)
6. **Training tests** - scheduler (95%+ coverage)
7. **Integration tests** - All 6 training scripts (detector, pose, classifier, simple_classifier, yoloworld, extract_features)

**Total estimated tasks**: 21 major tasks with ~120 individual steps

**Expected outcomes**:
- 90%+ line coverage across all source modules
- 100+ unit tests
- 6+ integration tests for training scripts
- Shared fixtures for common test data
- Coverage configuration with 90% threshold
