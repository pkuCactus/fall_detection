import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fall_detection.core.pose_estimator import PoseEstimator


def create_mock_tensor(data):
    """Helper to create a mock tensor that supports .cpu().numpy() chain."""
    mock_cpu = MagicMock()
    mock_cpu.numpy.return_value = data
    mock_tensor = MagicMock()
    mock_tensor.cpu.return_value = mock_cpu
    return mock_tensor


class TestPoseEstimator:
    """Tests for PoseEstimator class."""

    def test_init_with_model_name(self, monkeypatch):
        """Test initialization with model name."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        est = PoseEstimator(model_name='yolov8n-pose')

        mock_yolo_class.assert_called_once_with('yolov8n-pose.pt')
        assert est.model == mock_model

    def test_init_with_model_path(self, monkeypatch):
        """Test initialization with custom model path."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        est = PoseEstimator(model_path='/custom/path/to/model.pt')

        mock_yolo_class.assert_called_once_with('/custom/path/to/model.pt')
        assert est.model == mock_model

    def test_call_with_empty_bboxes(self, monkeypatch):
        """Test pose estimation with empty bboxes returns empty list."""
        mock_yolo_class = MagicMock()
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = []

        result = est(img, bboxes)

        assert result == []
        mock_yolo_class.return_value.assert_not_called()

    def test_call_with_single_person(self, monkeypatch):
        """Test pose estimation with single person detection."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        # Create mock result with single person
        mock_result = MagicMock()
        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0  # person class
        mock_box.xyxy = create_mock_tensor(np.array([[50, 50, 150, 250]]))

        mock_kpt = MagicMock()
        mock_kpt.xy = create_mock_tensor(np.random.rand(17, 2) * 100)
        mock_kpt.conf = create_mock_tensor(np.random.rand(17))

        mock_result.boxes = [mock_box]
        mock_result.keypoints = [mock_kpt]

        mock_model.return_value = [mock_result]

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [[50, 50, 150, 250]]

        result = est(img, bboxes)

        assert len(result) == 1
        assert result[0].shape == (17, 3)
        assert result[0].dtype == np.float32

    def test_call_with_multiple_people(self, monkeypatch):
        """Test pose estimation with multiple people."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        # Create mock result with two people
        mock_result = MagicMock()

        mock_box1 = MagicMock()
        mock_box1.cls.item.return_value = 0
        mock_box1.xyxy = create_mock_tensor(np.array([[50, 50, 150, 250]]))

        mock_box2 = MagicMock()
        mock_box2.cls.item.return_value = 0
        mock_box2.xyxy = create_mock_tensor(np.array([[200, 100, 300, 350]]))

        mock_kpt1 = MagicMock()
        mock_kpt1.xy = create_mock_tensor(np.random.rand(17, 2) * 100)
        mock_kpt1.conf = create_mock_tensor(np.random.rand(17))

        mock_kpt2 = MagicMock()
        mock_kpt2.xy = create_mock_tensor(np.random.rand(17, 2) * 100 + 50)
        mock_kpt2.conf = create_mock_tensor(np.random.rand(17))

        mock_result.boxes = [mock_box1, mock_box2]
        mock_result.keypoints = [mock_kpt1, mock_kpt2]

        mock_model.return_value = [mock_result]

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [[50, 50, 150, 250], [200, 100, 300, 350]]

        result = est(img, bboxes)

        assert len(result) == 2
        assert all(r.shape == (17, 3) for r in result)
        assert all(r.dtype == np.float32 for r in result)

    def test_call_with_no_pose_detections(self, monkeypatch):
        """Test when YOLO returns no pose detections."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        # Create mock result with no detections
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_result.keypoints = []

        mock_model.return_value = [mock_result]

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [[50, 50, 150, 250], [200, 100, 300, 350]]

        result = est(img, bboxes)

        assert len(result) == 2
        assert all(np.allclose(r, np.zeros((17, 3), dtype=np.float32)) for r in result)

    def test_call_with_none_boxes_and_keypoints(self, monkeypatch):
        """Test when YOLO returns None for boxes and keypoints."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        # Create mock result with None boxes and keypoints
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_result.keypoints = None

        mock_model.return_value = [mock_result]

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [[50, 50, 150, 250]]

        result = est(img, bboxes)

        assert len(result) == 1
        assert np.allclose(result[0], np.zeros((17, 3), dtype=np.float32))

    def test_call_with_non_person_class(self, monkeypatch):
        """Test filtering out non-person detections (class_id != 0)."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        # Create mock result with non-person class
        mock_result = MagicMock()

        mock_box = MagicMock()
        mock_box.cls.item.return_value = 15  # not a person
        mock_box.xyxy = create_mock_tensor(np.array([[50, 50, 150, 250]]))

        mock_kpt = MagicMock()
        mock_kpt.xy = create_mock_tensor(np.random.rand(17, 2) * 100)
        mock_kpt.conf = create_mock_tensor(np.random.rand(17))

        mock_result.boxes = [mock_box]
        mock_result.keypoints = [mock_kpt]

        mock_model.return_value = [mock_result]

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [[50, 50, 150, 250]]

        result = est(img, bboxes)

        assert len(result) == 1
        assert np.allclose(result[0], np.zeros((17, 3), dtype=np.float32))

    def test_call_with_mixed_classes(self, monkeypatch):
        """Test with mixed person and non-person detections."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        mock_result = MagicMock()

        # Person class
        mock_box1 = MagicMock()
        mock_box1.cls.item.return_value = 0
        mock_box1.xyxy = create_mock_tensor(np.array([[50, 50, 150, 250]]))

        # Non-person class
        mock_box2 = MagicMock()
        mock_box2.cls.item.return_value = 15
        mock_box2.xyxy = create_mock_tensor(np.array([[200, 100, 300, 350]]))

        mock_kpt1 = MagicMock()
        mock_kpt1.xy = create_mock_tensor(np.random.rand(17, 2) * 100)
        mock_kpt1.conf = create_mock_tensor(np.random.rand(17))

        mock_kpt2 = MagicMock()
        mock_kpt2.xy = create_mock_tensor(np.random.rand(17, 2) * 100 + 50)
        mock_kpt2.conf = create_mock_tensor(np.random.rand(17))

        mock_result.boxes = [mock_box1, mock_box2]
        mock_result.keypoints = [mock_kpt1, mock_kpt2]

        mock_model.return_value = [mock_result]

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [[50, 50, 150, 250]]

        result = est(img, bboxes)

        assert len(result) == 1
        assert result[0].shape == (17, 3)

    def test_iou_threshold_matching(self, monkeypatch):
        """Test that bboxes with IoU <= 0.1 return zeros."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        mock_result = MagicMock()

        # Pose detection far from input bbox
        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0
        mock_box.xyxy = create_mock_tensor(np.array([[400, 400, 500, 600]]))  # Far from [50, 50, 150, 250]

        mock_kpt = MagicMock()
        mock_kpt.xy = create_mock_tensor(np.random.rand(17, 2) * 100)
        mock_kpt.conf = create_mock_tensor(np.random.rand(17))

        mock_result.boxes = [mock_box]
        mock_result.keypoints = [mock_kpt]

        mock_model.return_value = [mock_result]

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [[50, 50, 150, 250]]  # Far from detection

        result = est(img, bboxes)

        assert len(result) == 1
        # Should return zeros because IoU is 0 (below 0.1 threshold)
        assert np.allclose(result[0], np.zeros((17, 3), dtype=np.float32))

    def test_unmatched_bbox_returns_zeros(self, monkeypatch):
        """Test that unmatched bboxes return zero keypoints."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        mock_result = MagicMock()

        # Only one pose detection
        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0
        mock_box.xyxy = create_mock_tensor(np.array([[50, 50, 150, 250]]))

        mock_kpt = MagicMock()
        mock_kpt.xy = create_mock_tensor(np.random.rand(17, 2) * 100)
        mock_kpt.conf = create_mock_tensor(np.random.rand(17))

        mock_result.boxes = [mock_box]
        mock_result.keypoints = [mock_kpt]

        mock_model.return_value = [mock_result]

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Two bboxes but only one pose detection
        bboxes = [[50, 50, 150, 250], [300, 300, 400, 500]]

        result = est(img, bboxes)

        assert len(result) == 2
        # First should be matched
        assert result[0].shape == (17, 3)
        assert not np.allclose(result[0], np.zeros((17, 3), dtype=np.float32))
        # Second should be zeros (unmatched)
        assert np.allclose(result[1], np.zeros((17, 3), dtype=np.float32))

    def test_keypoints_3d_shape_handling(self, monkeypatch):
        """Test handling of 3D keypoint shapes from YOLOv8."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        mock_result = MagicMock()

        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0
        mock_box.xyxy = create_mock_tensor(np.array([[50, 50, 150, 250]]))

        mock_kpt = MagicMock()
        # Return 3D shape (1, 17, 2) instead of (17, 2)
        mock_kpt.xy = create_mock_tensor(np.random.rand(1, 17, 2) * 100)
        mock_kpt.conf = create_mock_tensor(np.random.rand(17))

        mock_result.boxes = [mock_box]
        mock_result.keypoints = [mock_kpt]

        mock_model.return_value = [mock_result]

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [[50, 50, 150, 250]]

        result = est(img, bboxes)

        assert len(result) == 1
        assert result[0].shape == (17, 3)

    def test_conf_0d_shape_handling(self, monkeypatch):
        """Test handling of 0D confidence array from YOLOv8 (single keypoint case)."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        mock_result = MagicMock()

        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0
        mock_box.xyxy = create_mock_tensor(np.array([[50, 50, 150, 250]]))

        mock_kpt = MagicMock()
        # Single keypoint case where conf is 0D
        mock_kpt.xy = create_mock_tensor(np.random.rand(1, 2) * 100)
        # Return 0D array for conf (single keypoint)
        mock_kpt.conf = create_mock_tensor(np.array(0.5))

        mock_result.boxes = [mock_box]
        mock_result.keypoints = [mock_kpt]

        mock_model.return_value = [mock_result]

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [[50, 50, 150, 250]]

        result = est(img, bboxes)

        assert len(result) == 1
        assert result[0].shape == (1, 3)

    def test_conf_2d_shape_handling(self, monkeypatch):
        """Test handling of 2D confidence array from YOLOv8."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        mock_result = MagicMock()

        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0
        mock_box.xyxy = create_mock_tensor(np.array([[50, 50, 150, 250]]))

        mock_kpt = MagicMock()
        mock_kpt.xy = create_mock_tensor(np.random.rand(17, 2) * 100)
        # Return 2D array (1, 17) for conf
        mock_kpt.conf = create_mock_tensor(np.random.rand(1, 17))

        mock_result.boxes = [mock_box]
        mock_result.keypoints = [mock_kpt]

        mock_model.return_value = [mock_result]

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [[50, 50, 150, 250]]

        result = est(img, bboxes)

        assert len(result) == 1
        assert result[0].shape == (17, 3)

    def test_greedy_matching_multiple_poses(self, monkeypatch):
        """Test greedy IoU matching with more poses than bboxes."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        mock_result = MagicMock()

        # Three pose detections
        mock_box1 = MagicMock()
        mock_box1.cls.item.return_value = 0
        mock_box1.xyxy = create_mock_tensor(np.array([[50, 50, 150, 250]]))

        mock_box2 = MagicMock()
        mock_box2.cls.item.return_value = 0
        mock_box2.xyxy = create_mock_tensor(np.array([[200, 100, 300, 350]]))

        mock_box3 = MagicMock()
        mock_box3.cls.item.return_value = 0
        mock_box3.xyxy = create_mock_tensor(np.array([[400, 200, 500, 450]]))

        mock_kpt1 = MagicMock()
        mock_kpt1.xy = create_mock_tensor(np.random.rand(17, 2) * 100)
        mock_kpt1.conf = create_mock_tensor(np.random.rand(17))

        mock_kpt2 = MagicMock()
        mock_kpt2.xy = create_mock_tensor(np.random.rand(17, 2) * 100 + 50)
        mock_kpt2.conf = create_mock_tensor(np.random.rand(17))

        mock_kpt3 = MagicMock()
        mock_kpt3.xy = create_mock_tensor(np.random.rand(17, 2) * 100 + 100)
        mock_kpt3.conf = create_mock_tensor(np.random.rand(17))

        mock_result.boxes = [mock_box1, mock_box2, mock_box3]
        mock_result.keypoints = [mock_kpt1, mock_kpt2, mock_kpt3]

        mock_model.return_value = [mock_result]

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Only two bboxes - should match to closest poses
        bboxes = [[50, 50, 150, 250], [200, 100, 300, 350]]

        result = est(img, bboxes)

        assert len(result) == 2
        assert all(r.shape == (17, 3) for r in result)

    def test_multiple_results_from_model(self, monkeypatch):
        """Test handling multiple result objects from model."""
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        monkeypatch.setattr('fall_detection.core.pose_estimator.YOLO', mock_yolo_class)

        # Multiple results (can happen with batch processing)
        mock_result1 = MagicMock()
        mock_box1 = MagicMock()
        mock_box1.cls.item.return_value = 0
        mock_box1.xyxy = create_mock_tensor(np.array([[50, 50, 150, 250]]))

        mock_kpt1 = MagicMock()
        mock_kpt1.xy = create_mock_tensor(np.random.rand(17, 2) * 100)
        mock_kpt1.conf = create_mock_tensor(np.random.rand(17))

        mock_result1.boxes = [mock_box1]
        mock_result1.keypoints = [mock_kpt1]

        mock_result2 = MagicMock()
        mock_box2 = MagicMock()
        mock_box2.cls.item.return_value = 0
        mock_box2.xyxy = create_mock_tensor(np.array([[200, 100, 300, 350]]))

        mock_kpt2 = MagicMock()
        mock_kpt2.xy = create_mock_tensor(np.random.rand(17, 2) * 100 + 50)
        mock_kpt2.conf = create_mock_tensor(np.random.rand(17))

        mock_result2.boxes = [mock_box2]
        mock_result2.keypoints = [mock_kpt2]

        mock_model.return_value = [mock_result1, mock_result2]

        est = PoseEstimator(model_name='yolov8n-pose')
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bboxes = [[50, 50, 150, 250], [200, 100, 300, 350]]

        result = est(img, bboxes)

        assert len(result) == 2
        assert all(r.shape == (17, 3) for r in result)
