import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from fall_detection.core.detector import PersonDetector


class TestPersonDetector:
    """Tests for PersonDetector class."""

    def test_init_with_model_path(self, mocker):
        """Test initialization with model_path parameter."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 640}
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_path='custom_model.pt')

        mock_yolo.assert_called_once_with('custom_model.pt')
        assert detector.model == mock_model
        assert detector.imgsz == 640

    def test_init_with_model_name(self, mocker):
        """Test initialization with model_name parameter."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 640}
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_name='yolov8n')

        mock_yolo.assert_called_once_with('yolov8n.pt')
        assert detector.model == mock_model

    def test_init_default_model_name(self, mocker):
        """Test initialization with default model_name."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 640}
        mock_yolo.return_value = mock_model

        detector = PersonDetector()

        mock_yolo.assert_called_once_with('yolov8n.pt')
        assert detector.model == mock_model

    def test_init_with_set_classes(self, mocker):
        """Test initialization with classes parameter calls set_classes."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 640}
        mock_model.set_classes = MagicMock()
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_name='yolov8n', classes=['person', 'car'])

        mock_model.set_classes.assert_called_once_with(['person', 'car'])

    def test_init_without_set_classes_attribute(self, mocker):
        """Test initialization when model doesn't have set_classes attribute."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 640}
        # Remove set_classes attribute
        del mock_model.set_classes
        mock_yolo.return_value = mock_model

        # Should not raise error
        detector = PersonDetector(model_name='yolov8n', classes=['person'])
        assert detector.model == mock_model

    def test_init_default_imgsz_when_no_args(self, mocker):
        """Test default imgsz when model has no args attribute."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        # No args attribute
        del mock_model.args
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_name='yolov8n')

        assert detector.imgsz == 640

    def test_input_size_property(self, mocker):
        """Test input_size property returns imgsz."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 320}
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_name='yolov8n')

        assert detector.input_size == 320

    def test_call_returns_person_detections(self, mocker):
        """Test __call__ returns person class detections."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 640}
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_name='yolov8n')

        # Mock detection result
        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0  # person class
        mock_box.conf.item.return_value = 0.85
        mock_box.xyxy.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [100.0, 200.0, 300.0, 400.0]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_model.return_value = [mock_result]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = detector(img)

        assert len(boxes) == 1
        assert boxes[0]['bbox'] == [100.0, 200.0, 300.0, 400.0]
        assert boxes[0]['conf'] == 0.85
        assert boxes[0]['class_id'] == 0
        mock_model.assert_called_once_with(img, verbose=False, device=None)

    def test_call_filters_non_person_classes(self, mocker):
        """Test __call__ filters out non-person class detections."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 640}
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_name='yolov8n')

        # Mock detection results - person and car
        mock_person_box = MagicMock()
        mock_person_box.cls.item.return_value = 0  # person class
        mock_person_box.conf.item.return_value = 0.9
        mock_person_box.xyxy.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [100.0, 100.0, 200.0, 200.0]

        mock_car_box = MagicMock()
        mock_car_box.cls.item.return_value = 2  # car class
        mock_car_box.conf.item.return_value = 0.8
        mock_car_box.xyxy.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [300.0, 300.0, 500.0, 500.0]

        mock_result = MagicMock()
        mock_result.boxes = [mock_person_box, mock_car_box]

        mock_model.return_value = [mock_result]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = detector(img)

        assert len(boxes) == 1
        assert boxes[0]['class_id'] == 0

    def test_call_filters_by_confidence_threshold(self, mocker):
        """Test __call__ filters by confidence threshold."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 640}
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_name='yolov8n')

        # Mock detection results - high and low confidence
        mock_high_conf_box = MagicMock()
        mock_high_conf_box.cls.item.return_value = 0
        mock_high_conf_box.conf.item.return_value = 0.5  # above default 0.3
        mock_high_conf_box.xyxy.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [100.0, 100.0, 200.0, 200.0]

        mock_low_conf_box = MagicMock()
        mock_low_conf_box.cls.item.return_value = 0
        mock_low_conf_box.conf.item.return_value = 0.2  # below default 0.3
        mock_low_conf_box.xyxy.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [300.0, 300.0, 400.0, 400.0]

        mock_result = MagicMock()
        mock_result.boxes = [mock_high_conf_box, mock_low_conf_box]

        mock_model.return_value = [mock_result]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = detector(img, conf_thresh=0.3)

        assert len(boxes) == 1
        assert boxes[0]['conf'] == 0.5

    def test_call_with_custom_conf_thresh(self, mocker):
        """Test __call__ with custom confidence threshold."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 640}
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_name='yolov8n')

        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0
        mock_box.conf.item.return_value = 0.5
        mock_box.xyxy.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [100.0, 100.0, 200.0, 200.0]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_model.return_value = [mock_result]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = detector(img, conf_thresh=0.6)

        # Should be filtered out due to conf_thresh
        assert len(boxes) == 0

    def test_call_handles_none_boxes(self, mocker):
        """Test __call__ handles when result.boxes is None."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 640}
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_name='yolov8n')

        mock_result = MagicMock()
        mock_result.boxes = None

        mock_model.return_value = [mock_result]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = detector(img)

        assert boxes == []

    def test_call_handles_empty_boxes_list(self, mocker):
        """Test __call__ handles empty boxes list."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 640}
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_name='yolov8n')

        mock_result = MagicMock()
        mock_result.boxes = []

        mock_model.return_value = [mock_result]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = detector(img)

        assert boxes == []

    def test_call_handles_multiple_results(self, mocker):
        """Test __call__ handles multiple results from model."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 640}
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_name='yolov8n')

        # First result with person
        mock_box1 = MagicMock()
        mock_box1.cls.item.return_value = 0
        mock_box1.conf.item.return_value = 0.8
        mock_box1.xyxy.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [10.0, 10.0, 50.0, 50.0]

        mock_result1 = MagicMock()
        mock_result1.boxes = [mock_box1]

        # Second result with person
        mock_box2 = MagicMock()
        mock_box2.cls.item.return_value = 0
        mock_box2.conf.item.return_value = 0.75
        mock_box2.xyxy.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = [100.0, 100.0, 150.0, 150.0]

        mock_result2 = MagicMock()
        mock_result2.boxes = [mock_box2]

        mock_model.return_value = [mock_result1, mock_result2]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = detector(img)

        assert len(boxes) == 2

    def test_call_bbox_conversion(self, mocker):
        """Test bbox values are properly converted to float."""
        mock_yolo = mocker.patch('fall_detection.core.detector.YOLO')
        mock_model = MagicMock()
        mock_model.args = {'imgsz': 640}
        mock_yolo.return_value = mock_model

        detector = PersonDetector(model_name='yolov8n')

        mock_box = MagicMock()
        mock_box.cls.item.return_value = 0
        mock_box.conf.item.return_value = 0.9
        # Return numpy array that needs conversion
        mock_xyxy = MagicMock()
        mock_xyxy.flatten.return_value.tolist.return_value = [10, 20, 30, 40]
        mock_cpu = MagicMock()
        mock_cpu.numpy.return_value = mock_xyxy
        mock_box.xyxy.cpu.return_value = mock_cpu

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_model.return_value = [mock_result]

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = detector(img)

        assert len(boxes) == 1
        assert boxes[0]['bbox'] == [10.0, 20.0, 30.0, 40.0]
        assert all(isinstance(v, float) for v in boxes[0]['bbox'])
