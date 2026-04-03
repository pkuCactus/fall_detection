from fall_detection.core.detector import PersonDetector


def test_detector_loads():
    det = PersonDetector(model_name='yolov8n')
    assert det is not None


def test_detector_inference_shape():
    import numpy as np
    det = PersonDetector(model_name='yolov8n')
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = det(img)
    assert isinstance(boxes, list)
