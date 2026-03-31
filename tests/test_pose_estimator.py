from fall_detection.pose_estimator import PoseEstimator
import numpy as np


def test_pose_inference():
    est = PoseEstimator(model_name='yolov8n-pose')
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    bboxes = [[10, 10, 100, 200]]
    kpts_list = est(img, bboxes)
    assert len(kpts_list) == 1
    assert kpts_list[0].shape == (17, 3)
