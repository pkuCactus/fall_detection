import numpy as np
from fall_detection.pipeline.pipeline import FallDetectionPipeline


def test_pipeline_runs_3_frames_with_mock():
    pipe = FallDetectionPipeline("configs/default.yaml")
    # mock detector 返回固定人体框
    pipe.detector = lambda img, conf_thresh=0.3: [
        {"bbox": [100.0, 100.0, 200.0, 400.0], "conf": 0.9, "class_id": 0}
    ]
    # mock pose estimator 返回有效的关键点（站立姿态）
    def mock_pose(img, bboxes):
        kpts = np.zeros((17, 3), dtype=np.float32)
        # 设置一些关键点模拟站立姿态
        kpts[0] = [150, 120, 0.9]   # nose
        kpts[1] = [145, 125, 0.9]   # leye
        kpts[2] = [155, 125, 0.9]   # reye
        kpts[5] = [130, 150, 0.9]   # lshoulder
        kpts[6] = [170, 150, 0.9]   # rshoulder
        kpts[11] = [140, 250, 0.9]  # lhip
        kpts[12] = [160, 250, 0.9]  # rhip
        kpts[15] = [135, 380, 0.9]  # lankle
        kpts[16] = [165, 380, 0.9]  # rankle
        return [kpts] if bboxes else []
    pipe.pose_estimator = mock_pose

    for _ in range(3):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = pipe.process_frame(frame)
        assert "tracks" in out
        assert "track_kpts" in out
        assert len(out["tracks"]) >= 1
