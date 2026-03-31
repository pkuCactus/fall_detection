import numpy as np
from fall_detection.pipeline import FallDetectionPipeline


def test_pipeline_runs_3_frames_with_mock():
    pipe = FallDetectionPipeline("configs/default.yaml")
    # mock detector 返回固定人体框
    pipe.detector = lambda img, conf_thresh=0.3: [
        {"bbox": [100.0, 100.0, 200.0, 400.0], "conf": 0.9, "class_id": 0}
    ]
    for _ in range(3):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = pipe.process_frame(frame)
        assert "tracks" in out
        assert "track_kpts" in out
        assert len(out["tracks"]) >= 1
