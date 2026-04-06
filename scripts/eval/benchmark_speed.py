import sys
import time
import numpy as np
import torch

sys.path.insert(0, "src")
from fall_detection.pipeline.pipeline import FallDetectionPipeline


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(pipeline, frames, n_frames):
    sync()
    t0 = time.perf_counter()
    for _ in range(n_frames):
        pipeline.process_frame(frames.pop(0))
    sync()
    return time.perf_counter() - t0


def benchmark():
    n_frames = 100
    frames = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(n_frames)
    ]

    base = FallDetectionPipeline("configs/default.yaml")
    base.detector = lambda img, conf_thresh=0.3: [
        {"bbox": [100.0, 100.0, 200.0, 400.0], "conf": 0.9, "class_id": 0}
    ]
    base.skip_frames = 0

    # 预热
    for i in range(min(5, n_frames)):
        base.process_frame(frames[i])

    def clone_state(p):
        q = FallDetectionPipeline("configs/default.yaml")
        q.detector = p.detector
        q.skip_frames = p.skip_frames
        q.pose_estimator = p.pose_estimator
        q.rule_engine.evaluate = p.rule_engine.evaluate
        q.classifier = p.classifier
        return q

    # A. 完整链路
    pA = clone_state(base)
    pA._frame_counter = 0
    total = measure(pA, frames[:n_frames], n_frames)

    # B. 无 Pose
    pB = clone_state(base)
    pB.pose_estimator = lambda img, bboxes: [
        np.zeros((17, 3), dtype=np.float32) for _ in bboxes
    ]
    pB._frame_counter = 0
    t_no_pose = measure(pB, frames[:n_frames], n_frames)

    # C. 无 Pose + 无 Rules + 无 Classifier
    pC = clone_state(base)
    pC.pose_estimator = lambda img, bboxes: [
        np.zeros((17, 3), dtype=np.float32) for _ in bboxes
    ]
    pC.rule_engine.evaluate = lambda kpts, bbox, history: (0.0, {"A": False, "B": False, "C": False, "D": False, "E": False}, {})
    pC.classifier = lambda roi, kpts, motion: 0.0
    pC._frame_counter = 0
    t_no_pose_rules_cls = measure(pC, frames[:n_frames], n_frames)

    # D. 仅 Tracker（检测返回空）
    pD = clone_state(base)
    pD.detector = lambda img, conf_thresh=0.3: []
    pD._frame_counter = 0
    t_only_track = measure(pD, frames[:n_frames], n_frames)

    t_pose = max(0.0, total - t_no_pose)
    t_rules_cls = max(0.0, t_no_pose - t_no_pose_rules_cls)
    t_det_track = max(0.0, t_no_pose_rules_cls - t_only_track)
    t_track = max(0.0, t_only_track)

    avg_ms = (total / n_frames) * 1000
    fps = n_frames / total if total > 0 else 0

    print("=" * 50)
    print(f"Benchmark Results ({n_frames} frames, 640x480)")
    print("=" * 50)
    print(f"Total time:        {total:.3f}s")
    print(f"Avg per frame:     {avg_ms:.2f}ms")
    print(f"FPS:               {fps:.2f}")
    print("-" * 50)
    print(f"Detection+Track:   {t_det_track / n_frames * 1000:.2f}ms/frame")
    print(f"Tracking only:     {t_track / n_frames * 1000:.2f}ms/frame")
    print(f"Pose:              {t_pose / n_frames * 1000:.2f}ms/frame")
    print(f"Rules+Classifier:  {t_rules_cls / n_frames * 1000:.2f}ms/frame")
    print("=" * 50)


if __name__ == "__main__":
    benchmark()
