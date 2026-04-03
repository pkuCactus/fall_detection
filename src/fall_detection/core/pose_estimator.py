from typing import List
import numpy as np
from ultralytics import YOLO

from fall_detection.utils.geometry import iou


class PoseEstimator:
    """封装 YOLOv8n-pose 17 关键点估计器."""

    def __init__(self, model_name: str = "yolov8n-pose", model_path: str = None):
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(f"{model_name}.pt")

    def __call__(self, img: np.ndarray, bboxes: List[List[float]]) -> List[np.ndarray]:
        """
        对单帧图像跑整图姿态估计，然后按 IoU 关联到输入的 bboxes.

        Args:
            img: numpy array, HWC, BGR.
            bboxes: List of [x1, y1, x2, y2].

        Returns:
            List[np.ndarray]: 每个元素 shape=(17, 3), 格式 [x, y, conf].
                              未匹配到则返回全零数组。
        """
        if len(bboxes) == 0:
            return []

        results = self.model(img, verbose=False)
        pose_boxes = []
        pose_kpts = []
        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue
            # persons only
            for box, kpt in zip(result.boxes, result.keypoints):
                cls_id = int(box.cls.item())
                if cls_id != 0:
                    continue
                xyxy = box.xyxy.cpu().numpy().flatten().tolist()
                pose_boxes.append(xyxy)
                # keypoints: shape (17, 3) -> [x, y, conf]
                kpt_arr = kpt.xy.cpu().numpy()  # shape (17, 2) or (1, 17, 2)
                conf_arr = kpt.conf.cpu().numpy()  # shape (17,) or (1, 17)
                # Handle different shapes from YOLOv8
                if kpt_arr.ndim == 3:
                    kpt_arr = kpt_arr.reshape(-1, 2)
                if conf_arr.ndim == 0:
                    conf_arr = conf_arr.reshape(1)
                conf_arr = conf_arr.reshape(-1, 1)  # shape (17, 1)
                pose_kpts.append(np.concatenate([kpt_arr, conf_arr], axis=1))

        if len(pose_boxes) == 0:
            return [np.zeros((17, 3), dtype=np.float32) for _ in bboxes]

        # Greedy IoU matching: for each input bbox, find best pose box
        matched_indices = []
        used_pose = set()
        for bbox in bboxes:
            best_iou = 0.0
            best_idx = -1
            for idx, pb in enumerate(pose_boxes):
                if idx in used_pose:
                    continue
                val = iou(bbox, pb)
                if val > best_iou:
                    best_iou = val
                    best_idx = idx
            if best_idx >= 0 and best_iou > 0.1:
                matched_indices.append(best_idx)
                used_pose.add(best_idx)
            else:
                matched_indices.append(-1)

        out = []
        for idx in matched_indices:
            if idx >= 0:
                out.append(pose_kpts[idx].astype(np.float32))
            else:
                out.append(np.zeros((17, 3), dtype=np.float32))
        return out
