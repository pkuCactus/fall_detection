from typing import List, Dict
import numpy as np
import torch.nn as nn
from ultralytics import YOLO, YOLOWorld
from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.detect.predict import DetectionPredictor

from fall_detection.utils.common import normalize_device


class PersonDetector:
    """封装 YOLOv8 人体检测器."""

    def __init__(self, model_name: str = "yolov8n", model_path: str = None, classes: list = None,
                 device: str = None, model_type: str = "yolo", imgsz=None):
        if model_type not in ["yolo", "yolo_world"]:
            raise ValueError(f"Unsupported model_type: {model_type}. Supported types: 'yolo', 'yolo_world'.")
        MODEL = YOLO if model_type == "yolo" else YOLOWorld
        if model_path:
            self.model = MODEL(model_path)
        else:
            self.model = MODEL(f"{model_name}.pt")
        self.model.to(normalize_device(device))
        self.imgsz = imgsz if imgsz is not None else getattr(self.model, "args", {}).get("imgsz", 640)
        if classes and hasattr(self.model, "set_classes"):
            self.model.set_classes(classes)
        self.rect = False  # ultralytics letter box not auto

    @property
    def input_size(self):
        """返回模型输入分辨率（int 或 [w, h] list）."""
        return self.imgsz

    def __call__(self, img: np.ndarray, conf_thresh: float = 0.3, filter_class_id: int = 0) -> List[Dict]:
        """
        对单帧图像执行人体检测.

        Args:
            img: numpy array, HWC, BGR (OpenCV 默认格式).
            conf_thresh: 置信度阈值.
            filter_class_id: 仅返回指定类别，None 表示返回所有类别.

        Returns:
            List[Dict]: 每个元素包含 bbox [x1, y1, x2, y2], conf, class_id, class_name.
        """
        results = self.model(img, verbose=False, imgsz=self.imgsz, rect=self.rect)
        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            names = getattr(result, "names", {})
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if filter_class_id is not None and cls_id != filter_class_id:
                    continue
                if conf < conf_thresh:
                    continue
                xyxy = box.xyxy.cpu().numpy().flatten().tolist()
                boxes.append(
                    {
                        "bbox": [float(v) for v in xyxy],
                        "conf": conf,
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                    }
                )
        return boxes
