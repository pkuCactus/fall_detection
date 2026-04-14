from typing import List, Dict
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """封装 YOLOv8 人体检测器."""

    def __init__(self, model_name: str = "yolov8n", model_path: str = None, classes: list = None, device: str = None):
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(f"{model_name}.pt")
        self.device = device
        self.imgsz = getattr(self.model, "args", {}).get("imgsz", 640)
        if classes and hasattr(self.model, "set_classes"):
            self.model.set_classes(classes)
        self.device = device

    @property
    def input_size(self) -> int:
        """返回模型输入分辨率."""
        return self.imgsz

    def __call__(self, img: np.ndarray, conf_thresh: float = 0.3) -> List[Dict]:
        """
        对单帧图像执行人体检测.

        Args:
            img: numpy array, HWC, BGR (OpenCV 默认格式).
            conf_thresh: 置信度阈值.

        Returns:
            List[Dict]: 每个元素包含 bbox [x1, y1, x2, y2], conf, class_id.
                        仅返回 person 类 (COCO class_id == 0).
        """
        results = self.model(img, verbose=False, device=self.device)
        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if cls_id == 0 and conf >= conf_thresh:
                    xyxy = box.xyxy.cpu().numpy().flatten().tolist()
                    boxes.append(
                        {
                            "bbox": [float(v) for v in xyxy],
                            "conf": conf,
                            "class_id": cls_id,
                        }
                    )
        return boxes
