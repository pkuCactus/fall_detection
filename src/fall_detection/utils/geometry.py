"""几何工具函数."""


def iou(bbox1, bbox2) -> float:
    """计算两个边界框的 IoU.

    Args:
        bbox1: [x1, y1, x2, y2].
        bbox2: [x1, y1, x2, y2].

    Returns:
        float: IoU 值.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0
