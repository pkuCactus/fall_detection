from typing import List, Dict, Tuple, Any
import numpy as np


class RuleEngine:
    """规则引擎 A/B/C."""

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        self.h_ratio_thresh = cfg.get("h_ratio_thresh", 0.5)
        self.n_ground_min = cfg.get("n_ground_min", 3)
        self.motion_window_seconds = cfg.get("motion_window_seconds", 1.5)
        self.trigger_thresh = cfg.get("trigger_thresh", 0.6)
        # 默认地面 ROI 为全图（测试方便），格式: List[List[tuple]] 多边形列表
        self.ground_roi = cfg.get("ground_roi", None)
        self.motion_thresh = cfg.get("motion_thresh", 1.0)
        self.static_thresh = cfg.get("static_thresh", 0.5)
        self.ground_ratio = cfg.get("ground_ratio", 0.15)

    def evaluate(
        self,
        kpts: np.ndarray,
        bbox: List[float],
        history: Dict[str, Any],
    ) -> Tuple[float, Dict[str, bool]]:
        """
        评估规则得分.

        Args:
            kpts: (17, 3) 关键点 [x, y, conf].
            bbox: [x1, y1, x2, y2].
            history: dict 包含 'centers': List[(cx, cy), ...].

        Returns:
            (S_rule, {"A": bool, "B": bool, "C": bool})
        """
        flags = {"A": False, "B": False, "C": False}

        # ---- A: 高度压缩 + 多点贴地 ----
        head_idxs = [0, 1, 2]   # nose, left eye, right eye
        ankle_idxs = [15, 16]   # left ankle, right ankle
        head_y = np.mean([kpts[i, 1] for i in head_idxs if kpts[i, 2] > 0.1])
        ankle_y = np.mean([kpts[i, 1] for i in ankle_idxs if kpts[i, 2] > 0.1])
        bbox_h = max(1.0, bbox[3] - bbox[1])
        h_ratio = abs(head_y - ankle_y) / bbox_h

        # 贴地判定：y 在 bbox 底部 ground_ratio 范围内
        ground_y_thresh = bbox[1] + (1.0 - self.ground_ratio) * bbox_h
        n_ground = int(np.sum((kpts[:, 2] > 0.1) & (kpts[:, 1] >= ground_y_thresh)))

        flags["A"] = (h_ratio < self.h_ratio_thresh) and (n_ground >= self.n_ground_min)

        # ---- B: 地面 ROI 判定 ----
        # 取 y 最大的 3 个可见关键点（图像底部）
        visible = kpts[kpts[:, 2] > 0.1]
        if len(visible) >= 3:
            lowest3 = visible[np.argsort(visible[:, 1])[-3:]]
        elif len(visible) > 0:
            lowest3 = visible
        else:
            lowest3 = np.empty((0, 3))

        if self.ground_roi is None:
            # 默认全图通过
            flags["B"] = True
        else:
            flags["B"] = all(
                self._point_in_polygon((p[0], p[1]), self.ground_roi)
                for p in lowest3
            )

        # ---- C: 由动到静 + 持续静止 ----
        centers = history.get("centers", []) if isinstance(history, dict) else []
        if len(centers) >= 4:
            mid = len(centers) // 2
            early = np.array(centers[:mid], dtype=np.float32)
            late = np.array(centers[mid:], dtype=np.float32)
            std_early = np.sqrt(np.var(early[:, 0]) + np.var(early[:, 1]))
            std_late = np.sqrt(np.var(late[:, 0]) + np.var(late[:, 1]))
            flags["C"] = (std_early > self.motion_thresh) and (std_late < self.static_thresh)

        # S_rule: 三个规则按等权平均
        score = sum(flags.values()) / 3.0
        # 转成 Python 原生 bool 方便断言
        return float(score), {k: bool(v) for k, v in flags.items()}

    @staticmethod
    def _point_in_polygon(point, polygon) -> bool:
        """Ray-casting algorithm for a single polygon."""
        x, y = point
        n = len(polygon)
        inside = False
        x1, y1 = polygon[0]
        for i in range(1, n + 1):
            x2, y2 = polygon[i % n]
            if y > min(y1, y2):
                if y <= max(y1, y2):
                    if x <= max(x1, x2):
                        if y1 != y2:
                            xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                        if x1 == x2 or x <= xinters:
                            inside = not inside
            x1, y1 = x2, y2
        return inside
