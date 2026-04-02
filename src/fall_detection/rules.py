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
        self.fall_vy_thresh = cfg.get("fall_vy_thresh", 30.0)  # 规则D：垂直下降阈值

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
            (S_rule, {"A": bool, "B": bool, "C": bool, "D": bool})
        """
        flags = {"A": False, "B": False, "C": False, "D": False}

        # ---- A: 高度压缩 + 多点贴地 ----
        head_idxs = [0, 1, 2]   # nose, left eye, right eye
        # 使用髋部作为下半身参考（脚踝经常被截断）
        hip_idxs = [11, 12]     # left hip, right hip
        ankle_idxs = [15, 16]   # left ankle, right ankle

        head_vals = [kpts[i, 1] for i in head_idxs if kpts[i, 2] > 0.1]
        # 优先使用脚踝，如果没有则使用髋部，最后使用bbox
        ankle_vals = [kpts[i, 1] for i in ankle_idxs if kpts[i, 2] > 0.1]
        hip_vals = [kpts[i, 1] for i in hip_idxs if kpts[i, 2] > 0.1]

        head_y = np.mean(head_vals) if head_vals else bbox[1]
        if ankle_vals:
            lower_y = np.mean(ankle_vals)
        elif hip_vals:
            lower_y = np.mean(hip_vals)
        else:
            lower_y = bbox[3]

        bbox_h = max(1.0, bbox[3] - bbox[1])
        h_ratio = abs(head_y - lower_y) / bbox_h

        # 贴地判定：y 在 bbox 底部 ground_ratio 范围内
        ground_y_thresh = bbox[1] + (1.0 - self.ground_ratio) * bbox_h
        n_ground = int(np.sum((kpts[:, 2] > 0.1) & (kpts[:, 1] >= ground_y_thresh)))
        # 如果没有足够的关键点可见，放宽到髋部
        if n_ground < self.n_ground_min and hip_vals:
            n_ground = max(n_ground, len(hip_vals))

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

        # ---- C: 由动到静 + 持续静止 (改进版：使用位移而不是方差) ----
        centers = history.get("centers", []) if isinstance(history, dict) else []
        # 需要至少 motion_window_seconds * fps 帧的历史数据
        min_history_frames = max(8, int(self.motion_window_seconds * 25))
        if len(centers) >= min_history_frames:
            # 计算每帧位移
            displacements = []
            for i in range(1, len(centers)):
                dx = centers[i][0] - centers[i-1][0]
                dy = centers[i][1] - centers[i-1][1]
                displacements.append(np.sqrt(dx**2 + dy**2))

            # 分割前后半段
            mid = len(displacements) // 2
            if mid > 0:
                avg_early = np.mean(displacements[:mid])
                avg_late = np.mean(displacements[mid:])
                # 由动到静：前半段有位移，后半段静止
                flags["C"] = (avg_early > self.motion_thresh) and (avg_late < self.static_thresh)

        # ---- D: 垂直方向快速下降 (检测坐着→跌倒的快速高度变化) ----
        # 使用历史中的高度变化来检测快速下降
        if len(centers) >= 4:
            # 计算最近几帧的垂直速度
            recent_y = [c[1] for c in centers[-4:]]
            vy = recent_y[-1] - recent_y[0]  # 正值表示向下移动

            # 快速下降：垂直速度大且当前高度比低
            # 坐着跌倒时，人会向前/向下快速移动
            if vy > self.fall_vy_thresh and h_ratio < self.h_ratio_thresh:
                flags["D"] = True

        # S_rule: 四个规则按等权平均 (A,B,C,D)
        score = sum(flags.values()) / 4.0
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
