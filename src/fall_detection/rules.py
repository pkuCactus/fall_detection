from typing import List, Dict, Tuple, Any
import numpy as np


class RuleEngine:
    """规则引擎 A/B/C/D/E + 姿态分类."""

    def __init__(self, config: Dict[str, Any] = None, fps: int = 25):
        cfg = config or {}
        self.fps = fps
        self.h_ratio_thresh = cfg.get("h_ratio_thresh", 0.6)
        self.n_ground_min = cfg.get("n_ground_min", 2)
        self.motion_window_seconds = cfg.get("motion_window_seconds", 1.5)
        self.trigger_thresh = cfg.get("trigger_thresh", 0.75)
        self.ground_roi = cfg.get("ground_roi", None)
        self.motion_thresh = cfg.get("motion_thresh", 75.0)
        self.static_thresh = cfg.get("static_thresh", 20.0)
        self.ground_ratio = cfg.get("ground_ratio", 0.40)
        self.fall_vy_thresh = cfg.get("fall_vy_thresh", 400.0)
        self.accel_thresh = cfg.get("accel_thresh", 150.0)
        self.visible_ratio_min = cfg.get("visible_ratio_min", 0.6)

    def evaluate(
        self,
        kpts: np.ndarray,
        bbox: List[float],
        history: Dict[str, Any],
    ) -> Tuple[float, Dict[str, bool], Dict[str, Any]]:
        """
        评估规则得分.

        Args:
            kpts: (17, 3) 关键点 [x, y, conf].
            bbox: [x1, y1, x2, y2].
            history: dict 包含 'centers': List[(cx, cy), ...].

        Returns:
            (S_rule, {"A": bool, "B": bool, "C": bool, "D": bool, "E": bool}, debug_info)
        """
        flags = {"A": False, "B": False, "C": False, "D": False, "E": False}

        bbox_h = max(1.0, bbox[3] - bbox[1])
        visible_ratio = float(np.sum(kpts[:, 2] > 0.1) / 17.0)

        # ---- 姿态预分类 ----
        posture = self._classify_posture(kpts, bbox, visible_ratio)

        # ---- A: 高度压缩 + 多点贴地 ----
        head_idxs = [0, 1, 2]
        ankle_idxs = [15, 16]

        head_vals = [kpts[i, 1] for i in head_idxs if kpts[i, 2] > 0.1]
        ankle_vals = [kpts[i, 1] for i in ankle_idxs if kpts[i, 2] > 0.1]

        head_y = np.mean(head_vals) if head_vals else bbox[1] + 0.1 * bbox_h
        # 脚踝被截断时，使用bbox底部作为fallback，避免hip替代导致h_ratio虚低
        lower_y = np.mean(ankle_vals) if ankle_vals else bbox[3]

        # 同时计算基于hip的备用ratio，但只用于检测站立/坐姿，不用于跌倒触发
        hip_vals = [kpts[i, 1] for i in [11, 12] if kpts[i, 2] > 0.1]
        hip_y = np.mean(hip_vals) if hip_vals else bbox[3]
        hip_ratio = abs(head_y - hip_y) / bbox_h

        h_ratio = abs(head_y - lower_y) / bbox_h

        # 贴地判定：y 在 bbox 底部 ground_ratio 范围内
        ground_y_thresh = bbox[1] + (1.0 - self.ground_ratio) * bbox_h
        n_ground = int(np.sum((kpts[:, 2] > 0.1) & (kpts[:, 1] >= ground_y_thresh)))
        # 不再无条件放宽到 hip

        # 规则A：h_ratio低且多点贴地，且关键点可见性足够
        flags["A"] = (
            (h_ratio < self.h_ratio_thresh)
            and (n_ground >= self.n_ground_min)
            and (visible_ratio >= self.visible_ratio_min)
        )

        # 额外约束：如果脚踝不可见但hip_ratio显示人明显直立，压低规则A置信
        if not ankle_vals and hip_ratio > 0.7:
            flags["A"] = False

        # ---- B: 地面 ROI 判定 ----
        visible = kpts[kpts[:, 2] > 0.1]
        if len(visible) >= 3:
            lowest3 = visible[np.argsort(visible[:, 1])[-3:]]
        elif len(visible) > 0:
            lowest3 = visible
        else:
            lowest3 = np.empty((0, 3))

        if self.ground_roi is None:
            if len(lowest3) > 0:
                # 使用与规则A一致的ground_ratio判定贴地
                ground_y_thresh = bbox[1] + (1.0 - self.ground_ratio) * bbox_h
                n_near_ground = int(np.sum(lowest3[:, 1] >= ground_y_thresh))
                flags["B"] = n_near_ground >= 1
            else:
                flags["B"] = False
        else:
            flags["B"] = all(
                self._point_in_polygon((p[0], p[1]), self.ground_roi)
                for p in lowest3
            )

        # ---- C: 由动到静 (fps归一化：px/s) ----
        centers = history.get("centers", []) if isinstance(history, dict) else []
        min_history_frames = max(4, int(self.motion_window_seconds * self.fps))
        if len(centers) >= min_history_frames:
            displacements = []
            for i in range(1, len(centers)):
                dx = centers[i][0] - centers[i - 1][0]
                dy = centers[i][1] - centers[i - 1][1]
                displacements.append(np.sqrt(dx ** 2 + dy ** 2) * self.fps)

            mid = len(displacements) // 2
            if mid > 0:
                avg_early = np.mean(displacements[:mid])
                avg_late = np.mean(displacements[mid:])
                flags["C"] = (avg_early > self.motion_thresh) and (avg_late < self.static_thresh)
        else:
            avg_early = avg_late = 0.0

        # ---- D: 垂直方向快速下降 (px/s) ----
        vy_px_s = 0.0
        if len(centers) >= max(3, int(0.2 * self.fps)):
            window = min(len(centers), max(3, int(0.2 * self.fps)))
            recent_y = [c[1] for c in centers[-window:]]
            # px/s: 每帧位移 * fps
            vy_px_s = (recent_y[-1] - recent_y[0]) / (window - 1) * self.fps
            flags["D"] = (vy_px_s > self.fall_vy_thresh) and (h_ratio < self.h_ratio_thresh)

        # ---- E: 加速度辅助判定 (px/s^2) ----
        accel_mag = 0.0
        if len(centers) >= 4:
            # 计算最近几帧速度，再求加速度
            recent = centers[-4:]
            vys = []
            for i in range(1, len(recent)):
                vys.append((recent[i][1] - recent[i - 1][1]) * self.fps)
            if len(vys) >= 2:
                # 加速度 = 速度差 / 时间(秒)
                accel = (vys[-1] - vys[0]) / ((len(vys)) / self.fps)
                accel_mag = abs(accel)
                # 向下加速度大且当前姿态低 => 快速跌落特征
                flags["E"] = (accel > self.accel_thresh) and (h_ratio < self.h_ratio_thresh)

        # S_rule: 等权平均 (A,B,C,D,E)
        score = sum(flags.values()) / 5.0

        # 可见性不足时，规则分 cripple
        if visible_ratio < self.visible_ratio_min:
            score *= 0.5

        debug_info = {
            "h_ratio": float(h_ratio),
            "hip_ratio": float(hip_ratio),
            "n_ground": int(n_ground),
            "bbox_h": float(bbox_h),
            "head_y": float(head_y) if head_vals else None,
            "lower_y": float(lower_y),
            "vy_px_s": float(vy_px_s),
            "accel_mag": float(accel_mag),
            "visible_ratio": float(visible_ratio),
            "posture": posture,
            "centers_len": len(centers),
            "min_history_frames": min_history_frames,
        }

        return float(score), {k: bool(v) for k, v in flags.items()}, debug_info

    def _classify_posture(self, kpts: np.ndarray, bbox: List[float], visible_ratio: float) -> str:
        """基于h_ratio和关键点做粗粒度姿态分类."""
        if visible_ratio < 0.4:
            return "unknown"

        bbox_h = max(1.0, bbox[3] - bbox[1])
        head_vals = [kpts[i, 1] for i in [0, 1, 2] if kpts[i, 2] > 0.1]
        ankle_vals = [kpts[i, 1] for i in [15, 16] if kpts[i, 2] > 0.1]
        head_y = np.mean(head_vals) if head_vals else bbox[1] + 0.1 * bbox_h
        lower_y = np.mean(ankle_vals) if ankle_vals else bbox[3]
        h_ratio = abs(head_y - lower_y) / bbox_h

        if h_ratio > 0.75:
            return "standing"
        elif h_ratio > 0.5:
            # 坐下时 hip 会明显低于 standing
            hip_vals = [kpts[i, 1] for i in [11, 12] if kpts[i, 2] > 0.1]
            if hip_vals:
                hip_y = np.mean(hip_vals)
                hip_ratio = abs(head_y - hip_y) / bbox_h
                if hip_ratio < 0.45:
                    return "sitting"
            return "standing" if h_ratio > 0.65 else "sitting"
        elif h_ratio > 0.35:
            return "crouching"
        else:
            return "lying"

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
