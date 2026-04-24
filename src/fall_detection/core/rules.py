from typing import List, Dict, Tuple, Any
import numpy as np


class RuleEngine:
    """规则引擎 A/B/C/D/E + 姿态分类."""

    KEYPOINT_CONF_THRESH = 0.1
    NUM_KEYPOINTS = 17
    POSTURE_VISIBLE_THRESH = 0.4
    POSTURE_HIP_RATIO_THRESH = 0.45
    POSTURE_STANDING_THRESH = 0.75
    POSTURE_SITTING_THRESH = 0.5
    POSTURE_CROUCHING_THRESH = 0.35

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
        self.no_keypoint_thresh = cfg.get("no_keypoint_thresh", 0.1)  # 关键点几乎不可见阈值
        self.cls_posture_t1 = cfg.get("cls_posture_t1", 1.0)  # 分类器高分阈值，>t1强制lying
        self.cls_posture_t2 = cfg.get("cls_posture_t2", -1.0)  # 分类器低分阈值，<t2强制standing

    def _compute_body_metrics(
        self, kpts: np.ndarray, bbox: List[float]
    ) -> Dict[str, Any]:
        """计算人体关键指标：head_y, lower_y, h_ratio, hip_ratio, visible_ratio, kpt_aspect, torso_angle 等."""
        bbox_h = max(1.0, bbox[3] - bbox[1])
        bbox_w = max(1.0, bbox[2] - bbox[0])
        visible_ratio = float(np.sum(kpts[:, 2] > self.KEYPOINT_CONF_THRESH) / self.NUM_KEYPOINTS)

        head_vals = [kpts[i, 1] for i in (0, 1, 2) if kpts[i, 2] > self.KEYPOINT_CONF_THRESH]
        ankle_vals = [kpts[i, 1] for i in (15, 16) if kpts[i, 2] > self.KEYPOINT_CONF_THRESH]
        hip_vals = [kpts[i, 1] for i in (11, 12) if kpts[i, 2] > self.KEYPOINT_CONF_THRESH]

        head_y = np.mean(head_vals) if head_vals else bbox[1] + 0.1 * bbox_h
        lower_y = np.mean(ankle_vals) if ankle_vals else bbox[3]
        hip_y = np.mean(hip_vals) if hip_vals else bbox[3]

        h_ratio = abs(head_y - lower_y) / bbox_h
        hip_ratio = abs(head_y - hip_y) / bbox_h

        visible_kpts = kpts[kpts[:, 2] > self.KEYPOINT_CONF_THRESH]
        if len(visible_kpts) >= 4:
            kpt_span_w = float(np.max(visible_kpts[:, 0]) - np.min(visible_kpts[:, 0]))
            kpt_span_h = float(np.max(visible_kpts[:, 1]) - np.min(visible_kpts[:, 1]))
        else:
            kpt_span_w, kpt_span_h = bbox_w, bbox_h

        def _midpoint(i1, i2):
            if kpts[i1, 2] > self.KEYPOINT_CONF_THRESH and kpts[i2, 2] > self.KEYPOINT_CONF_THRESH:
                return ((kpts[i1, 0] + kpts[i2, 0]) / 2.0, (kpts[i1, 1] + kpts[i2, 1]) / 2.0)
            elif kpts[i1, 2] > self.KEYPOINT_CONF_THRESH:
                return (kpts[i1, 0], kpts[i1, 1])
            elif kpts[i2, 2] > self.KEYPOINT_CONF_THRESH:
                return (kpts[i2, 0], kpts[i2, 1])
            return (None, None)

        sh_x, sh_y = _midpoint(5, 6)
        hp_x, hp_y = _midpoint(11, 12)
        torso_angle = 0.0
        if sh_x is not None and hp_x is not None:
            dx, dy = abs(hp_x - sh_x), abs(hp_y - sh_y)
            torso_angle = float(np.degrees(np.arctan2(dx, max(dy, 1e-6))))

        return {
            "bbox_h": float(bbox_h),
            "bbox_w": float(bbox_w),
            "bbox_aspect": float(bbox_w / bbox_h),
            "visible_ratio": float(visible_ratio),
            "head_y": float(head_y),
            "lower_y": float(lower_y),
            "hip_y": float(hip_y),
            "h_ratio": float(h_ratio),
            "hip_ratio": float(hip_ratio),
            "has_ankles": bool(ankle_vals),
            "kpt_span_w": float(kpt_span_w),
            "kpt_span_h": float(kpt_span_h),
            "kpt_aspect": float(kpt_span_w / max(kpt_span_h, 1.0)),
            "torso_angle": float(torso_angle),
        }

    def evaluate(
        self,
        kpts: np.ndarray,
        bbox: List[float],
        history: Dict[str, Any],
        cls_score: float = 0.0,
    ) -> Tuple[float, Dict[str, bool], Dict[str, Any]]:
        """
        评估规则得分.

        Args:
            kpts: (17, 3) 关键点 [x, y, conf].
            bbox: [x1, y1, x2, y2].
            history: dict 包含 'centers': List[(cx, cy), ...].
            cls_score: 分类器跌倒概率，用于辅助姿态判定.

        Returns:
            (S_rule, {"A": bool, "B": bool, "C": bool, "D": bool, "E": bool, "F": bool}, debug_info)
        """
        flags = {"A": False, "B": False, "C": False, "D": False, "E": False, "F": False}

        metrics = self._compute_body_metrics(kpts, bbox)
        bbox_h = metrics["bbox_h"]
        visible_ratio = metrics["visible_ratio"]
        head_y = metrics["head_y"]
        lower_y = metrics["lower_y"]
        hip_y = metrics["hip_y"]
        h_ratio = metrics["h_ratio"]
        hip_ratio = metrics["hip_ratio"]
        has_ankles = metrics["has_ankles"]

        # ---- 姿态预分类（结合分类器得分） ----
        posture = self._classify_posture(kpts, bbox, visible_ratio, cls_score)

        # ---- A: 高度压缩 + 多点贴地 ----
        ground_y_thresh = bbox[1] + (1.0 - self.ground_ratio) * bbox_h
        n_ground = int(
            np.sum(
                (kpts[:, 2] > self.KEYPOINT_CONF_THRESH) & (kpts[:, 1] >= ground_y_thresh)
            )
        )
        # 不再无条件放宽到 hip

        kpt_aspect = metrics.get("kpt_aspect", 0.0)
        torso_angle = metrics.get("torso_angle", 0.0)
        is_compressed = (
            (h_ratio < self.h_ratio_thresh)
            or (kpt_aspect > 1.5)
            or (torso_angle > 55)
        )

        # 规则A：高度压缩（含光轴方向倒下）+ 多点贴地 + 可见性足够
        flags["A"] = (
            is_compressed
            and (n_ground >= self.n_ground_min)
            and (visible_ratio >= self.visible_ratio_min)
        )

        # 明确坐姿不触发 Rule A（sitting 时 hip_ratio < 0.45 且躯干不水平）
        if posture == "sitting" and hip_ratio < 0.45 and torso_angle <= 45:
            flags["A"] = False

        # 额外约束：如果脚踝不可见但hip_ratio显示人明显直立且无水平展开特征，压低规则A置信
        if not has_ankles and hip_ratio > 0.7 and kpt_aspect <= 1.2 and torso_angle <= 45:
            flags["A"] = False

        # ---- B: 地面 ROI 判定 ----
        visible = kpts[kpts[:, 2] > self.KEYPOINT_CONF_THRESH]
        if len(visible) >= 3:
            lowest3 = visible[np.argsort(visible[:, 1])[-3:]]
        elif len(visible) > 0:
            lowest3 = visible
        else:
            lowest3 = np.empty((0, 3))

        if self.ground_roi is None:
            if len(lowest3) > 0:
                n_near_ground = int(np.sum(lowest3[:, 1] >= ground_y_thresh))
                flags["B"] = n_near_ground >= 1
            else:
                flags["B"] = False
        else:
            flags["B"] = all(
                self._point_in_polygon((p[0], p[1]), self.ground_roi)
                for p in lowest3
            )

        # ---- C: 由动到静 或 持续静止 (fps归一化：px/s) ----
        centers = history.get("centers", []) if isinstance(history, dict) else []
        min_history_frames = max(4, int(self.motion_window_seconds * self.fps))
        avg_early = avg_late = 0.0
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
                # 由动到静
                flags["C"] = (avg_early > self.motion_thresh) and (avg_late < self.static_thresh)
                # 或者：已经倒地且持续静止（躺着不动也算符合）
                if not flags["C"] and posture == "lying":
                    # 检查整段时间都是静止状态
                    total_avg = np.mean(displacements)
                    flags["C"] = total_avg < self.static_thresh
            else:
                avg_early = avg_late = 0.0
        else:
            avg_early = avg_late = 0.0

        # ---- D: 垂直方向快速下降 (px/s) ----
        vy_px_s = 0.0
        if len(centers) >= max(3, int(0.2 * self.fps)):
            window = min(len(centers), max(3, int(0.2 * self.fps)))
            recent_y = [c[1] for c in centers[-window:]]
            # px/s: 每帧位移 * fps
            vy_px_s = (recent_y[-1] - recent_y[0]) / (window - 1) * self.fps
            flags["D"] = (vy_px_s > self.fall_vy_thresh) and ((h_ratio < self.h_ratio_thresh) or (posture == "lying"))

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
                flags["E"] = (accel > self.accel_thresh) and ((h_ratio < self.h_ratio_thresh) or (posture == "lying"))

        # ---- F: 关键点完全不可见检测 ----
        # 当关键点几乎完全检测不到时，说明有异常（可能被遮挡或倒地后姿态极端）
        flags["F"] = visible_ratio < self.no_keypoint_thresh

        # S_rule: 等权平均 (A,B,C,D,E,F)
        score = sum(flags.values()) / 6.0

        # 可见性不足时，规则分 cripple（但规则F触发时除外，因为F就是检测不可见）
        if visible_ratio < self.visible_ratio_min and not flags["F"]:
            score *= 0.5

        debug_info = {
            "h_ratio": float(h_ratio),
            "hip_ratio": float(hip_ratio),
            "n_ground": int(n_ground),
            "bbox_h": float(bbox_h),
            "head_y": float(head_y),
            "lower_y": float(lower_y),
            "vy_px_s": float(vy_px_s),
            "accel_mag": float(accel_mag),
            "visible_ratio": float(visible_ratio),
            "posture": posture,
            "kpt_aspect": float(kpt_aspect),
            "torso_angle": float(torso_angle),
            "centers_len": len(centers),
            "min_history_frames": min_history_frames,
        }

        return float(score), {k: bool(v) for k, v in flags.items()}, debug_info

    def _classify_posture(
        self, kpts: np.ndarray, bbox: List[float], visible_ratio: float, cls_score: float = 0.0
    ) -> str:
        """基于h_ratio、关键点二维分布、躯干角度和分类器得分做粗粒度姿态分类."""
        if visible_ratio < self.POSTURE_VISIBLE_THRESH:
            return "unknown"

        metrics = self._compute_body_metrics(kpts, bbox)
        h_ratio = metrics["h_ratio"]
        hip_ratio = metrics["hip_ratio"]
        kpt_aspect = metrics.get("kpt_aspect", 0.0)
        torso_angle = metrics.get("torso_angle", 0.0)
        bbox_aspect = metrics.get("bbox_aspect", 1.0)

        # 光轴方向跌倒：关键点水平展开或躯干接近水平
        if kpt_aspect > 1.5 or torso_angle > 55:
            posture = "lying"
        # 检测框明显变宽且人有一定压缩，也判为跌倒
        elif bbox_aspect > 1.3 and kpt_aspect > 1.0 and h_ratio < 0.65:
            posture = "lying"
        elif h_ratio > self.POSTURE_STANDING_THRESH:
            posture = "standing"
        elif h_ratio > self.POSTURE_SITTING_THRESH:
            if hip_ratio < self.POSTURE_HIP_RATIO_THRESH:
                posture = "sitting"
            else:
                posture = "standing" if h_ratio > 0.65 else "sitting"
        elif h_ratio > self.POSTURE_CROUCHING_THRESH:
            # 上半身相对直立时判为 sitting 而非 crouching，减少坐姿误判
            if hip_ratio < self.POSTURE_HIP_RATIO_THRESH:
                posture = "sitting"
            else:
                posture = "crouching"
        else:
            posture = "lying"

        # 分类器得分辅助修正姿态（仅当阈值在合法范围0~1内时生效）
        if 0.0 <= self.cls_posture_t1 <= 1.0 and 0.0 <= self.cls_posture_t2 <= 1.0:
            if cls_score > self.cls_posture_t1:
                posture = "lying"
            elif cls_score < self.cls_posture_t2:
                posture = "standing"
            elif self.cls_posture_t2 <= cls_score <= self.cls_posture_t1:
                # 中间区间：若原姿态为standing，降级为sitting（非站立）
                if posture == "standing":
                    posture = "sitting"
        return posture

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
