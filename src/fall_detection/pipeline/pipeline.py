from typing import Dict, List, Any
from collections import defaultdict, deque
import cv2
import yaml
import numpy as np
import torch
import torch.nn.functional as F

from fall_detection.core import PersonDetector, ByteTrackLite, Detection, PoseEstimator, RuleEngine, FusionDecision, SimpleKeypointTracker
from fall_detection.models import FallClassifier, SimpleFallClassifier
from fall_detection.utils.common import normalize_device


class FallDetectionPipeline:
    """端到端跌倒检测 Pipeline."""

    def __init__(self, config_path: str = "configs/default.yaml", device: str = None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.device = normalize_device(device)

        det_cfg = self.cfg.get("detector", {})
        track_cfg = self.cfg.get("tracker", {})
        rules_cfg = self.cfg.get("rules", {})
        fusion_cfg = self.cfg.get("fusion", {})
        pipe_cfg = self.cfg.get("pipeline", {})
        cls_cfg = self.cfg.get("classifier", {})

        det_model_path = det_cfg.get("model_path")
        det_classes = det_cfg.get("classes")
        if det_model_path:
            self.detector = PersonDetector(model_path=det_model_path, classes=det_classes, device=device)
        else:
            self.detector = PersonDetector(model_name="yolov8n", classes=det_classes, device=device)
        self.detector_conf_thresh = det_cfg.get("conf_thresh", 0.3)

        self.tracker = ByteTrackLite(
            track_thresh=track_cfg.get("track_thresh", 0.5),
            match_thresh=track_cfg.get("match_thresh", 0.8),
            max_age=track_cfg.get("max_age", 30),
            min_hits=track_cfg.get("min_hits", 3),
        )

        self.skip_frames = pipe_cfg.get("skip_frames", 2)
        self.fps = pipe_cfg.get("fps", 25)

        pose_cfg = self.cfg.get("pose_estimator", {})
        pose_model_path = pose_cfg.get("model_path")
        if pose_model_path:
            self.pose_estimator = PoseEstimator(model_path=pose_model_path, device=device)
        else:
            self.pose_estimator = PoseEstimator(model_name="yolov8n-pose", device=device)
        self.rule_engine = RuleEngine(rules_cfg, fps=self.fps)

        # 初始化关键点跟踪器
        kpt_cfg = self.cfg.get("keypoint_tracker", {})
        self.use_kpt_tracker = kpt_cfg.get("enabled", False)  # 默认禁用，避免跳帧时关键点闪烁
        if self.use_kpt_tracker:
            self.kpt_tracker = SimpleKeypointTracker(
                n_kpts=17,
                smooth_alpha=kpt_cfg.get("smooth_alpha", 0.7),
                velocity_decay=kpt_cfg.get("velocity_decay", 0.9),
                max_history=kpt_cfg.get("max_history", 5),
                use_optical_flow=kpt_cfg.get("use_optical_flow", False),
            )
        else:
            self.kpt_tracker = None
        # 加载分类器模型，支持从配置指定类型和路径
        cls_type = cls_cfg.get("type", "fusion")  # "fusion" 或 "simple"
        cls_model_path = cls_cfg.get("model_path")
        fall_class_idx = cls_cfg.get("fall_class_idx", 1)  # 跌倒类别索引（默认1）
        if cls_type == "simple":
            # 使用简单的单分支分类器（仅图像输入）
            if cls_model_path is None:
                cls_model_path = "train/simple_classifier/best.pt"
            self.classifier = SimpleFallClassifier(model_path=cls_model_path, fall_class_idx=fall_class_idx, device=device)
            self.use_simple_classifier = True
        else:
            # 使用融合分类器（图像+关键点+运动）
            if cls_model_path is None:
                cls_model_path = "train/classifier/best.pt"
            self.classifier = FallClassifier(model_path=cls_model_path, device=device)
            self.use_simple_classifier = False
        self.fusion = {}  # track_id -> FusionDecision
        self.motion_window_frames = max(1, int(0.5 * self.fps))
        self.history_seconds = 1.5
        self.history_maxlen = max(1, int(self.history_seconds * self.fps))
        self.trigger_thresh = rules_cfg.get("trigger_thresh", 0.6)

        # 根据分类器类型调整融合权重
        if self.use_simple_classifier:
            # 使用图像分类器时，提高分类器权重，降低规则权重
            fusion_cfg["alpha"] = fusion_cfg.get("alpha", 0.5) * 0.3  # 规则分权重降低
            fusion_cfg["beta"] = fusion_cfg.get("beta", 0.3) * 2.0    # 分类器权重提高
            fusion_cfg["alarm_thresh"] = fusion_cfg.get("alarm_thresh", 0.5) * 0.8  # 降低告警阈值

        self._frame_counter = 0
        self._track_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.history_maxlen)
        )
        # 缓存上一帧活跃 tracks，用于抽帧补位
        self._last_active_tracks: List = []
        self._last_track_kpts: Dict[int, np.ndarray] = {}
        # 缓存分类器分数，抽帧时复用
        self._last_cls_scores: Dict[int, float] = {}
        # 维护每个track的融合分数历史（用于可视化时序）
        self._fusion_score_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        run_detection = (self._frame_counter % (self.skip_frames + 1)) == 0
        self._frame_counter += 1

        raw_detections: List = []
        if run_detection:
            det_results = self.detector(frame, conf_thresh=self.detector_conf_thresh)
            raw_detections = [d["bbox"] for d in det_results]
            detections = [Detection(d["bbox"], d["conf"]) for d in det_results]
            active_tracks = self.tracker.update(detections)
            track_kpts = self._estimate_keypoints(frame, active_tracks)
            cls_scores = self._compute_classifier_scores(frame, active_tracks, track_kpts)
        else:
            # 只预测已有轨迹，不跑检测和 pose，但分类器每帧都推理
            for track in self.tracker.tracks:
                track.predict()
            active_tracks = [t for t in self.tracker.tracks if t.time_since_update <= self.skip_frames]
            # 使用关键点跟踪器预测关键点位置（如果启用）
            if self.use_kpt_tracker:
                track_kpts = {}
                for track in active_tracks:
                    tid = track.track_id
                    predicted_kpts = self.kpt_tracker.predict(tid, frame, n_frames=1)
                    track_kpts[tid] = predicted_kpts
                    self._last_track_kpts[tid] = predicted_kpts
            else:
                # 禁用跟踪器时，复用上一帧关键点
                track_kpts = self._last_track_kpts
            cls_scores = self._compute_classifier_scores(frame, active_tracks, track_kpts)

        self._update_track_history(active_tracks)
        result = self._process_tracks(active_tracks, track_kpts, cls_scores)

        if run_detection:
            self._last_cls_scores = cls_scores.copy()

        # 更新帧缓存用于光流跟踪
        if self.use_kpt_tracker:
            self.kpt_tracker.update_frame_cache(frame)

        result["is_detection_frame"] = run_detection
        result["detections"] = raw_detections if run_detection else []
        self._last_active_tracks = active_tracks

        # 清理关键点跟踪器中不活跃的track
        if self.use_kpt_tracker:
            active_tids = {t.track_id for t in active_tracks}
            for tid in list(self.kpt_tracker._track_states.keys()):
                if tid not in active_tids:
                    self.kpt_tracker.remove_track(tid)
        return result

    def _estimate_keypoints(
        self, frame: np.ndarray, active_tracks: List
    ) -> Dict[int, np.ndarray]:
        """对活跃轨迹估计关键点."""
        bboxes = []
        valid_tids = []
        for track in active_tracks:
            tid = track.track_id
            x1, y1, x2, y2 = track.to_tlbr()
            h = y2 - y1
            if h >= 20:
                bboxes.append([x1, y1, x2, y2])
                valid_tids.append(tid)
            else:
                self._last_track_kpts[tid] = np.zeros((17, 3), dtype=np.float32)

        track_kpts: Dict[int, np.ndarray] = {}
        if bboxes:
            kpts_list = self.pose_estimator(frame, bboxes)
            for tid, kpts in zip(valid_tids, kpts_list):
                track_kpts[tid] = kpts
                self._last_track_kpts[tid] = kpts
        return track_kpts

    @staticmethod
    def _preprocess_roi(bbox: List[float], frame: np.ndarray) -> np.ndarray:
        """提取并预处理 ROI：保持长宽比 resize + padding 到 96x96."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
        x2, y2 = min(w, x2 + 10), min(h, y2 + 10)
        roi_crop = frame[y1:y2, x1:x2]

        roi_h, roi_w = roi_crop.shape[:2]
        scale = 96.0 / max(roi_h, roi_w)
        new_h, new_w = int(roi_h * scale), int(roi_w * scale)
        roi_resized = cv2.resize(roi_crop, (new_w, new_h))

        roi = np.zeros((96, 96, 3), dtype=np.uint8)
        roi[:new_h, :new_w] = roi_resized
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        return roi

    def _compute_classifier_scores(
        self, frame: np.ndarray, active_tracks: List, track_kpts: Dict[int, np.ndarray]
    ) -> Dict[int, float]:
        """为每个活跃轨迹计算分类器分数."""
        cls_scores: Dict[int, float] = {}
        if self.use_simple_classifier:
            self.classifier.eval()

        for track in active_tracks:
            tid = track.track_id
            kpts = track_kpts.get(tid, np.zeros((17, 3), dtype=np.float32))
            bbox = track.to_tlbr().tolist()
            history = {"centers": list(self._track_history[tid])}
            roi = self._preprocess_roi(bbox, frame)

            if self.use_simple_classifier:
                roi_tensor = torch.from_numpy(roi).unsqueeze(0)  # (1, 3, 96, 96)
                if self.device:
                    roi_tensor = roi_tensor.to(self.device)
                with torch.no_grad():
                    logits = self.classifier(roi_tensor)
                    probs = F.softmax(logits, dim=1)
                fall_class_idx = getattr(self.classifier, 'fall_class_idx', 1)
                s_cls = float(probs[0, fall_class_idx])
            else:
                motion = self._extract_motion(tid, kpts, bbox, history)
                s_cls = self.classifier(roi, kpts, motion)

            cls_scores[tid] = s_cls
        return cls_scores

    def _update_track_history(self, active_tracks: List) -> None:
        """更新活跃轨迹的中心点历史."""
        for track in active_tracks:
            tid = track.track_id
            tlwh = track.to_tlwh()
            cx = tlwh[0] + tlwh[2] / 2.0
            cy = tlwh[1] + tlwh[3] / 2.0
            self._track_history[tid].append((cx, cy))

    def _process_tracks(
        self, active_tracks: List, track_kpts: Dict[int, np.ndarray], cls_scores: Dict[int, float]
    ) -> Dict[str, Any]:
        """执行规则判定、融合决策并组装结果."""
        track_scores: Dict[int, dict] = {}
        track_falling: Dict[int, bool] = {}
        new_alarms: List[int] = []

        for track in active_tracks:
            tid = track.track_id
            kpts = track_kpts.get(tid, np.zeros((17, 3), dtype=np.float32))
            bbox = track.to_tlbr().tolist()
            history = {"centers": list(self._track_history[tid])}
            s_cls = cls_scores.get(tid, 0.0)
            s_rule, flags, rule_debug = self.rule_engine.evaluate(kpts, bbox, history, cls_score=s_cls)

            if tid not in self.fusion:
                self.fusion[tid] = FusionDecision(self.cfg.get("fusion", {}), fps=self.fps)

            posture = rule_debug.get("posture", "unknown")
            self.fusion[tid].update(rule_score=s_rule, cls_score=s_cls, posture=posture)
            is_falling = self.fusion[tid].decide()
            should_alarm = self.fusion[tid].should_alarm()
            state = self.fusion[tid].get_state()

            # 记录融合分数历史（用于可视化时序）
            self._fusion_score_history[tid].append(state["S_final"])

            track_scores[tid] = {
                "rule": s_rule,
                "cls": s_cls,
                "final": state["S_final"],
                "state": state["state"],
                "flags": flags,
                "debug": rule_debug,
            }
            track_falling[tid] = is_falling
            if should_alarm:
                new_alarms.append(tid)

        return {
            "tracks": active_tracks,
            "track_kpts": track_kpts,
            "track_scores": track_scores,
            "track_falling": track_falling,
            "new_alarms": new_alarms,
            "fusion_histories": dict(self._fusion_score_history),
        }

    def _extract_motion(
        self, tid: int, kpts: np.ndarray, bbox: List[float], history: Dict[str, Any]
    ) -> np.ndarray:
        """提取 8-d 运动特征."""
        centers = history.get("centers", [])
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        if len(centers) >= self.motion_window_frames + 2:
            recent = centers[-self.motion_window_frames:]
            arr = np.array(recent, dtype=np.float32)
            vx = arr[-1, 0] - arr[-2, 0]
            vy = arr[-1, 1] - arr[-2, 1]
            # 加速度用二阶差分近似
            ax = (arr[-1, 0] - arr[-2, 0]) - (arr[-2, 0] - arr[-3, 0])
            ay = (arr[-1, 1] - arr[-2, 1]) - (arr[-2, 1] - arr[-3, 1])
        else:
            vx = vy = ax = ay = 0.0

        # H_ratio (防止关键点全不可见导致 NaN)
        head_vals = [kpts[i, 1] for i in [0, 1, 2] if kpts[i, 2] > 0.1]
        ankle_vals = [kpts[i, 1] for i in [15, 16] if kpts[i, 2] > 0.1]
        head_y = np.mean(head_vals) if head_vals else bbox[1] + h * 0.1
        ankle_y = np.mean(ankle_vals) if ankle_vals else bbox[3] - h * 0.1
        h_ratio = abs(head_y - ankle_y) / max(1.0, h)

        # N_ground
        ground_y_thresh = bbox[1] + (1.0 - 0.15) * h
        n_ground = int(np.sum((kpts[:, 2] > 0.1) & (kpts[:, 1] >= ground_y_thresh)))

        return np.array([vx, vy, ax, ay, w, h, h_ratio, n_ground], dtype=np.float32)
