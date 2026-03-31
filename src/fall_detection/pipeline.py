from typing import Dict, List, Any
from collections import defaultdict, deque
import yaml
import numpy as np
import cv2

from fall_detection.detector import PersonDetector
from fall_detection.tracker import ByteTrackLite, Detection
from fall_detection.pose_estimator import PoseEstimator
from fall_detection.rules import RuleEngine
from fall_detection.classifier import FallClassifier
from fall_detection.fusion import FusionDecision


class FallDetectionPipeline:
    """端到端跌倒检测 Pipeline."""

    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        det_cfg = self.cfg.get("detector", {})
        track_cfg = self.cfg.get("tracker", {})
        rules_cfg = self.cfg.get("rules", {})
        fusion_cfg = self.cfg.get("fusion", {})
        pipe_cfg = self.cfg.get("pipeline", {})

        self.detector = PersonDetector(model_name="yolov8n")
        self.detector_conf_thresh = det_cfg.get("conf_thresh", 0.3)

        self.tracker = ByteTrackLite(
            track_thresh=track_cfg.get("track_thresh", 0.5),
            match_thresh=track_cfg.get("match_thresh", 0.8),
            max_age=track_cfg.get("max_age", 30),
            min_hits=track_cfg.get("min_hits", 3),
        )

        self.pose_estimator = PoseEstimator(model_name="yolov8n-pose")
        self.rule_engine = RuleEngine(rules_cfg)
        self.classifier = FallClassifier()
        self.fusion = {}  # track_id -> FusionDecision

        self.skip_frames = pipe_cfg.get("skip_frames", 2)
        self.fps = pipe_cfg.get("fps", 25)
        self.motion_window_frames = max(1, int(0.5 * self.fps))
        self.history_seconds = 1.5
        self.history_maxlen = max(1, int(self.history_seconds * self.fps))
        self.trigger_thresh = rules_cfg.get("trigger_thresh", 0.6)

        self._frame_counter = 0
        self._track_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.history_maxlen)
        )
        # 缓存上一帧活跃 tracks，用于抽帧补位
        self._last_active_tracks: List = []
        self._last_track_kpts: Dict[int, np.ndarray] = {}

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        run_detection = (self._frame_counter % (self.skip_frames + 1)) == 0
        self._frame_counter += 1

        if run_detection:
            det_results = self.detector(frame, conf_thresh=self.detector_conf_thresh)
            detections = [
                Detection(d["bbox"], d["conf"]) for d in det_results
            ]
            active_tracks = self.tracker.update(detections)
        else:
            # 只预测已有轨迹，不跑检测和 pose
            for track in self.tracker.tracks:
                track.predict()
            active_tracks = self.tracker.tracks
            # 非检测帧复用上一次的 kpts
            out = {
                "tracks": active_tracks,
                "track_kpts": self._last_track_kpts,
                "track_scores": {},
                "track_falling": {},
            }
            for t in active_tracks:
                tid = t.track_id
                if tid not in self.fusion:
                    self.fusion[tid] = FusionDecision(self.cfg.get("fusion", {}), fps=self.fps)
            self._last_active_tracks = active_tracks
            return out

        # ---- 3. 关键点估计 ----
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

        # ---- 更新历史 ----
        for track in active_tracks:
            tid = track.track_id
            tlwh = track.to_tlwh()
            cx = tlwh[0] + tlwh[2] / 2.0
            cy = tlwh[1] + tlwh[3] / 2.0
            self._track_history[tid].append((cx, cy))

        # ---- 4. 规则引擎 + 5. 分类器 + 6. 融合决策 ----
        track_scores: Dict[int, dict] = {}
        track_falling: Dict[int, bool] = {}

        for track in active_tracks:
            tid = track.track_id
            kpts = track_kpts.get(tid, np.zeros((17, 3), dtype=np.float32))
            bbox = track.to_tlbr().tolist()
            history = {"centers": list(self._track_history[tid])}
            s_rule, flags = self.rule_engine.evaluate(kpts, bbox, history)

            if s_rule >= self.trigger_thresh:
                # 提取 ROI
                x1, y1, x2, y2 = map(int, bbox)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                roi = cv2.resize(frame[y1:y2, x1:x2], (96, 96))
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0

                # motion 特征
                motion = self._extract_motion(tid, kpts, bbox, history)
                s_cls = self.classifier(roi, kpts, motion)
            else:
                s_cls = 0.0

            if tid not in self.fusion:
                self.fusion[tid] = FusionDecision(self.cfg.get("fusion", {}), fps=self.fps)

            self.fusion[tid].update(rule_score=s_rule, cls_score=s_cls)
            is_falling = self.fusion[tid].decide()
            state = self.fusion[tid].get_state()

            track_scores[tid] = {
                "rule": s_rule,
                "cls": s_cls,
                "final": state["S_final"],
            }
            track_falling[tid] = is_falling

        self._last_active_tracks = active_tracks

        return {
            "tracks": active_tracks,
            "track_kpts": track_kpts,
            "track_scores": track_scores,
            "track_falling": track_falling,
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

        # H_ratio
        head_y = np.mean([kpts[i, 1] for i in [0, 1, 2] if kpts[i, 2] > 0.1])
        ankle_y = np.mean([kpts[i, 1] for i in [15, 16] if kpts[i, 2] > 0.1])
        h_ratio = abs(head_y - ankle_y) / max(1.0, h)

        # N_ground
        ground_y_thresh = bbox[1] + (1.0 - 0.15) * h
        n_ground = int(np.sum((kpts[:, 2] > 0.1) & (kpts[:, 1] >= ground_y_thresh)))

        return np.array([vx, vy, ax, ay, w, h, h_ratio, n_ground], dtype=np.float32)
