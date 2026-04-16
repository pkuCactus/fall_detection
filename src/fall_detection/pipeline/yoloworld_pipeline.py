from typing import Dict, List, Any
from collections import defaultdict, deque
import yaml
import numpy as np

from fall_detection.core import PersonDetector, ByteTrackLite, Detection, FusionDecision
from fall_detection.utils.common import normalize_device
from fall_detection.utils.geometry import iou


class YOLOWorldFallPipeline:
    """基于 YOLO-World 多姿态检测的端到端跌倒检测 Pipeline.

    该 Pipeline 不使用姿态估计器和复杂分类器，而是直接利用
    YOLO-World 开放词汇检测的多个姿态类别，结合跟踪和时序融合
    进行跌倒判定。
    """

    def __init__(self, config_path: str = "configs/pipeline/yolo_world.yaml", device: str = None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.device = normalize_device(device)

        det_cfg = self.cfg.get("detector", {})
        yw_cfg = self.cfg.get("yolo_world_fall", {})
        track_cfg = self.cfg.get("tracker", {})
        pipe_cfg = self.cfg.get("pipeline", {})

        classes = yw_cfg.get("classes")
        model_path = det_cfg.get("model_path")
        det_imgsz = det_cfg.get("imgsz")
        self.detector = PersonDetector(
            model_path=model_path,
            classes=classes,
            device=device,
            model_type="yolo_world",
            imgsz=det_imgsz,
        )
        self.detector_conf_thresh = det_cfg.get("conf_thresh", 0.3)

        self.tracker = ByteTrackLite(
            track_thresh=track_cfg.get("track_thresh", 0.5),
            match_thresh=track_cfg.get("match_thresh", 0.8),
            max_age=track_cfg.get("max_age", 30),
            min_hits=track_cfg.get("min_hits", 3),
        )

        self.skip_frames = pipe_cfg.get("skip_frames", 2)
        self.fps = pipe_cfg.get("fps", 25)

        self.posture_map = yw_cfg.get("posture_map", {})
        self.fall_scores = yw_cfg.get("fall_scores", {})
        self.motion_window_frames = max(1, int(0.5 * self.fps))
        self.history_maxlen = max(1, int(1.5 * self.fps))

        self.fusion_cfg = self.cfg.get("fusion", {})
        self.fusion = {}  # track_id -> FusionDecision

        self._frame_counter = 0
        self._track_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.history_maxlen)
        )
        self._last_det_info: Dict[int, dict] = {}
        self._fusion_score_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        run_detection = (self._frame_counter % (self.skip_frames + 1)) == 0
        self._frame_counter += 1

        raw_detections: List = []
        if run_detection:
            det_results = self.detector(
                frame, conf_thresh=self.detector_conf_thresh, filter_class_id=None
            )
            raw_detections = [d["bbox"] for d in det_results]
            detections = [Detection(d["bbox"], d["conf"]) for d in det_results]
            active_tracks = self.tracker.update(detections)
            det_info = self._match_detections_to_tracks(det_results, active_tracks)
            self._last_det_info = det_info
        else:
            for track in self.tracker.tracks:
                track.predict()
            active_tracks = [t for t in self.tracker.tracks if t.time_since_update <= self.skip_frames]
            det_info = {
                tid: self._last_det_info.get(tid, {"class_name": "unknown", "class_id": -1, "conf": 0.0})
                for tid in [t.track_id for t in active_tracks]
            }

        self._update_track_history(active_tracks)
        result = self._process_tracks(active_tracks, det_info, frame)

        result["is_detection_frame"] = run_detection
        result["detections"] = raw_detections
        return result

    def _match_detections_to_tracks(
        self, det_results: List[Dict], active_tracks: List
    ) -> Dict[int, dict]:
        """用 IoU 将检测框的类别信息关联到活跃轨迹."""
        det_info = {}
        for track in active_tracks:
            best_iou = -1.0
            best_info = {"class_name": "unknown", "class_id": -1, "conf": 0.0}
            track_box = track.to_tlbr().tolist()
            for d in det_results:
                i = iou(track_box, d["bbox"])
                if i > best_iou:
                    best_iou = i
                    best_info = {
                        "class_name": d.get("class_name", "unknown"),
                        "class_id": d.get("class_id", -1),
                        "conf": d["conf"],
                    }
            det_info[track.track_id] = best_info
        return det_info

    def _update_track_history(self, active_tracks: List) -> None:
        """更新活跃轨迹的中心点历史."""
        for track in active_tracks:
            tid = track.track_id
            tlwh = track.to_tlwh()
            cx = tlwh[0] + tlwh[2] / 2.0
            cy = tlwh[1] + tlwh[3] / 2.0
            self._track_history[tid].append((cx, cy))

    def _compute_fall_score(
        self, track, det_info: Dict[int, dict], frame: np.ndarray
    ) -> tuple:
        """计算单个轨迹的跌倒得分、姿态和调试信息."""
        tid = track.track_id
        info = det_info.get(tid, {})
        class_name = info.get("class_name", "unknown")
        det_conf = info.get("conf", 0.0)

        # 基础跌倒分（由配置映射）
        base_score = self.fall_scores.get(class_name, 0.0)

        # 运动特征辅助
        history = {"centers": list(self._track_history[tid])}
        motion_bonus = self._compute_motion_bonus(track, history)

        # 宽高比辅助（倒下的人 bbox 通常 wider than tall）
        x1, y1, x2, y2 = track.to_tlbr()
        w, h = x2 - x1, y2 - y1
        aspect = w / max(h, 1.0)
        aspect_bonus = 0.0
        if aspect > 1.2:
            aspect_bonus = min(0.2, (aspect - 1.2) * 0.3)

        # 综合得分，检测置信度作为基础分权重
        score = base_score * det_conf + motion_bonus * 0.1 + aspect_bonus
        score = min(1.0, max(0.0, score))

        posture = self.posture_map.get(class_name, "unknown")
        debug = {
            "posture": posture,
            "class_name": class_name,
            "det_conf": round(det_conf, 3),
            "aspect": round(aspect, 2),
            "motion_bonus": round(motion_bonus, 3),
            "aspect_bonus": round(aspect_bonus, 3),
        }
        return score, posture, debug

    def _compute_motion_bonus(self, track, history: Dict[str, Any]) -> float:
        """基于垂直快速下降给运动加分."""
        centers = history.get("centers", [])
        if len(centers) >= self.motion_window_frames + 2:
            recent = centers[-self.motion_window_frames:]
            arr = np.array(recent, dtype=np.float32)
            vy = arr[-1, 1] - arr[-2, 1]
            if vy > 50:
                return min(0.5, vy / 400.0)
        return 0.0

    def _process_tracks(
        self, active_tracks: List, det_info: Dict[int, dict], frame: np.ndarray
    ) -> Dict[str, Any]:
        """执行姿态-跌倒得分计算、融合决策并组装结果."""
        track_scores: Dict[int, dict] = {}
        track_falling: Dict[int, bool] = {}
        new_alarms: List[int] = []

        for track in active_tracks:
            tid = track.track_id
            fall_score, posture, debug = self._compute_fall_score(track, det_info, frame)

            if tid not in self.fusion:
                self.fusion[tid] = FusionDecision(self.fusion_cfg, fps=self.fps)

            # YOLO-World pipeline 中：rule_score 与 cls_score 均使用 fall_score
            self.fusion[tid].update(rule_score=fall_score, cls_score=fall_score, posture=posture)
            is_falling = self.fusion[tid].decide()
            should_alarm = self.fusion[tid].should_alarm()
            state = self.fusion[tid].get_state()

            self._fusion_score_history[tid].append(state["S_final"])

            track_scores[tid] = {
                "rule": fall_score,
                "cls": fall_score,
                "final": state["S_final"],
                "state": state["state"],
                "flags": {},
                "debug": debug,
            }
            track_falling[tid] = is_falling
            if should_alarm:
                new_alarms.append(tid)

        return {
            "tracks": active_tracks,
            "track_kpts": {},  # 无关键点
            "track_scores": track_scores,
            "track_falling": track_falling,
            "new_alarms": new_alarms,
            "fusion_histories": dict(self._fusion_score_history),
        }
