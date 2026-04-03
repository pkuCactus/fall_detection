from dataclasses import dataclass
from typing import List
import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment

from fall_detection.utils.geometry import iou


@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2]
    conf: float
    embed: np.ndarray = None

    @property
    def tlwh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1])


class KalmanFilter:
    """轻量 8 状态卡尔曼滤波器 (x, y, w, h, vx, vy, vw, vh)."""

    NDIM = 4
    DT = 1.0
    STD_WEIGHT_POSITION = 1.0 / 20
    STD_WEIGHT_VELOCITY = 1.0 / 160
    MIN_STD_W = 1e-2
    MIN_STD_VW = 1e-5

    def __init__(self):
        ndim = self.NDIM
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = self.DT
        self._update_mat = np.eye(ndim, 2 * ndim)

    def initiate(self, measurement: np.ndarray) -> tuple:
        mean_pos = measurement[:4]
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self.STD_WEIGHT_POSITION * measurement[3],
            2 * self.STD_WEIGHT_POSITION * measurement[3],
            self.MIN_STD_W,
            2 * self.STD_WEIGHT_POSITION * measurement[3],
            10 * self.STD_WEIGHT_VELOCITY * measurement[3],
            10 * self.STD_WEIGHT_VELOCITY * measurement[3],
            self.MIN_STD_VW,
            10 * self.STD_WEIGHT_VELOCITY * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        std_pos = [
            self.STD_WEIGHT_POSITION * mean[3],
            self.STD_WEIGHT_POSITION * mean[3],
            self.MIN_STD_W,
            self.STD_WEIGHT_POSITION * mean[3],
        ]
        std_vel = [
            self.STD_WEIGHT_VELOCITY * mean[3],
            self.STD_WEIGHT_VELOCITY * mean[3],
            self.MIN_STD_VW,
            self.STD_WEIGHT_VELOCITY * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        std = [
            self.STD_WEIGHT_POSITION * mean[3],
            self.STD_WEIGHT_POSITION * mean[3],
            self.MIN_STD_W,
            self.STD_WEIGHT_POSITION * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> tuple:
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class Track:
    def __init__(self, detection: Detection, track_id: int, max_age: int = 30, min_hits: int = 3):
        self.track_id = track_id
        self.bbox = detection.bbox
        self.conf = detection.conf
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = "tentative"
        self.max_age = max_age
        self.min_hits = min_hits
        # 初始化卡尔曼
        self.kf = KalmanFilter()
        tlwh = detection.tlwh
        cx = tlwh[0] + tlwh[2] / 2.0
        cy = tlwh[1] + tlwh[3] / 2.0
        measurement = np.array([cx, cy, tlwh[2], tlwh[3]])
        self.mean, self.covariance = self.kf.initiate(measurement)

    def predict(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection):
        tlwh = detection.tlwh
        cx = tlwh[0] + tlwh[2] / 2.0
        cy = tlwh[1] + tlwh[3] / 2.0
        measurement = np.array([cx, cy, tlwh[2], tlwh[3]])
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, measurement)
        self.bbox = detection.bbox
        self.conf = detection.conf
        self.hits += 1
        self.time_since_update = 0
        if self.state == "tentative" and self.hits >= self.min_hits:
            self.state = "confirmed"

    def mark_missed(self):
        self.time_since_update += 1
        if self.state == "tentative":
            return False
        elif self.time_since_update > self.max_age:
            return False
        return True

    def to_tlwh(self) -> np.ndarray:
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2.0
        return ret

    def to_tlbr(self) -> np.ndarray:
        tlwh = self.to_tlwh()
        ret = tlwh.copy()
        ret[2:] += ret[:2]
        return ret


def iou_cost(tracks, detections) -> np.ndarray:
    cost_matrix = np.zeros((len(tracks), len(detections)))
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            cost_matrix[i, j] = 1.0 - iou(track.to_tlbr().tolist(), det.bbox)
    return cost_matrix


class ByteTrackLite:
    """裁剪版 ByteTrack，仅用 IoU 匹配，无外观特征."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        # 1. 预测已有轨迹
        for track in self.tracks:
            track.predict()

        # 2. 分离高、低分检测框
        dets_high = [d for d in detections if d.conf >= self.track_thresh]
        dets_low = [d for d in detections if d.conf < self.track_thresh]

        # 3. 第一次匹配：confirmed + tentative tracks + 高分检测框
        active_tracks = [t for t in self.tracks if t.state in ("confirmed", "tentative")]
        matches1, unmatched_tracks1, unmatched_dets1 = self._match(
            active_tracks, dets_high
        )
        for track_idx, det_idx in matches1:
            active_tracks[track_idx].update(dets_high[det_idx])

        # 4. 第二次匹配：未匹配 tracks + 低分检测框
        tracks_pool = [active_tracks[i] for i in unmatched_tracks1]
        matches2, unmatched_tracks2, _ = self._match(tracks_pool, dets_low)
        for track_idx, det_idx in matches2:
            tracks_pool[track_idx].update(dets_low[det_idx])

        # 5. 标记未匹配轨迹为丢失
        matched_tracks = set()
        for track_idx, _ in matches1:
            matched_tracks.add(id(active_tracks[track_idx]))
        for track_idx, _ in matches2:
            matched_tracks.add(id(tracks_pool[track_idx]))

        remaining_tracks = []
        for t in self.tracks:
            if id(t) in matched_tracks:
                remaining_tracks.append(t)
            else:
                if t.mark_missed():
                    remaining_tracks.append(t)
        self.tracks = remaining_tracks

        # 6. 未匹配高分检测框 → 新建 tentative track
        matched_dets = set()
        for _, det_idx in matches1:
            matched_dets.add(id(dets_high[det_idx]))
        for _, det_idx in matches2:
            matched_dets.add(id(dets_low[det_idx]))
        for det in detections:
            if id(det) not in matched_dets and det.conf >= self.track_thresh:
                self.tracks.append(Track(det, self._next_id, self.max_age, self.min_hits))
                self._next_id += 1

        # 重新分配 track_id（保证稳定递增）
        active = [t for t in self.tracks if t.time_since_update == 0]
        # 返回当前活跃的 tracks
        return active

    def _match(self, tracks, detections):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        cost = iou_cost(tracks, detections)
        track_indices, det_indices = linear_sum_assignment(cost)
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        for t, d in zip(track_indices, det_indices):
            if cost[t, d] <= self.match_thresh:
                matches.append((t, d))
                unmatched_tracks.remove(t)
                unmatched_dets.remove(d)
        return matches, unmatched_tracks, unmatched_dets
