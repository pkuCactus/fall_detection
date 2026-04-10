"""轻量级关键点跟踪器.

提供关键点平滑和跳过帧预测功能，优化边缘设备上的姿态估计稳定性.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2


class SimpleKeypointTracker:
    """简单的关键点跟踪器，支持平滑和预测.

    Features:
    1. EMA平滑: 减少检测噪声
    2. 速度预测: 基于历史速度预测下一帧位置
    3. 光流辅助: 可选的光流跟踪用于跳过帧
    """

    def __init__(
        self,
        n_kpts: int = 17,
        smooth_alpha: float = 0.7,
        velocity_decay: float = 0.9,
        max_history: int = 5,
        use_optical_flow: bool = False,
    ):
        """
        Args:
            n_kpts: 关键点数量 (COCO格式默认17)
            smooth_alpha: EMA平滑系数 (0-1, 越大越跟随最新检测)
            velocity_decay: 速度衰减系数
            max_history: 历史帧缓存数量
            use_optical_flow: 是否使用光流辅助跟踪
        """
        self.n_kpts = n_kpts
        self.smooth_alpha = smooth_alpha
        self.velocity_decay = velocity_decay
        self.max_history = max_history
        self.use_optical_flow = use_optical_flow

        # 每个track的历史关键点和速度
        # {track_id: {'kpts': deque, 'velocity': ndarray, 'last_frame': int}}
        self._track_states: Dict[int, Dict] = {}

        # 光流相关
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None

    def _init_track(self, track_id: int) -> Dict:
        """初始化track状态."""
        return {
            'kpts': deque(maxlen=self.max_history),  # 历史关键点 [(n_kpts, 3), ...]
            'velocity': np.zeros((self.n_kpts, 2), dtype=np.float32),  # 每个关键点的速度
            'last_frame': 0,
            'last_kpts': None,  # 用于光流跟踪
        }

    def _get_track_state(self, track_id: int) -> Dict:
        """获取或创建track状态."""
        if track_id not in self._track_states:
            self._track_states[track_id] = self._init_track(track_id)
        return self._track_states[track_id]

    def _smooth_keypoints(
        self,
        new_kpts: np.ndarray,
        history: deque,
    ) -> np.ndarray:
        """EMA平滑关键点.

        Args:
            new_kpts: 新检测的关键点 (n_kpts, 3) [x, y, conf]
            history: 历史关键点队列

        Returns:
            平滑后的关键点
        """
        if len(history) == 0:
            return new_kpts.copy()

        # 使用最近的历史关键点
        prev_kpts = history[-1]  # (n_kpts, 3)

        # 只对可见关键点进行平滑 (conf > 0)
        mask = (new_kpts[:, 2] > 0) & (prev_kpts[:, 2] > 0)

        smoothed = new_kpts.copy()
        if mask.any():
            # EMA: alpha * new + (1-alpha) * prev
            smoothed[mask, :2] = (
                self.smooth_alpha * new_kpts[mask, :2] +
                (1 - self.smooth_alpha) * prev_kpts[mask, :2]
            )

        return smoothed

    def _update_velocity(
        self,
        state: Dict,
        new_kpts: np.ndarray,
    ) -> np.ndarray:
        """更新关键点速度.

        Args:
            state: track状态
            new_kpts: 新关键点

        Returns:
            当前速度
        """
        if len(state['kpts']) == 0:
            return np.zeros((self.n_kpts, 2), dtype=np.float32)

        prev_kpts = state['kpts'][-1]

        # 计算当前速度
        mask = (new_kpts[:, 2] > 0) & (prev_kpts[:, 2] > 0)
        velocity = np.zeros((self.n_kpts, 2), dtype=np.float32)

        if mask.any():
            velocity[mask] = new_kpts[mask, :2] - prev_kpts[mask, :2]

        # EMA更新速度
        state['velocity'] = (
            self.velocity_decay * state['velocity'] +
            (1 - self.velocity_decay) * velocity
        )

        return state['velocity']

    def _predict_with_velocity(
        self,
        state: Dict,
        n_frames: int = 1,
    ) -> np.ndarray:
        """使用速度预测关键点位置.

        Args:
            state: track状态
            n_frames: 预测多少帧之后的位置

        Returns:
            预测的关键点 (n_kpts, 3)
        """
        if len(state['kpts']) == 0:
            return np.zeros((self.n_kpts, 3), dtype=np.float32)

        last_kpts = state['kpts'][-1].copy()

        # 对可见关键点进行预测
        visible = last_kpts[:, 2] > 0
        if visible.any():
            last_kpts[visible, :2] += state['velocity'][visible] * n_frames

        return last_kpts

    def _track_with_optical_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_kpts: np.ndarray,
    ) -> np.ndarray:
        """使用光流跟踪关键点.

        Args:
            prev_frame: 上一帧 (H, W, 3)
            curr_frame: 当前帧 (H, W, 3)
            prev_kpts: 上一帧关键点 (n_kpts, 3)

        Returns:
            跟踪后的关键点 (n_kpts, 3)
        """
        # 转换为灰度图
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 只对高置信度关键点进行光流跟踪
        visible_mask = prev_kpts[:, 2] > 0.3
        visible_indices = np.where(visible_mask)[0]

        if len(visible_indices) == 0:
            return prev_kpts.copy()

        # 准备点坐标 (N, 1, 2)
        prev_points = prev_kpts[visible_indices, :2].reshape(-1, 1, 2).astype(np.float32)

        # 光流跟踪
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # 构建输出
        tracked_kpts = prev_kpts.copy()

        # 更新跟踪成功的点
        valid_mask = status.flatten() == 1
        valid_indices = visible_indices[valid_mask]

        if len(valid_indices) > 0:
            tracked_kpts[valid_indices, :2] = curr_points[valid_mask].reshape(-1, 2)
            # 略微降低置信度表示这是跟踪而非检测
            tracked_kpts[valid_indices, 2] *= 0.95

        return tracked_kpts

    def update(
        self,
        track_id: int,
        kpts: np.ndarray,
        frame_idx: Optional[int] = None,
    ) -> np.ndarray:
        """更新检测到的关键点.

        在检测帧调用，进行平滑并更新历史.

        Args:
            track_id: 跟踪ID
            kpts: 检测到的关键点 (n_kpts, 3) [x, y, conf]
            frame_idx: 当前帧索引 (可选)

        Returns:
            平滑后的关键点
        """
        state = self._get_track_state(track_id)

        # 平滑
        smoothed = self._smooth_keypoints(kpts, state['kpts'])

        # 更新速度
        self._update_velocity(state, smoothed)

        # 保存到历史
        state['kpts'].append(smoothed.copy())
        state['last_kpts'] = smoothed.copy()
        if frame_idx is not None:
            state['last_frame'] = frame_idx

        return smoothed

    def predict(
        self,
        track_id: int,
        curr_frame: Optional[np.ndarray] = None,
        n_frames: int = 1,
    ) -> np.ndarray:
        """预测关键点位置.

        在跳过帧调用，基于速度或光流预测.

        Args:
            track_id: 跟踪ID
            curr_frame: 当前帧图像 (H, W, 3), 用于光流 (可选)
            n_frames: 跳过多少帧

        Returns:
            预测的关键点 (n_kpts, 3)
        """
        state = self._get_track_state(track_id)

        # 使用速度预测
        predicted = self._predict_with_velocity(state, n_frames)

        # 如果有当前帧且启用光流，进行光流修正
        if self.use_optical_flow and curr_frame is not None and state['last_kpts'] is not None:
            # 这里简化处理：使用上一次缓存的帧
            if hasattr(self, '_cached_frame') and self._cached_frame is not None:
                try:
                    flow_tracked = self._track_with_optical_flow(
                        self._cached_frame, curr_frame, state['last_kpts']
                    )
                    # 融合速度预测和光流跟踪结果
                    visible = (predicted[:, 2] > 0) & (flow_tracked[:, 2] > 0)
                    if visible.any():
                        predicted[visible, :2] = (
                            0.5 * predicted[visible, :2] +
                            0.5 * flow_tracked[visible, :2]
                        )
                        predicted[visible, 2] = flow_tracked[visible, 2]
                except Exception:
                    pass

        # 更新缓存
        if curr_frame is not None:
            self._cached_frame = curr_frame.copy()

        return predicted

    def update_frame_cache(self, frame: np.ndarray):
        """更新帧缓存（用于光流）.

        Args:
            frame: 当前帧图像
        """
        self._cached_frame = frame.copy() if self.use_optical_flow else None

    def remove_track(self, track_id: int):
        """移除track状态（当track结束）."""
        if track_id in self._track_states:
            del self._track_states[track_id]

    def clear(self):
        """清空所有状态."""
        self._track_states.clear()
        self._cached_frame = None
