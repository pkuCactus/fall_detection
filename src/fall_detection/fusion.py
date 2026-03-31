from typing import Dict, Any
from collections import deque


class FusionDecision:
    """融合决策器：规则分 + 分类器分 + 时序平滑."""

    def __init__(
        self,
        config: Dict[str, Any] = None,
        fps: int = 25,
        alpha: float = 0.35,
        beta: float = 0.45,
        gamma: float = 0.20,
        alarm_thresh: float = 0.70,
        alarm_min_frames: int = 5,
        reset_seconds: float = 3.0,
    ):
        cfg = config or {}
        self.alpha = cfg.get("alpha", alpha)
        self.beta = cfg.get("beta", beta)
        self.gamma = cfg.get("gamma", gamma)
        self.alarm_thresh = cfg.get("alarm_thresh", alarm_thresh)
        self.alarm_min_frames = cfg.get("alarm_min_frames", alarm_min_frames)
        self.reset_seconds = cfg.get("reset_seconds", reset_seconds)
        self.fps = fps

        self._cls_history = deque(maxlen=max(5, fps))
        self._alarm_frames = 0
        self._miss_frames = 0
        self._is_falling = False
        self._S_final = 0.0
        self._S_temporal = 0.0

    def update(self, rule_score: float, cls_score: float):
        """更新一帧融合得分."""
        self._cls_history.append(cls_score)
        self._S_temporal = sum(self._cls_history) / len(self._cls_history)
        self._S_final = (
            self.alpha * rule_score
            + self.beta * cls_score
            + self.gamma * self._S_temporal
        )

        if self._S_final >= self.alarm_thresh:
            self._alarm_frames += 1
            self._miss_frames = 0
        else:
            self._miss_frames += 1
            if self._miss_frames >= int(self.reset_seconds * self.fps):
                self._alarm_frames = 0
                self._miss_frames = 0
                self._is_falling = False

        if self._alarm_frames >= self.alarm_min_frames:
            self._is_falling = True

    def decide(self) -> bool:
        """返回当前是否触发跌倒告警."""
        return self._is_falling

    def get_state(self) -> Dict[str, Any]:
        """获取当前内部状态."""
        return {
            "S_final": self._S_final,
            "S_temporal": self._S_temporal,
            "alarm_frames": self._alarm_frames,
            "is_falling": self._is_falling,
        }
