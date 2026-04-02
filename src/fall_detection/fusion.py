from typing import Dict, Any, Optional
from collections import deque
from enum import Enum


class FallState(Enum):
    """跌倒检测状态机."""
    NORMAL = "normal"              # 正常状态
    SUSPECTED = "suspected"        # 疑似跌倒（得分超过阈值，但持续时间不够）
    FALLING = "falling"            # 确认跌倒，等待上报
    ALARM_SENT = "alarm_sent"      # 已发送告警，持续监测中
    RECOVERING = "recovering"      # 跌倒结束，恢复期


class FusionDecision:
    """融合决策器：规则分 + 分类器分 + 时序平滑 + 状态机防抖动."""

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
        cooldown_seconds: float = 5.0,  # 告警冷却期，防止频繁上报
        recovery_seconds: float = 2.0,   # 恢复确认期，防止抖动
    ):
        cfg = config or {}
        self.alpha = cfg.get("alpha", alpha)
        self.beta = cfg.get("beta", beta)
        self.gamma = cfg.get("gamma", gamma)
        self.alarm_thresh = cfg.get("alarm_thresh", alarm_thresh)
        self.alarm_min_frames = cfg.get("alarm_min_frames", alarm_min_frames)
        self.reset_seconds = cfg.get("reset_seconds", reset_seconds)
        self.cooldown_frames = int(cfg.get("cooldown_seconds", cooldown_seconds) * fps)
        self.recovery_frames = int(cfg.get("recovery_seconds", recovery_seconds) * fps)
        self.fps = fps

        self._cls_history = deque(maxlen=max(5, fps))
        self._alarm_frames = 0
        self._miss_frames = 0
        self._recovery_frames = 0
        self._cooldown_counter = 0

        # 状态机
        self._state = FallState.NORMAL
        self._S_final = 0.0
        self._S_temporal = 0.0

        # 告警触发标记（每帧重置，外部需检查）
        self._should_alarm = False

    def update(self, rule_score: float, cls_score: float):
        """更新一帧融合得分，驱动状态机."""
        self._should_alarm = False  # 重置告警标记

        # 更新分类器历史和平滑得分
        self._cls_history.append(cls_score)
        self._S_temporal = sum(self._cls_history) / len(self._cls_history)
        self._S_final = (
            self.alpha * rule_score
            + self.beta * cls_score
            + self.gamma * self._S_temporal
        )

        # 冷却期处理
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        # 得分是否超过阈值
        is_above_thresh = self._S_final >= self.alarm_thresh

        # 状态机转移
        if self._state == FallState.NORMAL:
            if is_above_thresh:
                self._alarm_frames = 1
                self._miss_frames = 0
                self._state = FallState.SUSPECTED
            # 否则保持 NORMAL

        elif self._state == FallState.SUSPECTED:
            if is_above_thresh:
                self._alarm_frames += 1
                self._miss_frames = 0
                if self._alarm_frames >= self.alarm_min_frames:
                    self._state = FallState.FALLING
            else:
                self._miss_frames += 1
                if self._miss_frames >= int(self.reset_seconds * self.fps):
                    # 重置回正常
                    self._alarm_frames = 0
                    self._miss_frames = 0
                    self._state = FallState.NORMAL

        elif self._state == FallState.FALLING:
            # 首次确认跌倒，触发告警
            self._should_alarm = True
            self._state = FallState.ALARM_SENT
            self._cooldown_counter = self.cooldown_frames

        elif self._state == FallState.ALARM_SENT:
            # 持续监测跌倒状态
            if not is_above_thresh:
                self._miss_frames += 1
                if self._miss_frames >= int(self.reset_seconds * self.fps):
                    # 进入恢复期
                    self._state = FallState.RECOVERING
                    self._recovery_frames = 0
            else:
                self._miss_frames = 0
            # 冷却期内不上报新的告警

        elif self._state == FallState.RECOVERING:
            # 恢复期：确保跌倒真的结束了
            if is_above_thresh:
                # 又检测到跌倒，回到告警状态（但不触发新告警）
                self._miss_frames = 0
                self._state = FallState.ALARM_SENT
            else:
                self._recovery_frames += 1
                if self._recovery_frames >= self.recovery_frames:
                    # 确认恢复，回到正常状态
                    self._alarm_frames = 0
                    self._miss_frames = 0
                    self._state = FallState.NORMAL

    def decide(self) -> bool:
        """返回当前是否处于跌倒状态（持续状态）."""
        return self._state in (FallState.FALLING, FallState.ALARM_SENT, FallState.RECOVERING)

    def should_alarm(self) -> bool:
        """返回是否应该触发新的告警（每帧只返回一次 True）."""
        return self._should_alarm

    def get_state(self) -> Dict[str, Any]:
        """获取当前内部状态."""
        return {
            "S_final": self._S_final,
            "S_temporal": self._S_temporal,
            "alarm_frames": self._alarm_frames,
            "is_falling": self.decide(),
            "state": self._state.value,
            "should_alarm": self._should_alarm,
            "cooldown_remaining": self._cooldown_counter,
        }

    def reset(self):
        """手动重置状态（例如外部确认了告警后）."""
        self._state = FallState.NORMAL
        self._alarm_frames = 0
        self._miss_frames = 0
        self._recovery_frames = 0
        self._cooldown_counter = 0
        self._should_alarm = False
