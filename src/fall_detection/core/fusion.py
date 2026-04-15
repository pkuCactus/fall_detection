from typing import Dict, Any, Optional
from collections import deque
from enum import Enum


class FallState(Enum):
    """跌倒检测状态机."""
    NORMAL = "normal"
    SUSPECTED = "suspected"
    FALLING = "falling"
    ALARM_SENT = "alarm_sent"
    RECOVERING = "recovering"


class FusionDecision:
    """融合决策器：规则分 + 分类器分 + 时序平滑 + 姿态序列 + 状态机防抖动."""

    def __init__(
        self,
        config: Dict[str, Any] = None,
        fps: int = 25,
        alpha: float = 0.35,
        beta: float = 0.45,
        gamma: float = 0.20,
        alarm_thresh: float = 0.70,
        alarm_min_frames: int = 5,
        sequence_check_frames: int = 8,
        reset_seconds: float = 3.0,
        cooldown_seconds: float = 5.0,
        recovery_seconds: float = 2.0,
        cls_bypass_thresh: float = 1.0,
    ):
        cfg = config or {}
        self.alpha = cfg.get("alpha", alpha)
        self.beta = cfg.get("beta", beta)
        self.gamma = cfg.get("gamma", gamma)
        self.alarm_thresh = cfg.get("alarm_thresh", alarm_thresh)
        self.alarm_min_frames = cfg.get("alarm_min_frames", alarm_min_frames)
        self.sequence_check_frames = cfg.get("sequence_check_frames", sequence_check_frames)
        self.reset_seconds = cfg.get("reset_seconds", reset_seconds)
        self.cooldown_frames = int(cfg.get("cooldown_seconds", cooldown_seconds) * fps)
        self.recovery_frames = int(cfg.get("recovery_seconds", recovery_seconds) * fps)
        self.cls_bypass_thresh = cfg.get("cls_bypass_thresh", cls_bypass_thresh)
        self.fps = fps

        self._cls_history = deque(maxlen=max(5, fps))
        self._posture_history = deque(maxlen=max(sequence_check_frames, fps))
        self._alarm_frames = 0
        self._miss_frames = 0
        self._recovery_frames = 0
        self._cooldown_counter = 0

        self._state = FallState.NORMAL
        self._S_final = 0.0
        self._S_temporal = 0.0
        self._should_alarm = False

    def update(self, rule_score: float, cls_score: float, posture: str = "unknown"):
        """更新一帧融合得分，驱动状态机."""
        self._should_alarm = False

        self._cls_history.append(cls_score)
        self._posture_history.append(posture)
        self._S_temporal = sum(self._cls_history) / len(self._cls_history)
        self._S_final = (
            self.alpha * rule_score
            + self.beta * cls_score
            + self.gamma * self._S_temporal
        )

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        is_above_thresh = self._S_final >= self.alarm_thresh

        if self._state == FallState.NORMAL:
            if is_above_thresh:
                self._alarm_frames = 1
                self._miss_frames = 0
                self._state = FallState.SUSPECTED

        elif self._state == FallState.SUSPECTED:
            if is_above_thresh:
                self._alarm_frames += 1
                self._miss_frames = 0
                cls_bypass = cls_score >= self.cls_bypass_thresh
                min_frames = max(2, self.alarm_min_frames // 2) if cls_bypass else self.alarm_min_frames
                if self._alarm_frames >= min_frames:
                    # 增加姿态序列一致性检查：必须有站/坐 -> 跌的转换
                    # 分类器高置信度时可绕过该检查
                    if cls_bypass or self._check_fall_sequence():
                        self._state = FallState.FALLING
                    # 序列不匹配时继续停留在 SUSPECTED 观察，不贸然跌落也不回 NORMAL
            else:
                self._miss_frames += 1
                if self._miss_frames >= int(self.reset_seconds * self.fps):
                    self._alarm_frames = 0
                    self._miss_frames = 0
                    self._state = FallState.NORMAL

        elif self._state == FallState.FALLING:
            self._should_alarm = True
            self._state = FallState.ALARM_SENT
            self._cooldown_counter = self.cooldown_frames

        elif self._state == FallState.ALARM_SENT:
            if not is_above_thresh:
                self._miss_frames += 1
                if self._miss_frames >= int(self.reset_seconds * self.fps):
                    self._state = FallState.RECOVERING
                    self._recovery_frames = 0
            else:
                self._miss_frames = 0

        elif self._state == FallState.RECOVERING:
            if is_above_thresh:
                self._miss_frames = 0
                self._state = FallState.ALARM_SENT
            else:
                self._recovery_frames += 1
                if self._recovery_frames >= self.recovery_frames:
                    self._alarm_frames = 0
                    self._miss_frames = 0
                    self._state = FallState.NORMAL

    def _check_fall_sequence(self) -> bool:
        """
        检查姿态历史是否包含"站/坐 -> 跌"的转换序列.
        要求：最近 sequence_check_frames 帧内，既有站立/坐姿，又有倒下/躺卧姿态.
        """
        if len(self._posture_history) < self.sequence_check_frames:
            return False

        recent = list(self._posture_history)[-self.sequence_check_frames:]
        upright_postures = {"standing", "sitting"}
        fall_postures = {"crouching", "lying"}

        has_upright = any(p in upright_postures for p in recent)
        has_fall = any(p in fall_postures for p in recent)
        # 进一步约束：最近一帧最好是跌倒姿态或分值仍高
        current_is_fall = recent[-1] in fall_postures

        return has_upright and has_fall and current_is_fall

    def decide(self) -> bool:
        # 只在FALLING和ALARM_SENT返回True，RECOVERING是恢复期不应算跌倒
        return self._state in (FallState.FALLING, FallState.ALARM_SENT)

    def should_alarm(self) -> bool:
        return self._should_alarm

    def get_state(self) -> Dict[str, Any]:
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
        self._state = FallState.NORMAL
        self._alarm_frames = 0
        self._miss_frames = 0
        self._recovery_frames = 0
        self._cooldown_counter = 0
        self._should_alarm = False
        self._cls_history.clear()
        self._posture_history.clear()
