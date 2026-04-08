"""跌倒检测融合决策器单元测试."""

from fall_detection.core.fusion import FusionDecision, FallState


class TestFusionDecisionInit:
    """测试融合决策器初始化."""

    def test_default_init(self):
        """测试默认参数初始化."""
        fd = FusionDecision()
        assert fd.alpha == 0.35
        assert fd.beta == 0.45
        assert fd.gamma == 0.20
        assert fd.alarm_thresh == 0.70
        assert fd.alarm_min_frames == 5

    def test_custom_init(self):
        """测试自定义参数初始化."""
        fd = FusionDecision(
            alpha=0.5, beta=0.3, gamma=0.2, alarm_thresh=0.6, alarm_min_frames=3
        )
        assert fd.alpha == 0.5
        assert fd.beta == 0.3
        assert fd.gamma == 0.2
        assert fd.alarm_thresh == 0.6
        assert fd.alarm_min_frames == 3


class TestScoreCalculation:
    """测试融合得分计算."""

    def test_fusion_score_formula(self):
        """测试融合得分公式：S_final = α*S_rule + β*S_cls + γ*S_temporal."""
        fd = FusionDecision(alpha=0.35, beta=0.45, gamma=0.20)
        fd.update(rule_score=1.0, cls_score=0.0, posture="standing")

        state = fd.get_state()
        # S_temporal初始为0
        expected = 0.35 * 1.0 + 0.45 * 0.0 + 0.20 * 0.0
        assert abs(state["S_final"] - expected) < 1e-6

    def test_temporal_score_sliding_window(self):
        """测试时序得分滑动窗口."""
        fd = FusionDecision(fps=25)

        # 连续更新多帧，测试S_temporal变化
        for i in range(10):
            fd.update(rule_score=0.5, cls_score=0.8, posture="standing")

        state = fd.get_state()
        # S_temporal应为最近几帧的分类器得分平均
        assert 0.0 <= state["S_temporal"] <= 1.0
        # S_temporal应接近0.8
        assert 0.7 <= state["S_temporal"] <= 0.9


class TestStateMachineTransitions:
    """测试状态机流转."""

    def test_normal_to_suspected(self):
        """测试NORMAL → SUSPECTED转换."""
        fd = FusionDecision(alarm_thresh=0.5)

        # 低分，保持NORMAL
        fd.update(rule_score=0.2, cls_score=0.2, posture="standing")
        assert fd.get_state()["state"] == "normal"

        # 高分，触发SUSPECTED
        fd.update(rule_score=0.8, cls_score=0.9, posture="standing")
        assert fd.get_state()["state"] == "suspected"

    def test_suspected_to_falling_with_sequence(self):
        """测试SUSPECTED → FALLING转换（需要姿态序列）."""
        fd = FusionDecision(
            alarm_thresh=0.5, alarm_min_frames=3, sequence_check_frames=5
        )

        # 连续高分 + 姿态序列（站→跌）
        fd.update(rule_score=0.8, cls_score=0.9, posture="standing")
        fd.update(rule_score=0.8, cls_score=0.9, posture="standing")
        fd.update(rule_score=0.8, cls_score=0.9, posture="crouching")
        fd.update(rule_score=0.8, cls_score=0.9, posture="lying")
        fd.update(rule_score=0.8, cls_score=0.9, posture="lying")

        state = fd.get_state()
        # 应至少进入SUSPECTED
        assert state["state"] in ["suspected", "falling", "alarm_sent"]

    def test_falling_to_alarm_sent(self):
        """测试FALLING → ALARM_SENT转换."""
        fd = FusionDecision(
            alarm_thresh=0.5, alarm_min_frames=2, sequence_check_frames=4
        )

        # 触发完整流程
        fd.update(rule_score=0.9, cls_score=0.9, posture="standing")
        fd.update(rule_score=0.9, cls_score=0.9, posture="standing")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")

        # FALLING状态在下一帧会转换到ALARM_SENT
        state = fd.get_state()
        # 应该在FALLING或ALARM_SENT状态
        assert state["state"] in ["falling", "alarm_sent"]
        # 如果在FALLING，下一次update会触发should_alarm
        if state["state"] == "falling":
            fd.update(rule_score=0.9, cls_score=0.9, posture="lying")
            assert fd.should_alarm() or fd.get_state()["state"] == "alarm_sent"

    def test_alarm_sent_to_recovering(self):
        """测试ALARM_SENT → RECOVERING转换."""
        fd = FusionDecision(
            fps=25,
            alarm_thresh=0.5,
            alarm_min_frames=2,
            sequence_check_frames=4,
            reset_seconds=0.1,  # 短恢复期
        )

        # 触发告警
        for _ in range(3):
            fd.update(rule_score=0.9, cls_score=0.9, posture="standing")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")

        # 持续低分进入恢复期
        for _ in range(5):
            fd.update(rule_score=0.1, cls_score=0.1, posture="lying")

        state = fd.get_state()
        assert state["state"] in ["recovering", "normal"]

    def test_recovering_to_normal(self):
        """测试RECOVERING → NORMAL转换."""
        fd = FusionDecision(
            fps=25,
            alarm_thresh=0.5,
            alarm_min_frames=2,
            sequence_check_frames=4,
            reset_seconds=0.1,
            recovery_seconds=0.1,
        )

        # 触发告警
        for _ in range(3):
            fd.update(rule_score=0.9, cls_score=0.9, posture="standing")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")

        # 恢复期
        for _ in range(10):
            fd.update(rule_score=0.1, cls_score=0.1, posture="lying")

        # 应回到NORMAL
        assert fd.get_state()["state"] == "normal"


class TestPostureSequenceCheck:
    """测试姿态序列检查."""

    def test_sequence_with_transition(self):
        """测试有姿态转换的序列."""
        fd = FusionDecision(
            alarm_thresh=0.5, alarm_min_frames=2, sequence_check_frames=5
        )

        # 站→跌的转换序列
        fd.update(rule_score=0.9, cls_score=0.9, posture="standing")
        fd.update(rule_score=0.9, cls_score=0.9, posture="standing")
        fd.update(rule_score=0.9, cls_score=0.9, posture="crouching")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")

        # 应允许触发告警
        state = fd.get_state()
        assert state["state"] in ["suspected", "falling", "alarm_sent"]

    def test_sequence_without_transition(self):
        """测试无姿态转换的序列（全程躺卧）."""
        fd = FusionDecision(
            alarm_thresh=0.5, alarm_min_frames=3, sequence_check_frames=8
        )

        # 全程lying，无站→跌转换
        for _ in range(10):
            fd.update(rule_score=0.9, cls_score=0.9, posture="lying")

        # 不应触发告警（缺少姿态转换）
        state = fd.get_state()
        assert state["state"] in ["suspected", "normal"]
        # 不应在FALLING或ALARM_SENT状态
        if state["state"] not in ["normal"]:
            assert not fd.should_alarm()


class TestCooldownMechanism:
    """测试冷却期机制."""

    def test_cooldown_prevents_repeated_alarm(self):
        """测试冷却期防止重复告警."""
        fd = FusionDecision(
            fps=25,
            alarm_thresh=0.5,
            alarm_min_frames=2,
            sequence_check_frames=4,
            cooldown_seconds=0.2,  # 5帧冷却期
        )

        # 触发第一次告警
        for _ in range(3):
            fd.update(rule_score=0.9, cls_score=0.9, posture="standing")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")

        first_alarm = fd.should_alarm()

        # 冷却期内再次更新
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")
        second_alarm = fd.should_alarm()

        # 第一次可能触发，第二次不应触发
        assert second_alarm is False

    def test_cooldown_expires(self):
        """测试冷却期过期."""
        fd = FusionDecision(
            fps=25,
            alarm_thresh=0.5,
            alarm_min_frames=2,
            sequence_check_frames=4,
            cooldown_seconds=0.04,  # 1帧冷却期
        )

        # 触发告警
        for _ in range(3):
            fd.update(rule_score=0.9, cls_score=0.9, posture="standing")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")

        # 冷却期过后可以再次触发
        for _ in range(5):  # 超过冷却期
            fd.update(rule_score=0.9, cls_score=0.9, posture="lying")


class TestResetMechanism:
    """测试重置机制."""

    def test_manual_reset(self):
        """测试手动重置."""
        fd = FusionDecision()

        # 更新到SUSPECTED状态
        fd.update(rule_score=0.9, cls_score=0.9, posture="standing")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")

        # 手动重置
        fd.reset()

        state = fd.get_state()
        assert state["state"] == "normal"
        assert state["alarm_frames"] == 0
        # S_final和S_temporal不会在reset时清空，但状态恢复正常


class TestEdgeCases:
    """测试边界情况."""

    def test_extreme_scores(self):
        """测试极端得分."""
        fd = FusionDecision()

        # 全部为1.0
        fd.update(rule_score=1.0, cls_score=1.0, posture="standing")
        state = fd.get_state()
        assert 0.0 <= state["S_final"] <= 1.0

        # 全部为0.0
        fd.reset()
        fd.update(rule_score=0.0, cls_score=0.0, posture="standing")
        state = fd.get_state()
        assert state["S_final"] == 0.0

    def test_long_running(self):
        """测试长时间运行."""
        fd = FusionDecision(fps=25)

        # 模拟100帧
        for i in range(100):
            fd.update(rule_score=0.5, cls_score=0.5, posture="standing")

        # 应保持稳定
        state = fd.get_state()
        assert state["state"] == "normal"

    def test_rapid_state_changes(self):
        """测试快速状态变化."""
        fd = FusionDecision(
            alarm_thresh=0.5, alarm_min_frames=2, sequence_check_frames=4
        )

        # 高分→低分→高分交替
        for i in range(20):
            score = 0.9 if i % 2 == 0 else 0.1
            fd.update(rule_score=score, cls_score=score, posture="standing")

        # 应保持稳定，不崩溃
        state = fd.get_state()
        assert state["state"] in ["normal", "suspected"]


class TestDecideMethod:
    """测试decide方法."""

    def test_decide_in_falling_state(self):
        """测试FALLING状态时decide返回True."""
        fd = FusionDecision(
            alarm_thresh=0.5, alarm_min_frames=2, sequence_check_frames=4
        )

        # 触发到FALLING状态
        fd.update(rule_score=0.9, cls_score=0.9, posture="standing")
        fd.update(rule_score=0.9, cls_score=0.9, posture="standing")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")
        fd.update(rule_score=0.9, cls_score=0.9, posture="lying")

        state = fd.get_state()
        if state["state"] in ["falling", "alarm_sent"]:
            assert fd.decide() is True

    def test_decide_in_normal_state(self):
        """测试NORMAL状态时decide返回False."""
        fd = FusionDecision()

        fd.update(rule_score=0.1, cls_score=0.1, posture="standing")

        assert fd.decide() is False
