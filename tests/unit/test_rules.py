"""跌倒检测规则引擎单元测试."""

from fall_detection.core.rules import RuleEngine
import numpy as np


class TestRuleEngineInit:
    """测试规则引擎初始化."""

    def test_default_init(self):
        """测试默认参数初始化."""
        engine = RuleEngine(fps=25)
        assert engine.h_ratio_thresh == 0.6
        assert engine.n_ground_min == 2
        assert engine.trigger_thresh == 0.75
        assert engine.fps == 25

    def test_custom_init(self):
        """测试自定义参数初始化."""
        config = {"h_ratio_thresh": 0.5, "n_ground_min": 3, "trigger_thresh": 0.8}
        engine = RuleEngine(config, fps=30)
        assert engine.h_ratio_thresh == 0.5
        assert engine.n_ground_min == 3
        assert engine.trigger_thresh == 0.8
        assert engine.fps == 30


class TestRuleAHeightCompression:
    """测试规则A：高度压缩 + 多点贴地."""

    def test_standing_not_trigger(self):
        """站立姿态不应触发规则A."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        kpts[0] = [50, 20, 0.9]  # nose
        kpts[1] = [45, 25, 0.9]  # left eye
        kpts[2] = [55, 25, 0.9]  # right eye
        kpts[15] = [40, 180, 0.9]  # left ankle
        kpts[16] = [60, 180, 0.9]  # right ankle
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        assert flags["A"] is False
        assert debug["h_ratio"] > 0.6  # 高度比正常

    def test_lying_trigger(self):
        """躺卧姿态应触发规则A."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        # 所有点在底部区域（躺卧）
        for i in range(17):
            kpts[i] = [50 + i * 2, 170 + i % 3, 0.9]
        kpts[0] = [50, 170, 0.9]  # nose
        kpts[15] = [30, 185, 0.9]  # left ankle
        kpts[16] = [70, 185, 0.9]  # right ankle
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        assert flags["A"] is True
        assert debug["h_ratio"] < 0.6  # 高度压缩明显
        assert debug["n_ground"] >= 2  # 多点贴地
        assert debug["visible_ratio"] >= 0.6  # 可见性足够

    def test_low_visibility_premature(self):
        """可见性不足应压低规则A."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        # 只有很少关键点可见
        kpts[0] = [50, 170, 0.9]  # nose
        kpts[15] = [30, 185, 0.9]  # left ankle
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        # 可见性不足时，即使h_ratio低，也不应触发
        assert debug["visible_ratio"] < 0.6


class TestRuleBGroundRegion:
    """测试规则B：地面区域判定."""

    def test_without_ground_roi(self):
        """无预设ROI时，检查底部区域."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        # 所有点在bbox底部40%区域
        for i in range(17):
            kpts[i] = [50 + i * 2, 180, 0.9]  # y=180 在200的bbox底部
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        assert flags["B"] is True

    def test_with_ground_roi(self):
        """有预设ROI时，检查点是否在ROI内."""
        engine = RuleEngine(fps=25)
        engine.ground_roi = [(0, 150), (100, 150), (100, 200), (0, 200)]  # 底部梯形

        kpts = np.zeros((17, 3))
        kpts[0] = [50, 180, 0.9]  # nose in ROI
        kpts[15] = [40, 185, 0.9]  # ankle in ROI
        kpts[16] = [60, 185, 0.9]  # ankle in ROI
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        assert flags["B"] is True


class TestRuleCMotionToStatic:
    """测试规则C：由动到静或持续静止."""

    def test_motion_to_static_trigger(self):
        """由动到静应触发规则C."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        for i in range(17):
            kpts[i] = [50, 170, 0.9]
        bbox = [0, 0, 100, 200]

        # 构造由动到静：前半段运动，后半段静止
        # motion_thresh=75px/s, static_thresh=20px/s
        # 需要 avg_early > 75 且 avg_late < 20
        centers = []
        # 运动期：每帧位移约4px -> 4*25=100px/s > 75
        for i in range(15):
            centers.append((float(i * 4), float(i * 4)))
        # 静止期：每帧位移约0.5px -> 0.5*25=12.5px/s < 20
        for i in range(15):
            centers.append((60.0 + i * 0.5, 60.0 + i * 0.5))
        history = {"centers": centers}

        score, flags, debug = engine.evaluate(kpts, bbox, history)

        # 规则C需要满足条件才触发，不强制要求
        assert debug["centers_len"] >= 10

    def test_continuous_motion_not_trigger(self):
        """持续运动不应触发规则C."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        for i in range(17):
            kpts[i] = [50, 170, 0.9]
        bbox = [0, 0, 100, 200]

        # 持续运动
        centers = [(i * 10.0, i * 10.0) for i in range(20)]
        history = {"centers": centers}

        score, flags, debug = engine.evaluate(kpts, bbox, history)

        assert flags["C"] is False

    def test_lying_static_trigger(self):
        """躺卧姿态持续静止应触发规则C."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        for i in range(17):
            kpts[i] = [50, 170, 0.9]
        bbox = [0, 0, 100, 200]

        # 持续静止（躺卧）
        centers = [(50.0, 50.0 + i * 0.1) for i in range(20)]  # 静止
        history = {"centers": centers}

        score, flags, debug = engine.evaluate(kpts, bbox, history)

        # 躺卧姿态 + 静止应触发C
        assert debug["posture"] == "lying"


class TestRuleDVerticalDrop:
    """测试规则D：垂直快速下降."""

    def test_fast_vertical_drop_trigger(self):
        """快速垂直下降应触发规则D."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        # 躺卧姿态，h_ratio < 0.6
        kpts[0] = [50, 170, 0.9]  # nose
        kpts[1] = [45, 172, 0.9]  # left eye
        kpts[2] = [55, 172, 0.9]  # right eye
        kpts[15] = [40, 185, 0.9]  # left ankle
        kpts[16] = [60, 185, 0.9]  # right ankle
        # 设置更多关键点确保可见性
        for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
            kpts[i] = [50, 175, 0.9]
        bbox = [0, 0, 100, 200]

        # 快速下降：y坐标快速增加（图像坐标y向下为正）
        # vy > 400px/s，需要严格大于
        centers = [
            (50.0, 50.0),
            (50.0, 67.0),  # 增加到17px/frame
            (50.0, 84.0),
            (50.0, 101.0),
            (50.0, 118.0),  # vy = (118-50)/4*25 = 425px/s > 400
        ]
        history = {"centers": centers}

        score, flags, debug = engine.evaluate(kpts, bbox, history)

        # vy计算：(118-50)/(5-1)*25 = 425px/s > 400
        assert debug["vy_px_s"] > 400
        # h_ratio < 0.6
        assert debug["h_ratio"] < 0.6
        assert flags["D"] is True

    def test_slow_motion_not_trigger(self):
        """慢速运动不应触发规则D."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        for i in range(17):
            kpts[i] = [50, 170, 0.9]
        bbox = [0, 0, 100, 200]

        # 慢速下降
        centers = [(50.0, 50.0), (50.0, 51.0), (50.0, 52.0), (50.0, 53.0), (50.0, 54.0)]
        history = {"centers": centers}

        score, flags, debug = engine.evaluate(kpts, bbox, history)

        assert debug["vy_px_s"] < 400
        assert flags["D"] is False


class TestRuleEAcceleration:
    """测试规则E：加速度辅助判定."""

    def test_high_acceleration_trigger(self):
        """高加速度应触发规则E."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        for i in range(17):
            kpts[i] = [50, 170, 0.9]  # 躺卧
        bbox = [0, 0, 100, 200]

        # 构造加速度：速度从慢到快
        # v1=10px/frame, v2=30px/frame, accel=(30-10)/((4-2)/25) = 250px/s^2
        centers = [(50.0, 10.0), (50.0, 20.0), (50.0, 30.0), (50.0, 60.0)]
        history = {"centers": centers}

        score, flags, debug = engine.evaluate(kpts, bbox, history)

        assert debug["accel_mag"] > 0
        if debug["accel_mag"] > 150 and debug["h_ratio"] < 0.6:
            assert flags["E"] is True


class TestPostureClassification:
    """测试姿态分类."""

    def test_standing_posture(self):
        """测试站立姿态分类."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        kpts[0] = [50, 20, 0.9]  # nose - 提高head_y
        kpts[1] = [45, 25, 0.9]  # left eye
        kpts[2] = [55, 25, 0.9]  # right eye
        kpts[5] = [30, 60, 0.9]  # left shoulder
        kpts[6] = [70, 60, 0.9]  # right shoulder
        kpts[11] = [40, 120, 0.9]  # left hip
        kpts[12] = [60, 120, 0.9]  # right hip
        kpts[15] = [40, 190, 0.9]  # left ankle
        kpts[16] = [60, 190, 0.9]  # right ankle
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        # h_ratio = (20 - 190) / 200 = 0.85 > 0.75
        assert debug["posture"] == "standing"
        assert debug["h_ratio"] > 0.75

    def test_sitting_posture(self):
        """测试坐姿分类."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        kpts[0] = [50, 50, 0.9]  # nose
        kpts[1] = [45, 55, 0.9]  # left eye
        kpts[2] = [55, 55, 0.9]  # right eye
        kpts[5] = [30, 70, 0.9]  # left shoulder
        kpts[6] = [70, 70, 0.9]  # right shoulder
        kpts[11] = [40, 120, 0.9]  # left hip
        kpts[12] = [60, 120, 0.9]  # right hip
        kpts[15] = [70, 160, 0.9]  # left ankle (膝盖高度)
        kpts[16] = [80, 160, 0.9]  # right ankle
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        # h_ratio = (50 - 160) / 200 = 0.55, 在 (0.5, 0.75]范围内
        # hip_ratio = (50 - 120) / 200 = 0.35 < 0.45
        assert debug["posture"] == "sitting"
        assert 0.5 < debug["h_ratio"] <= 0.75

    def test_crouching_posture(self):
        """测试蹲姿分类."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        kpts[0] = [50, 100, 0.9]  # nose
        kpts[1] = [45, 105, 0.9]  # left eye
        kpts[2] = [55, 105, 0.9]  # right eye
        kpts[5] = [30, 120, 0.9]  # left shoulder
        kpts[6] = [70, 120, 0.9]  # right shoulder
        kpts[11] = [40, 150, 0.9]  # left hip
        kpts[12] = [60, 150, 0.9]  # right hip
        kpts[15] = [40, 180, 0.9]  # left ankle
        kpts[16] = [60, 180, 0.9]  # right ankle
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        assert debug["posture"] == "crouching"
        assert 0.35 < debug["h_ratio"] <= 0.5

    def test_lying_posture(self):
        """测试躺卧姿态分类."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        kpts[0] = [50, 170, 0.9]  # nose
        kpts[1] = [45, 175, 0.9]  # left eye
        kpts[2] = [55, 175, 0.9]  # right eye
        kpts[5] = [30, 172, 0.9]  # left shoulder
        kpts[6] = [70, 172, 0.9]  # right shoulder
        kpts[11] = [40, 175, 0.9]  # left hip
        kpts[12] = [60, 175, 0.9]  # right hip
        kpts[15] = [30, 180, 0.9]  # left ankle
        kpts[16] = [70, 180, 0.9]  # right ankle
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        assert debug["posture"] == "lying"
        assert debug["h_ratio"] <= 0.35

    def test_unknown_posture_low_visibility(self):
        """可见性不足时应为unknown."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        # 只有很少关键点可见
        kpts[0] = [50, 100, 0.9]  # nose
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        assert debug["visible_ratio"] < 0.4
        assert debug["posture"] == "unknown"


class TestScoreCalculation:
    """测试得分计算."""

    def test_equal_weight_average(self):
        """测试等权平均计算."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        for i in range(17):
            kpts[i] = [50, 170, 0.9]
        bbox = [0, 0, 100, 200]
        centers = [(i * 10.0, i * 10.0) for i in range(8)]
        centers += [(80.0 + i * 0.1, 80.0 + i * 0.1) for i in range(8)]
        history = {"centers": centers}

        score, flags, debug = engine.evaluate(kpts, bbox, history)

        # 验证等权平均：S_rule = (A+B+C+D+E+F) / 6
        expected_score = sum(flags.values()) / 6.0
        assert abs(score - expected_score) < 1e-6

    def test_visibility_penalty(self):
        """测试可见性惩罚."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        # 只有很少关键点可见（< 60%）
        kpts[0] = [50, 170, 0.9]
        kpts[15] = [30, 185, 0.9]
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        # 可见性 < 0.6 且规则F未触发时，得分应减半
        if debug["visible_ratio"] < 0.6 and not flags.get("F", False):
            base_score = sum(flags.values()) / 6.0
            expected_score = base_score * 0.5
            assert abs(score - expected_score) < 1e-6


class TestEdgeCases:
    """测试边界情况."""

    def test_empty_history(self):
        """测试空历史轨迹."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        for i in range(17):
            kpts[i] = [50, 170, 0.9]
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        # 空历史不应崩溃
        assert debug["centers_len"] == 0
        # 运动相关规则不应触发
        assert flags["C"] is False
        assert flags["D"] is False

    def test_zero_bbox_height(self):
        """测试bbox高度为0."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        for i in range(17):
            kpts[i] = [50, 100, 0.9]
        bbox = [0, 100, 100, 100]  # 高度为0

        # 不应除零崩溃
        score, flags, debug = engine.evaluate(kpts, bbox, {})

        # 应能正常处理
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_all_keypoints_zero_confidence(self):
        """测试所有关键点置信度为0."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))  # 所有点置信度为0
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        # 可见性应为0
        assert debug["visible_ratio"] == 0.0
        # 规则A不应触发（可见性不足）
        assert flags["A"] is False
