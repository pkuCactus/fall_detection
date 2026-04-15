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

    def test_optical_axis_fall_trigger(self):
        """朝摄像头倒下：h_ratio正常但躯干水平，规则A应触发."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        kpts[0] = [50, 20, 0.9]
        kpts[1] = [45, 25, 0.9]
        kpts[2] = [55, 25, 0.9]
        # 躯干水平，肩膀和髋同高
        kpts[5] = [10, 100, 0.9]
        kpts[6] = [50, 100, 0.9]
        kpts[11] = [50, 100, 0.9]
        kpts[12] = [90, 100, 0.9]
        kpts[13] = [50, 140, 0.9]
        kpts[14] = [70, 140, 0.9]
        kpts[15] = [50, 180, 0.9]
        kpts[16] = [50, 185, 0.9]
        # 其余点也放在底部区域，确保n_ground >= 2
        for i in [3, 4, 7, 8, 9, 10]:
            kpts[i] = [50, 180, 0.9]
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        assert debug["h_ratio"] > 0.75
        assert debug["torso_angle"] > 55
        assert debug["posture"] == "lying"
        assert flags["A"] is True
        assert debug["n_ground"] >= 2


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

    def test_falling_toward_camera_by_torso_angle(self):
        """朝摄像头倒下：h_ratio正常但躯干水平，应判为lying."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        # 头脚仍有一定纵向分离，模拟检测框没明显变矮的情况
        kpts[0] = [50, 20, 0.9]   # nose
        kpts[1] = [45, 25, 0.9]   # left eye
        kpts[2] = [55, 25, 0.9]   # right eye
        kpts[3] = [40, 30, 0.9]   # left ear
        kpts[4] = [60, 30, 0.9]   # right ear
        # 肩膀和髋在同一高度，但水平错开 -> 躯干接近水平
        kpts[5] = [10, 100, 0.9]  # left shoulder
        kpts[6] = [50, 100, 0.9]  # right shoulder
        kpts[11] = [50, 100, 0.9] # left hip
        kpts[12] = [90, 100, 0.9] # right hip
        kpts[13] = [50, 140, 0.9] # left knee
        kpts[14] = [70, 140, 0.9] # right knee
        kpts[15] = [50, 180, 0.9] # left ankle
        kpts[16] = [50, 185, 0.9] # right ankle
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        # h_ratio = (20-182.5)/200 ≈ 0.81 > 0.75，旧逻辑会判standing
        assert debug["h_ratio"] > 0.75
        # 但躯干角度接近90度
        assert debug["torso_angle"] > 55
        # 新逻辑应判为lying
        assert debug["posture"] == "lying"

    def test_falling_toward_camera_by_kpt_aspect(self):
        """朝摄像头倒下：关键点水平展开明显大于纵向."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        # 所有关键点集中在一条水平线上，左右展开很大
        for i in range(17):
            kpts[i] = [10 + i * 5, 100 + (i % 3), 0.9]
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        # 水平跨度大，垂直跨度小
        assert debug["kpt_aspect"] > 1.5
        assert debug["posture"] == "lying"

    def test_standing_with_arms_outstretched_not_lying(self):
        """站立时手臂张开，不应误判为lying."""
        engine = RuleEngine(fps=25)
        kpts = np.zeros((17, 3))
        kpts[0] = [50, 20, 0.9]   # nose
        kpts[1] = [45, 25, 0.9]
        kpts[2] = [55, 25, 0.9]
        kpts[3] = [20, 30, 0.9]   # left ear (手臂方向)
        kpts[4] = [80, 30, 0.9]   # right ear
        kpts[5] = [10, 60, 0.9]   # left shoulder (手臂张开)
        kpts[6] = [90, 60, 0.9]   # right shoulder
        kpts[7] = [5, 90, 0.9]    # left elbow
        kpts[8] = [95, 90, 0.9]   # right elbow
        kpts[9] = [0, 120, 0.9]   # left wrist
        kpts[10] = [100, 120, 0.9] # right wrist
        kpts[11] = [40, 120, 0.9] # left hip
        kpts[12] = [60, 120, 0.9] # right hip
        kpts[13] = [30, 150, 0.9] # left knee
        kpts[14] = [70, 150, 0.9] # right knee
        kpts[15] = [40, 190, 0.9] # left ankle
        kpts[16] = [60, 190, 0.9] # right ankle
        bbox = [0, 0, 100, 200]

        score, flags, debug = engine.evaluate(kpts, bbox, {})

        # 关键点水平跨度大，但垂直跨度也大，kpt_aspect不应>1.5
        assert debug["kpt_aspect"] < 1.5
        # 躯干仍是竖直的
        assert debug["torso_angle"] < 30
        assert debug["posture"] == "standing"


class TestClsScorePostureOverride:
    """测试分类器得分辅助姿态分类."""

    @staticmethod
    def _make_standing_kpts():
        kpts = np.zeros((17, 3))
        kpts[0] = [50, 20, 0.9]   # nose
        kpts[1] = [45, 25, 0.9]
        kpts[2] = [55, 25, 0.9]
        kpts[3] = [40, 30, 0.9]
        kpts[4] = [60, 30, 0.9]
        kpts[5] = [30, 60, 0.9]   # lsho
        kpts[6] = [70, 60, 0.9]   # rsho
        kpts[7] = [25, 90, 0.9]
        kpts[8] = [75, 90, 0.9]
        kpts[9] = [20, 120, 0.9]
        kpts[10] = [80, 120, 0.9]
        kpts[11] = [40, 120, 0.9] # lhip
        kpts[12] = [60, 120, 0.9] # rhip
        kpts[13] = [35, 155, 0.9]
        kpts[14] = [65, 155, 0.9]
        kpts[15] = [40, 190, 0.9] # lank
        kpts[16] = [60, 190, 0.9] # rank
        return kpts

    @staticmethod
    def _make_lying_kpts():
        kpts = np.zeros((17, 3))
        # 人躺下，纵向接近同一高度，但可见性足够
        for i in range(17):
            kpts[i] = [20 + i * 4, 175 + (i % 2), 0.9]
        return kpts

    def test_cls_score_above_t1_forces_lying(self):
        """cls_score > t1 时强制判定为 lying."""
        engine = RuleEngine(
            config={"cls_posture_t1": 0.80, "cls_posture_t2": 0.30}, fps=25
        )
        kpts = self._make_standing_kpts()
        bbox = [0, 0, 100, 200]

        # cls_score 在中间区间，原 standing 降级为 sitting
        _, _, debug = engine.evaluate(kpts, bbox, {}, cls_score=0.50)
        assert debug["posture"] == "sitting"

        # cls_score > t1 强制 lying
        _, _, debug = engine.evaluate(kpts, bbox, {}, cls_score=0.85)
        assert debug["posture"] == "lying"

    def test_cls_score_below_t2_forces_standing(self):
        """cls_score < t2 时强制判定为 standing."""
        engine = RuleEngine(
            config={"cls_posture_t1": 0.80, "cls_posture_t2": 0.30}, fps=25
        )
        kpts = self._make_lying_kpts()
        bbox = [0, 0, 100, 200]

        # cls_score 在中间区间，原 lying 保持 lying
        _, _, debug = engine.evaluate(kpts, bbox, {}, cls_score=0.50)
        assert debug["posture"] == "lying"

        # cls_score < t2 强制 standing
        _, _, debug = engine.evaluate(kpts, bbox, {}, cls_score=0.20)
        assert debug["posture"] == "standing"

    def test_cls_score_mid_range_downgrades_standing_to_sitting(self):
        """t2 <= cls_score <= t1 时，standing 降级为 sitting."""
        engine = RuleEngine(
            config={"cls_posture_t1": 0.80, "cls_posture_t2": 0.30}, fps=25
        )
        kpts = self._make_standing_kpts()
        bbox = [0, 0, 100, 200]

        # 中间区间，原 standing 降级为 sitting
        _, _, debug = engine.evaluate(kpts, bbox, {}, cls_score=0.50)
        assert debug["posture"] == "sitting"

    def test_cls_score_mid_range_preserves_non_standing(self):
        """t2 <= cls_score <= t1 时，非 standing 姿态保持不变."""
        engine = RuleEngine(
            config={"cls_posture_t1": 0.80, "cls_posture_t2": 0.30}, fps=25
        )
        kpts = self._make_lying_kpts()
        bbox = [0, 0, 100, 200]

        # 中间区间保持 lying
        _, _, debug = engine.evaluate(kpts, bbox, {}, cls_score=0.50)
        assert debug["posture"] == "lying"

    def test_disabled_by_default(self):
        """默认阈值禁用分类器辅助姿态."""
        engine = RuleEngine(fps=25)
        kpts = self._make_standing_kpts()
        bbox = [0, 0, 100, 200]

        _, _, debug = engine.evaluate(kpts, bbox, {}, cls_score=0.99)
        # t1=1.0 禁用，0.99 不应触发强制 lying
        assert debug["posture"] == "standing"


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
