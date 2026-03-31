from fall_detection.rules import RuleEngine
import numpy as np


def test_rule_engine_init():
    engine = RuleEngine()
    assert engine.h_ratio_thresh == 0.5
    assert engine.n_ground_min == 3


def test_stand_upright():
    engine = RuleEngine()
    kpts = np.zeros((17, 3))
    # 模拟直立：头在上，脚在下（图像坐标 y 向下为正）
    kpts[0] = [50, 20, 0.9]   # nose
    kpts[1] = [45, 25, 0.9]   # left eye
    kpts[2] = [55, 25, 0.9]   # right eye
    kpts[15] = [40, 180, 0.9] # left ankle
    kpts[16] = [60, 180, 0.9] # right ankle
    bbox = [0, 0, 100, 200]
    history = []
    score, flags = engine.evaluate(kpts, bbox, history)
    assert score < 0.6
    assert flags['A'] is False


def test_fall_detected():
    engine = RuleEngine()
    # 构造平躺姿势：头和脚在同一高度
    kpts = np.zeros((17, 3))
    kpts[0] = [50, 150, 0.9]   # nose
    kpts[1] = [45, 155, 0.9]   # left eye
    kpts[2] = [55, 155, 0.9]   # right eye
    kpts[5] = [30, 160, 0.9]   # left shoulder
    kpts[6] = [70, 160, 0.9]   # right shoulder
    kpts[11] = [20, 170, 0.9]  # left hip
    kpts[12] = [80, 170, 0.9]  # right hip
    kpts[15] = [10, 180, 0.9]  # left ankle
    kpts[16] = [90, 180, 0.9]  # right ankle
    bbox = [0, 0, 100, 200]
    # 模拟由动到静：前半段位移大，后半段几乎不动
    history = {
        'centers': [
            (10.0, 10.0), (12.0, 11.0), (11.0, 12.0),  # 早期有运动
            (50.0, 50.0), (50.5, 50.2), (50.1, 50.1),  # 后期静止
        ]
    }
    score, flags = engine.evaluate(kpts, bbox, history)
    assert score >= 0.6
    assert flags['A'] is True
    assert flags['B'] is True
    assert flags['C'] is True


def test_crouch_not_fall():
    engine = RuleEngine()
    # 蹲下：H_ratio 低，但运动历史持续在动（没有静止阶段）
    kpts = np.zeros((17, 3))
    kpts[0] = [50, 120, 0.9]
    kpts[15] = [40, 180, 0.9]
    kpts[16] = [60, 180, 0.9]
    bbox = [0, 0, 100, 200]
    history = {
        'centers': [
            (10.0, 10.0), (20.0, 20.0), (30.0, 30.0),
            (40.0, 40.0), (50.0, 50.0), (60.0, 60.0),
        ]
    }
    score, flags = engine.evaluate(kpts, bbox, history)
    # 蹲下时虽然 H_ratio 低，但 C 不触发，总分会不足
    assert score < 0.6
    assert flags['C'] is False
