from fall_detection.fusion import FusionDecision


def test_alarm_triggered():
    fd = FusionDecision()
    for _ in range(10):
        fd.update(rule_score=0.5, cls_score=0.9)
    is_fall = fd.decide()
    assert is_fall is True


def test_alarm_not_triggered():
    fd = FusionDecision()
    for _ in range(3):
        fd.update(rule_score=0.1, cls_score=0.2)
    is_fall = fd.decide()
    assert is_fall is False


def test_alarm_reset():
    fd = FusionDecision(fps=25, alarm_min_frames=2, reset_seconds=0.04)
    # trigger alarm
    for _ in range(3):
        fd.update(rule_score=0.8, cls_score=0.9)
    assert fd.decide() is True

    # reset after enough low frames
    for _ in range(5):
        fd.update(rule_score=0.1, cls_score=0.1)
    assert fd.decide() is False


def test_fusion_score():
    fd = FusionDecision()
    fd.update(rule_score=1.0, cls_score=0.0)
    # S_temporal = 0, S_final = 0.35*1 + 0.45*0 + 0.20*0 = 0.35
    state = fd.get_state()
    assert abs(state['S_final'] - 0.35) < 1e-6
