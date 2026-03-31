from fall_detection.tracker import ByteTrackLite, Detection


def test_tracker_update():
    tracker = ByteTrackLite()
    dets = [Detection([10, 10, 30, 30], 0.8)]
    tracks = tracker.update(dets)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_tracker_id_persistence():
    tracker = ByteTrackLite()
    tracks = tracker.update([Detection([10, 10, 30, 30], 0.8)])
    assert tracks[0].track_id == 1
    tracks = tracker.update([Detection([12, 12, 32, 32], 0.8)])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
