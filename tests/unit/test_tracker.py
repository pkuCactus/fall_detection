import numpy as np
import pytest

from fall_detection.core.tracker import ByteTrackLite, Detection, Track, KalmanFilter, iou_cost


class TestKalmanFilter:
    """Tests for the KalmanFilter class."""

    def test_init(self):
        kf = KalmanFilter()
        assert kf.NDIM == 4
        assert kf.DT == 1.0
        assert kf._motion_mat.shape == (8, 8)
        assert kf._update_mat.shape == (4, 8)

    def test_initiate(self):
        kf = KalmanFilter()
        measurement = np.array([50.0, 50.0, 20.0, 30.0])  # cx, cy, w, h
        mean, covariance = kf.initiate(measurement)
        assert mean.shape == (8,)
        assert covariance.shape == (8, 8)
        assert np.allclose(mean[:4], measurement)
        assert np.allclose(mean[4:], 0)  # velocities initialized to 0

    def test_predict(self):
        kf = KalmanFilter()
        measurement = np.array([50.0, 50.0, 20.0, 30.0])
        mean, covariance = kf.initiate(measurement)
        mean_pred, cov_pred = kf.predict(mean, covariance)
        assert mean_pred.shape == (8,)
        assert cov_pred.shape == (8, 8)

    def test_project(self):
        kf = KalmanFilter()
        measurement = np.array([50.0, 50.0, 20.0, 30.0])
        mean, covariance = kf.initiate(measurement)
        mean_proj, cov_proj = kf.project(mean, covariance)
        assert mean_proj.shape == (4,)
        assert cov_proj.shape == (4, 4)

    def test_update(self):
        kf = KalmanFilter()
        measurement = np.array([50.0, 50.0, 20.0, 30.0])
        mean, covariance = kf.initiate(measurement)
        new_measurement = np.array([52.0, 51.0, 21.0, 31.0])
        mean_new, cov_new = kf.update(mean, covariance, new_measurement)
        assert mean_new.shape == (8,)
        assert cov_new.shape == (8, 8)


class TestDetection:
    """Tests for the Detection dataclass."""

    def test_detection_creation(self):
        det = Detection([10, 10, 30, 30], 0.8)
        assert det.bbox == [10, 10, 30, 30]
        assert det.conf == 0.8
        assert det.embed is None

    def test_detection_with_embedding(self):
        embed = np.array([1.0, 2.0, 3.0])
        det = Detection([10, 10, 30, 30], 0.8, embed)
        assert np.array_equal(det.embed, embed)

    def test_tlwh_property(self):
        det = Detection([10, 10, 30, 40], 0.8)
        tlwh = det.tlwh
        expected = np.array([10, 10, 20, 30])  # x1, y1, w=x2-x1, h=y2-y1
        assert np.allclose(tlwh, expected)


class TestTrack:
    """Tests for the Track class."""

    def test_track_init(self):
        det = Detection([10, 10, 30, 30], 0.8)
        track = Track(det, track_id=1, max_age=30, min_hits=3)
        assert track.track_id == 1
        assert track.bbox == [10, 10, 30, 30]
        assert track.conf == 0.8
        assert track.hits == 1
        assert track.age == 0
        assert track.time_since_update == 0
        assert track.state == "tentative"
        assert track.max_age == 30
        assert track.min_hits == 3

    def test_track_predict(self):
        det = Detection([10, 10, 30, 30], 0.8)
        track = Track(det, track_id=1)
        track.predict()
        assert track.age == 1
        assert track.time_since_update == 1

    def test_track_update(self):
        det = Detection([10, 10, 30, 30], 0.8)
        track = Track(det, track_id=1, min_hits=3)
        new_det = Detection([12, 12, 32, 32], 0.85)
        track.update(new_det)
        assert track.hits == 2
        assert track.time_since_update == 0
        assert track.bbox == [12, 12, 32, 32]
        assert track.conf == 0.85
        assert track.state == "tentative"  # Still tentative, hits < min_hits

    def test_track_state_transition_to_confirmed(self):
        det = Detection([10, 10, 30, 30], 0.8)
        track = Track(det, track_id=1, min_hits=3)
        # Update twice more to reach min_hits
        track.update(Detection([11, 11, 31, 31], 0.8))
        track.update(Detection([12, 12, 32, 32], 0.8))
        assert track.hits == 3
        assert track.state == "confirmed"

    def test_track_mark_missed_tentative(self):
        det = Detection([10, 10, 30, 30], 0.8)
        track = Track(det, track_id=1)
        result = track.mark_missed()
        assert result is False  # Tentative tracks are deleted when missed
        assert track.time_since_update == 1

    def test_track_mark_missed_confirmed(self):
        det = Detection([10, 10, 30, 30], 0.8)
        track = Track(det, track_id=1, min_hits=1, max_age=5)
        # Make it confirmed
        track.update(Detection([11, 11, 31, 31], 0.8))
        assert track.state == "confirmed"
        result = track.mark_missed()
        assert result is True  # Confirmed track within max_age stays
        assert track.time_since_update == 1

    def test_track_mark_missed_confirmed_expired(self):
        det = Detection([10, 10, 30, 30], 0.8)
        track = Track(det, track_id=1, min_hits=1, max_age=2)
        # Make it confirmed
        track.update(Detection([11, 11, 31, 31], 0.8))
        assert track.state == "confirmed"
        track.mark_missed()
        track.mark_missed()
        result = track.mark_missed()  # Exceeds max_age
        assert result is False  # Should be deleted

    def test_track_to_tlwh(self):
        det = Detection([10, 10, 30, 40], 0.8)
        track = Track(det, track_id=1)
        tlwh = track.to_tlwh()
        assert tlwh.shape == (4,)
        # tlwh should be [x1, y1, w, h]
        assert tlwh[0] == 10  # x1
        assert tlwh[1] == 10  # y1
        assert tlwh[2] == 20  # w
        assert tlwh[3] == 30  # h

    def test_track_to_tlbr(self):
        det = Detection([10, 10, 30, 40], 0.8)
        track = Track(det, track_id=1)
        tlbr = track.to_tlbr()
        assert tlbr.shape == (4,)
        # tlbr should be [x1, y1, x2, y2]
        assert tlbr[0] == 10  # x1
        assert tlbr[1] == 10  # y1
        assert tlbr[2] == 30  # x2
        assert tlbr[3] == 40  # y2


class TestIouCost:
    """Tests for the iou_cost function."""

    def test_iou_cost_basic(self):
        det1 = Detection([10, 10, 30, 30], 0.8)
        track1 = Track(det1, track_id=1)
        det2 = Detection([50, 50, 70, 70], 0.8)
        track2 = Track(det2, track_id=2)

        cost = iou_cost([track1, track2], [det1, det2])
        assert cost.shape == (2, 2)
        # Cost should be 1 - IoU, so diagonal should be close to 0 (high IoU)
        assert cost[0, 0] < 0.1  # Same bbox, high IoU
        assert cost[1, 1] < 0.1
        assert cost[0, 1] > 0.9  # Different bboxes, low IoU
        assert cost[1, 0] > 0.9

    def test_iou_cost_empty_tracks(self):
        det = Detection([10, 10, 30, 30], 0.8)
        cost = iou_cost([], [det])
        assert cost.shape == (0, 1)

    def test_iou_cost_empty_detections(self):
        det = Detection([10, 10, 30, 30], 0.8)
        track = Track(det, track_id=1)
        cost = iou_cost([track], [])
        assert cost.shape == (1, 0)


class TestByteTrackLite:
    """Tests for the ByteTrackLite class."""

    def test_tracker_init_default(self):
        tracker = ByteTrackLite()
        assert tracker.track_thresh == 0.5
        assert tracker.match_thresh == 0.8
        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.tracks == []
        assert tracker._next_id == 1

    def test_tracker_init_custom_params(self):
        tracker = ByteTrackLite(
            track_thresh=0.6,
            match_thresh=0.7,
            max_age=50,
            min_hits=5
        )
        assert tracker.track_thresh == 0.6
        assert tracker.match_thresh == 0.7
        assert tracker.max_age == 50
        assert tracker.min_hits == 5

    def test_tracker_update_single_detection(self):
        tracker = ByteTrackLite()
        dets = [Detection([10, 10, 30, 30], 0.8)]
        tracks = tracker.update(dets)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1

    def test_tracker_id_persistence(self):
        tracker = ByteTrackLite()
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.8)])
        assert tracks[0].track_id == 1
        tracks = tracker.update([Detection([12, 12, 32, 32], 0.8)])
        assert len(tracks) == 1
        assert tracks[0].track_id == 1

    def test_tracker_multiple_ids(self):
        tracker = ByteTrackLite()
        dets = [
            Detection([10, 10, 30, 30], 0.8),
            Detection([50, 50, 70, 70], 0.9),
            Detection([100, 100, 120, 120], 0.85)
        ]
        tracks = tracker.update(dets)
        assert len(tracks) == 3
        track_ids = {t.track_id for t in tracks}
        assert track_ids == {1, 2, 3}

    def test_tracker_empty_detections(self):
        tracker = ByteTrackLite()
        # First create a track
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.8)])
        assert len(tracks) == 1
        # Empty update should mark track as missed
        tracks = tracker.update([])
        assert len(tracks) == 0  # No active tracks (time_since_update > 0)

    def test_tracker_state_transition_tentative_to_confirmed(self):
        tracker = ByteTrackLite(min_hits=3)
        # First detection creates tentative track
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.8)])
        assert len(tracks) == 1
        assert tracks[0].state == "tentative"

        # Second update
        tracks = tracker.update([Detection([11, 11, 31, 31], 0.8)])
        assert len(tracks) == 1
        assert tracks[0].state == "tentative"

        # Third update - should become confirmed
        tracks = tracker.update([Detection([12, 12, 32, 32], 0.8)])
        assert len(tracks) == 1
        assert tracks[0].state == "confirmed"

    def test_tracker_track_deletion_after_buffer_expires(self):
        tracker = ByteTrackLite(min_hits=1, max_age=2)
        # Create and confirm track
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.8)])
        assert len(tracks) == 1

        # Miss 3 times (exceeds max_age of 2)
        tracks = tracker.update([])
        tracks = tracker.update([])
        tracks = tracker.update([])
        assert len(tracks) == 0
        assert len(tracker.tracks) == 0  # Track should be deleted

    def test_tracker_tentative_deletion_on_miss(self):
        tracker = ByteTrackLite(min_hits=3)
        # Create tentative track
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.8)])
        assert len(tracks) == 1
        assert tracks[0].state == "tentative"

        # Miss once - tentative track should be deleted
        tracks = tracker.update([])
        assert len(tracks) == 0
        assert len(tracker.tracks) == 0

    def test_tracker_low_confidence_detection_handling(self):
        tracker = ByteTrackLite(track_thresh=0.5, min_hits=1)
        # Create confirmed track
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.8)])
        assert len(tracks) == 1

        # Low confidence detection should match in second round
        tracks = tracker.update([Detection([12, 12, 32, 32], 0.3)])
        assert len(tracks) == 1
        assert tracks[0].track_id == 1  # Same track

    def test_tracker_low_confidence_no_high_confidence(self):
        tracker = ByteTrackLite(track_thresh=0.5, min_hits=1)
        # Only low confidence detections - no new tracks created
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.3)])
        assert len(tracks) == 0
        assert len(tracker.tracks) == 0

    def test_tracker_iou_matching(self):
        tracker = ByteTrackLite(track_thresh=0.5, match_thresh=0.9, min_hits=1, max_age=2)
        # Create and confirm track (needs 2 frames with min_hits=1)
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.8)])
        tracks = tracker.update([Detection([11, 11, 31, 31], 0.8)])
        assert len(tracks) == 1
        assert tracks[0].state == "confirmed"
        assert tracks[0].track_id == 1

        # Far away detection with strict threshold should not match (IoU too low)
        # The confirmed track will persist (not deleted on miss), new track created
        tracks = tracker.update([Detection([100, 100, 120, 120], 0.8)])
        # Only 1 active track returned (new track 2), track 1 has time_since_update > 0
        assert len(tracks) == 1
        assert tracks[0].track_id == 2
        # But internally both tracks exist
        assert len(tracker.tracks) == 2
        track_ids = {t.track_id for t in tracker.tracks}
        assert track_ids == {1, 2}

    def test_tracker_iou_matching_overlap(self):
        tracker = ByteTrackLite(track_thresh=0.5, match_thresh=0.8, min_hits=1)
        # Create track
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.8)])
        assert len(tracks) == 1

        # Overlapping detection should match (IoU high enough)
        tracks = tracker.update([Detection([15, 15, 35, 35], 0.8)])
        assert len(tracks) == 1
        assert tracks[0].track_id == 1

    def test_tracker_match_empty_tracks(self):
        tracker = ByteTrackLite()
        dets = [Detection([10, 10, 30, 30], 0.8)]
        matches, unmatched_tracks, unmatched_dets = tracker._match([], dets)
        assert matches == []
        assert unmatched_tracks == []
        assert unmatched_dets == [0]

    def test_tracker_match_empty_detections(self):
        tracker = ByteTrackLite()
        det = Detection([10, 10, 30, 30], 0.8)
        track = Track(det, track_id=1)
        matches, unmatched_tracks, unmatched_dets = tracker._match([track], [])
        assert matches == []
        assert unmatched_tracks == [0]
        assert unmatched_dets == []

    def test_tracker_match_both_empty(self):
        tracker = ByteTrackLite()
        matches, unmatched_tracks, unmatched_dets = tracker._match([], [])
        assert matches == []
        assert unmatched_tracks == []
        assert unmatched_dets == []

    def test_tracker_multiple_frames_id_consistency(self):
        tracker = ByteTrackLite(min_hits=1)

        # Frame 1
        tracks = tracker.update([
            Detection([10, 10, 30, 30], 0.8),
            Detection([50, 50, 70, 70], 0.8)
        ])
        assert len(tracks) == 2
        ids_frame1 = {t.track_id for t in tracks}

        # Frame 2 - slight movement
        tracks = tracker.update([
            Detection([12, 12, 32, 32], 0.8),
            Detection([52, 52, 72, 72], 0.8)
        ])
        assert len(tracks) == 2
        ids_frame2 = {t.track_id for t in tracks}
        assert ids_frame1 == ids_frame2

        # Frame 3 - one disappears
        tracks = tracker.update([Detection([14, 14, 34, 34], 0.8)])
        assert len(tracks) == 1
        assert tracks[0].track_id in ids_frame1

    def test_tracker_new_track_after_deletion(self):
        tracker = ByteTrackLite(min_hits=1, max_age=1)

        # Create track
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.8)])
        assert tracks[0].track_id == 1

        # Delete track by missing
        tracks = tracker.update([])
        tracks = tracker.update([])
        assert len(tracker.tracks) == 0

        # Create new track - should get new ID
        tracks = tracker.update([Detection([100, 100, 120, 120], 0.8)])
        assert len(tracks) == 1
        assert tracks[0].track_id == 2

    def test_tracker_high_confidence_new_track(self):
        tracker = ByteTrackLite(track_thresh=0.5, min_hits=1)

        # First high confidence detection creates track
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.6)])
        assert len(tracks) == 1

        # Another high confidence detection at different location creates new track
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.6), Detection([50, 50, 70, 70], 0.6)])
        assert len(tracks) == 2
        track_ids = {t.track_id for t in tracks}
        assert track_ids == {1, 2}

    def test_tracker_predict_on_skip_frame(self):
        tracker = ByteTrackLite(min_hits=1, max_age=2)

        # Frame 1: Create track (tentative with 1 hit)
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.8)])
        assert len(tracks) == 1
        assert tracks[0].state == "tentative"

        # Frame 2: Update track to confirm it
        tracks = tracker.update([Detection([11, 11, 31, 31], 0.8)])
        assert len(tracks) == 1
        assert tracks[0].state == "confirmed"

        # Frame 3: Empty update should trigger predict (Kalman prediction)
        tracks = tracker.update([])
        # No active tracks returned (time_since_update > 0) but track persists
        assert len(tracks) == 0
        assert len(tracker.tracks) == 1  # Confirmed track persists
        # time_since_update is incremented by predict() and mark_missed()
        assert tracker.tracks[0].time_since_update == 2

    def test_tracker_complex_scenario(self):
        """Test a complex scenario with multiple tracks, matches, and deletions."""
        tracker = ByteTrackLite(track_thresh=0.5, match_thresh=0.8, min_hits=3, max_age=2)

        # Frame 1: Two new detections (tentative, hits=1)
        tracks = tracker.update([
            Detection([10, 10, 30, 30], 0.9),
            Detection([50, 50, 70, 70], 0.9)
        ])
        assert len(tracks) == 2
        assert all(t.state == "tentative" for t in tracks)

        # Frame 2: Same detections (tentative, hits=2)
        tracks = tracker.update([
            Detection([12, 12, 32, 32], 0.9),
            Detection([52, 52, 72, 72], 0.9)
        ])
        assert len(tracks) == 2
        assert all(t.state == "tentative" for t in tracks)

        # Frame 3: One more hit, tracks become confirmed (hits=3, hits >= min_hits)
        tracks = tracker.update([
            Detection([14, 14, 34, 34], 0.9),
            Detection([54, 54, 74, 74], 0.9)
        ])
        assert len(tracks) == 2
        assert all(t.state == "confirmed" for t in tracks)

        # Frame 4: One detection missing, one new detection
        # Track 1 continues, Track 2 misses (but stays confirmed), Track 3 is new
        tracks = tracker.update([
            Detection([16, 16, 36, 36], 0.9),  # Track 1 continues
            Detection([100, 100, 120, 120], 0.9)  # New detection
        ])
        # Track 1 (confirmed, updated), Track 3 (tentative, new)
        # Track 2 is still in tracker.tracks but not in active (time_since_update > 0)
        assert len(tracks) == 2  # Only tracks with time_since_update == 0
        states = {t.track_id: t.state for t in tracks}
        assert states[1] == "confirmed"
        assert states[3] == "tentative"  # New track

        # Verify Track 2 still exists internally (confirmed but missed)
        assert len(tracker.tracks) == 3

    def test_tracker_match_threshold_boundary(self):
        """Test matching behavior at the match_thresh boundary."""
        tracker = ByteTrackLite(track_thresh=0.5, match_thresh=0.9, min_hits=1)

        # Create track
        tracks = tracker.update([Detection([10, 10, 30, 30], 0.8)])
        assert len(tracks) == 1

        # Detection with high overlap should match (IoU high enough)
        # Using generous match_thresh to ensure matching
        tracks = tracker.update([Detection([12, 12, 32, 32], 0.8)])  # Good overlap
        assert len(tracks) == 1
        assert tracks[0].track_id == 1
