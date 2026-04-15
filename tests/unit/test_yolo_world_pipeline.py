import numpy as np
import pytest
import yaml
from unittest.mock import MagicMock, patch

import sys

sys.path.insert(0, "src")
from fall_detection.pipeline.yoloworld_pipeline import YOLOWorldFallPipeline
from fall_detection.core.tracker import Track, Detection


@pytest.fixture
def yolo_world_config(tmp_path):
    cfg = {
        "detector": {
            "conf_thresh": 0.3,
            "model_path": None,
        },
        "yolo_world_fall": {
            "classes": [
                "person standing",
                "person sitting",
                "person squatting",
                "person bending",
                "person half up or crouching",
                "person kneeling",
                "person crawling or crawling-like",
                "person lying on floor",
            ],
            "posture_map": {
                "person standing": "standing",
                "person sitting": "sitting",
                "person squatting": "crouching",
                "person bending": "crouching",
                "person half up or crouching": "crouching",
                "person kneeling": "crouching",
                "person crawling or crawling-like": "lying",
                "person lying on floor": "lying",
            },
            "fall_scores": {
                "person standing": 0.0,
                "person sitting": 0.15,
                "person squatting": 0.25,
                "person bending": 0.35,
                "person half up or crouching": 0.45,
                "person kneeling": 0.30,
                "person crawling or crawling-like": 0.55,
                "person lying on floor": 0.95,
            },
        },
        "tracker": {
            "track_thresh": 0.5,
            "match_thresh": 0.8,
            "max_age": 30,
            "min_hits": 3,
        },
        "pipeline": {
            "skip_frames": 2,
            "fps": 25,
        },
        "fusion": {
            "alpha": 0.3,
            "beta": 0.5,
            "gamma": 0.2,
            "alarm_thresh": 0.6,
            "alarm_min_frames": 3,
            "sequence_check_frames": 6,
            "cls_bypass_thresh": 1.0,
            "reset_seconds": 0.5,
            "cooldown_seconds": 3.0,
            "recovery_seconds": 0.5,
        },
    }
    p = tmp_path / "yolo_world.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return str(p)


class TestYOLOWorldFallPipeline:
    """Tests for YOLOWorldFallPipeline."""

    def test_init(self, mocker, yolo_world_config):
        mock_pd = mocker.patch("fall_detection.pipeline.yoloworld_pipeline.PersonDetector")
        mock_model = MagicMock()
        mock_model.input_size = 640
        mock_pd.return_value = mock_model

        pipeline = YOLOWorldFallPipeline(yolo_world_config, device="cpu")

        assert pipeline.skip_frames == 2
        assert pipeline.fps == 25
        assert pipeline.posture_map["person lying on floor"] == "lying"
        assert pipeline.fall_scores["person lying on floor"] == 0.95
        mock_pd.assert_called_once()

    def test_process_frame_detection(self, mocker, yolo_world_config, sample_frame):
        mock_pd = mocker.patch("fall_detection.pipeline.yoloworld_pipeline.PersonDetector")
        mock_tracker = mocker.patch("fall_detection.pipeline.yoloworld_pipeline.ByteTrackLite")

        mock_model = MagicMock()
        mock_model.input_size = 640
        mock_pd.return_value = mock_model

        mock_track = MagicMock()
        mock_track.track_id = 1
        mock_track.to_tlbr.return_value = np.array([100.0, 100.0, 200.0, 300.0])
        mock_track.to_tlwh.return_value = np.array([100.0, 100.0, 100.0, 200.0])
        mock_tracker_instance = MagicMock()
        mock_tracker_instance.update.return_value = [mock_track]
        mock_tracker.return_value = mock_tracker_instance

        pipeline = YOLOWorldFallPipeline(yolo_world_config, device="cpu")
        pipeline.detector.return_value = [
            {"bbox": [100.0, 100.0, 200.0, 300.0], "conf": 0.9, "class_id": 7, "class_name": "person lying on floor"}
        ]

        result = pipeline.process_frame(sample_frame)

        assert result["is_detection_frame"] is True
        assert len(result["tracks"]) == 1
        assert result["track_scores"][1]["debug"]["class_name"] == "person lying on floor"

    def test_process_frame_skip(self, mocker, yolo_world_config, sample_frame):
        mock_pd = mocker.patch("fall_detection.pipeline.yoloworld_pipeline.PersonDetector")
        mock_tracker = mocker.patch("fall_detection.pipeline.yoloworld_pipeline.ByteTrackLite")

        mock_model = MagicMock()
        mock_model.input_size = 640
        mock_pd.return_value = mock_model

        mock_track = MagicMock()
        mock_track.track_id = 1
        mock_track.to_tlbr.return_value = np.array([100.0, 100.0, 200.0, 300.0])
        mock_track.to_tlwh.return_value = np.array([100.0, 100.0, 100.0, 200.0])
        mock_track.time_since_update = 0

        mock_tracker_instance = MagicMock()
        mock_tracker_instance.tracks = [mock_track]
        mock_tracker_instance.update.return_value = [mock_track]
        mock_tracker.return_value = mock_tracker_instance

        pipeline = YOLOWorldFallPipeline(yolo_world_config, device="cpu")
        pipeline.detector.return_value = [
            {"bbox": [100.0, 100.0, 200.0, 300.0], "conf": 0.9, "class_id": 7, "class_name": "person lying on floor"}
        ]

        # First frame (detection)
        pipeline.process_frame(sample_frame)
        # Second frame (skip)
        result = pipeline.process_frame(sample_frame)

        assert result["is_detection_frame"] is False
        assert len(result["tracks"]) == 1

    def test_match_detections_to_tracks(self, mocker, yolo_world_config):
        mock_pd = mocker.patch("fall_detection.pipeline.yoloworld_pipeline.PersonDetector")
        mock_pd.return_value = MagicMock()

        pipeline = YOLOWorldFallPipeline(yolo_world_config, device="cpu")

        det_results = [
            {"bbox": [100.0, 100.0, 200.0, 300.0], "conf": 0.9, "class_id": 0, "class_name": "person standing"},
            {"bbox": [300.0, 300.0, 400.0, 500.0], "conf": 0.8, "class_id": 7, "class_name": "person lying on floor"},
        ]

        track1 = MagicMock()
        track1.track_id = 1
        track1.to_tlbr.return_value = np.array([105.0, 105.0, 205.0, 305.0])

        track2 = MagicMock()
        track2.track_id = 2
        track2.to_tlbr.return_value = np.array([305.0, 305.0, 405.0, 505.0])

        det_info = pipeline._match_detections_to_tracks(det_results, [track1, track2])

        assert det_info[1]["class_name"] == "person standing"
        assert det_info[2]["class_name"] == "person lying on floor"

    def test_compute_fall_score_fallen(self, mocker, yolo_world_config):
        mock_pd = mocker.patch("fall_detection.pipeline.yoloworld_pipeline.PersonDetector")
        mock_pd.return_value = MagicMock()

        pipeline = YOLOWorldFallPipeline(yolo_world_config, device="cpu")

        mock_track = MagicMock()
        mock_track.track_id = 1
        mock_track.to_tlbr.return_value = np.array([100.0, 100.0, 300.0, 200.0])  # wide box

        det_info = {1: {"class_name": "person lying on floor", "class_id": 7, "conf": 0.9}}
        score, posture, debug = pipeline._compute_fall_score(mock_track, det_info, np.zeros((480, 640, 3), dtype=np.uint8))

        assert posture == "lying"
        assert score > 0.8
        assert debug["aspect"] > 1.0

    def test_compute_fall_score_standing(self, mocker, yolo_world_config):
        mock_pd = mocker.patch("fall_detection.pipeline.yoloworld_pipeline.PersonDetector")
        mock_pd.return_value = MagicMock()

        pipeline = YOLOWorldFallPipeline(yolo_world_config, device="cpu")

        mock_track = MagicMock()
        mock_track.track_id = 1
        mock_track.to_tlbr.return_value = np.array([100.0, 100.0, 150.0, 300.0])  # tall box

        det_info = {1: {"class_name": "person standing", "class_id": 0, "conf": 0.9}}
        score, posture, debug = pipeline._compute_fall_score(mock_track, det_info, np.zeros((480, 640, 3), dtype=np.uint8))

        assert posture == "standing"
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_compute_motion_bonus_with_fast_drop(self, mocker, yolo_world_config):
        mock_pd = mocker.patch("fall_detection.pipeline.yoloworld_pipeline.PersonDetector")
        mock_pd.return_value = MagicMock()

        pipeline = YOLOWorldFallPipeline(yolo_world_config, device="cpu")

        mock_track = MagicMock()
        mock_track.track_id = 1
        mock_track.to_tlbr.return_value = np.array([100.0, 250.0, 200.0, 350.0])

        # Simulate fast vertical drop (need at least motion_window_frames + 2 centers)
        n_needed = pipeline.motion_window_frames + 2
        pipeline._track_history[1].extend([(150.0, 100.0 + i * 60.0) for i in range(n_needed)])

        history = {"centers": list(pipeline._track_history[1])}
        bonus = pipeline._compute_motion_bonus(mock_track, history)

        assert bonus > 0.0

    def test_process_tracks_alarm(self, mocker, yolo_world_config):
        mock_pd = mocker.patch("fall_detection.pipeline.yoloworld_pipeline.PersonDetector")
        mock_pd.return_value = MagicMock()

        pipeline = YOLOWorldFallPipeline(yolo_world_config, device="cpu")

        mock_track = MagicMock()
        mock_track.track_id = 1
        mock_track.to_tlbr.return_value = np.array([100.0, 100.0, 300.0, 200.0])

        det_info_standing = {1: {"class_name": "person standing", "class_id": 0, "conf": 1.0}}
        det_info_fallen = {1: {"class_name": "person lying on floor", "class_id": 7, "conf": 1.0}}

        # Feed standing frames first to build upright history
        for _ in range(pipeline.fusion_cfg["sequence_check_frames"]):
            pipeline._process_tracks([mock_track], det_info_standing, np.zeros((480, 640, 3), dtype=np.uint8))

        # Then feed fallen frames to trigger fall sequence and alarm
        result = None
        for _ in range(5):
            result = pipeline._process_tracks([mock_track], det_info_fallen, np.zeros((480, 640, 3), dtype=np.uint8))
            if 1 in result["new_alarms"]:
                break

        assert result["track_falling"][1] is True
        assert 1 in result["new_alarms"]

    def test_no_tracks(self, mocker, yolo_world_config, sample_frame):
        mock_pd = mocker.patch("fall_detection.pipeline.yoloworld_pipeline.PersonDetector")
        mock_tracker = mocker.patch("fall_detection.pipeline.yoloworld_pipeline.ByteTrackLite")

        mock_model = MagicMock()
        mock_model.input_size = 640
        mock_pd.return_value = mock_model

        mock_tracker_instance = MagicMock()
        mock_tracker_instance.update.return_value = []
        mock_tracker.return_value = mock_tracker_instance

        pipeline = YOLOWorldFallPipeline(yolo_world_config, device="cpu")
        pipeline.detector.return_value = []

        result = pipeline.process_frame(sample_frame)

        assert result["tracks"] == []
        assert result["track_scores"] == {}
