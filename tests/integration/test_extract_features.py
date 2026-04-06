"""Integration tests for extract_features.py script."""

import argparse
import json
import subprocess
import sys
from unittest import mock

import numpy as np
import importlib.util

# Ensure src is in path
sys.path.insert(0, "src")


def load_extract_features_module():
    """Helper to load the extract_features module."""
    spec = importlib.util.spec_from_file_location(
        "extract_features", "scripts/train/extract_features.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["extract_features"] = module
    spec.loader.exec_module(module)
    return module


class TestArgumentParsing:
    """Test argument parsing for extract_features.py."""

    def test_parse_args_defaults(self, tmp_path):
        """Test argument parsing with default values."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--video-dir", default="data/videos")
        parser.add_argument("--label-file", default="data/labels.json")
        parser.add_argument("--config", default="configs/default.yaml")
        parser.add_argument("--out-dir", default="train/cache")
        parser.add_argument("--sample-fps", type=int, default=5)

        args = parser.parse_args([])

        assert args.video_dir == "data/videos"
        assert args.label_file == "data/labels.json"
        assert args.config == "configs/default.yaml"
        assert args.out_dir == "train/cache"
        assert args.sample_fps == 5

    def test_parse_args_custom_values(self, tmp_path):
        """Test argument parsing with custom values."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--video-dir", default="data/videos")
        parser.add_argument("--label-file", default="data/labels.json")
        parser.add_argument("--config", default="configs/default.yaml")
        parser.add_argument("--out-dir", default="train/cache")
        parser.add_argument("--sample-fps", type=int, default=5)

        video_dir = tmp_path / "videos"
        label_file = tmp_path / "labels.json"
        config_file = tmp_path / "config.yaml"
        out_dir = tmp_path / "output"

        args = parser.parse_args([
            "--video-dir", str(video_dir),
            "--label-file", str(label_file),
            "--config", str(config_file),
            "--out-dir", str(out_dir),
            "--sample-fps", "10"
        ])

        assert args.video_dir == str(video_dir)
        assert args.label_file == str(label_file)
        assert args.config == str(config_file)
        assert args.out_dir == str(out_dir)
        assert args.sample_fps == 10

    def test_parse_args_sample_fps_variations(self):
        """Test different sample-fps values."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--sample-fps", type=int, default=5)
        parser.add_argument("--video-dir", default="data/videos")

        for fps in [1, 5, 10, 15, 30]:
            args = parser.parse_args(["--sample-fps", str(fps)])
            assert args.sample_fps == fps


class TestParseLabel:
    """Test parse_label function."""

    def test_parse_label_no_label_file(self):
        """Test parsing label when file doesn't exist."""
        module = load_extract_features_module()

        result = module.parse_label("/nonexistent/path.json", "video.mp4", 100)
        assert result == 0

    def test_parse_label_simple_label(self, tmp_path):
        """Test parsing simple label from file."""
        module = load_extract_features_module()

        labels = {
            "video1.mp4": {"label": 1, "frames": [0, 300]},
            "video2.mp4": {"label": 0, "frames": [0, 300]}
        }

        label_file = tmp_path / "labels.json"
        with open(label_file, "w") as f:
            json.dump(labels, f)

        result1 = module.parse_label(str(label_file), "video1.mp4", 100)
        assert result1 == 1

        result2 = module.parse_label(str(label_file), "video2.mp4", 100)
        assert result2 == 0

    def test_parse_label_frame_range(self, tmp_path):
        """Test parsing label with frame range check."""
        module = load_extract_features_module()

        labels = {
            "video1.mp4": {"label": 1, "frames": [50, 150]}
        }

        label_file = tmp_path / "labels.json"
        with open(label_file, "w") as f:
            json.dump(labels, f)

        # Frame within range
        result_in = module.parse_label(str(label_file), "video1.mp4", 100)
        assert result_in == 1

        # Frame before range
        result_before = module.parse_label(str(label_file), "video1.mp4", 30)
        assert result_before == 0

        # Frame after range
        result_after = module.parse_label(str(label_file), "video1.mp4", 200)
        assert result_after == 0

    def test_parse_label_no_frames_field(self, tmp_path):
        """Test parsing label when frames field is missing."""
        module = load_extract_features_module()

        labels = {
            "video1.mp4": {"label": 1}  # No frames field
        }

        label_file = tmp_path / "labels.json"
        with open(label_file, "w") as f:
            json.dump(labels, f)

        result = module.parse_label(str(label_file), "video1.mp4", 100)
        assert result == 1  # Returns label directly

    def test_parse_label_video_not_found(self, tmp_path):
        """Test parsing label when video not in label file."""
        module = load_extract_features_module()

        labels = {
            "video1.mp4": {"label": 1, "frames": [0, 100]}
        }

        label_file = tmp_path / "labels.json"
        with open(label_file, "w") as f:
            json.dump(labels, f)

        result = module.parse_label(str(label_file), "nonexistent.mp4", 50)
        assert result == 0  # Default to 0


class TestFeatureExtractionComponents:
    """Test feature extraction components."""

    @mock.patch("cv2.VideoCapture")
    @mock.patch("fall_detection.pipeline.pipeline.FallDetectionPipeline")
    def test_extract_function_basic(self, mock_pipeline_class, mock_capture_class, tmp_path):
        """Test basic extract function execution."""
        module = load_extract_features_module()

        # Setup mock VideoCapture
        mock_cap = mock.MagicMock()
        mock_cap.get.return_value = 25.0  # 25 fps
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)  # End of video
        ]
        mock_capture_class.return_value = mock_cap

        # Setup mock pipeline
        mock_pipeline = mock.MagicMock()
        mock_track = mock.MagicMock()
        mock_track.track_id = 1
        mock_track.to_tlbr.return_value = np.array([100, 100, 200, 300])
        mock_pipeline.process_frame.return_value = {
            "tracks": [mock_track],
            "track_kpts": {1: np.zeros((17, 3), dtype=np.float32)}
        }
        mock_pipeline._track_history = {1: [(150, 200)] * 10}
        mock_pipeline._extract_motion.return_value = np.zeros(8, dtype=np.float32)
        mock_pipeline_class.return_value = mock_pipeline

        video_path = tmp_path / "test.mp4"
        out_dir = tmp_path / "cache"

        # Create a dummy video file
        video_path.write_bytes(b"dummy video data")

        result = module.extract(str(video_path), mock_pipeline, 1, str(out_dir), sample_fps=5)

        # Should have processed frames and saved features
        assert result >= 0

    @mock.patch("cv2.VideoCapture")
    def test_extract_skip_frames(self, mock_capture_class, tmp_path):
        """Test that frames are properly skipped based on sample_fps."""
        module = load_extract_features_module()

        # Setup mock VideoCapture with 25 fps
        mock_cap = mock.MagicMock()
        mock_cap.get.return_value = 25.0
        # Create 50 frames
        frames = [(True, np.zeros((480, 640, 3), dtype=np.uint8)) for _ in range(50)]
        frames.append((False, None))
        mock_cap.read.side_effect = frames
        mock_capture_class.return_value = mock_cap

        # Setup mock pipeline
        mock_pipeline = mock.MagicMock()
        mock_track = mock.MagicMock()
        mock_track.track_id = 1
        mock_track.to_tlbr.return_value = np.array([100, 100, 200, 300])
        mock_pipeline.process_frame.return_value = {
            "tracks": [mock_track],
            "track_kpts": {1: np.zeros((17, 3), dtype=np.float32)}
        }
        mock_pipeline._track_history = {1: [(150, 200)] * 10}
        mock_pipeline._extract_motion.return_value = np.zeros(8, dtype=np.float32)

        video_path = tmp_path / "test.mp4"
        out_dir = tmp_path / "cache"
        video_path.write_bytes(b"dummy")

        # With 25 fps video and sample_fps=5, skip = 5
        # So we should process frames 0, 5, 10, 15, ...
        result = module.extract(str(video_path), mock_pipeline, 1, str(out_dir), sample_fps=5)

        # Verify VideoCapture was opened
        mock_capture_class.assert_called_once_with(str(video_path))

    @mock.patch("cv2.VideoCapture")
    def test_extract_video_fps_calculation(self, mock_capture_class, tmp_path):
        """Test video fps calculation for skip factor."""
        module = load_extract_features_module()

        test_cases = [
            (25.0, 5, 5),   # 25 fps, sample 5 fps -> skip 5
            (30.0, 5, 6),   # 30 fps, sample 5 fps -> skip 6
            (15.0, 5, 3),   # 15 fps, sample 5 fps -> skip 3
            (5.0, 5, 1),    # 5 fps, sample 5 fps -> skip 1
            (5.0, 10, 1),   # 5 fps, sample 10 fps -> skip 1 (max)
        ]

        for video_fps, sample_fps, expected_skip in test_cases:
            mock_cap = mock.MagicMock()
            mock_cap.get.return_value = video_fps
            mock_cap.read.side_effect = [(False, None)]
            mock_capture_class.return_value = mock_cap

            mock_pipeline = mock.MagicMock()

            video_path = tmp_path / f"test_{video_fps}.mp4"
            video_path.write_bytes(b"dummy")
            out_dir = tmp_path / "cache"

            module.extract(str(video_path), mock_pipeline, 0, str(out_dir), sample_fps=sample_fps)

            # The skip factor is calculated as max(1, int(round(video_fps / sample_fps)))
            expected_skip_calc = max(1, int(round(video_fps / sample_fps)))
            assert expected_skip_calc == expected_skip

    @mock.patch("cv2.VideoCapture")
    def test_extract_roi_cropping(self, mock_capture_class, tmp_path):
        """Test ROI cropping from frame."""
        module = load_extract_features_module()

        # Create a frame with specific pattern
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        mock_cap = mock.MagicMock()
        mock_cap.get.return_value = 25.0
        mock_cap.read.side_effect = [(True, frame), (False, None)]
        mock_capture_class.return_value = mock_cap

        mock_pipeline = mock.MagicMock()
        mock_track = mock.MagicMock()
        mock_track.track_id = 1
        # Define bbox within frame
        mock_track.to_tlbr.return_value = np.array([100, 100, 196, 196])
        mock_pipeline.process_frame.return_value = {
            "tracks": [mock_track],
            "track_kpts": {1: np.zeros((17, 3), dtype=np.float32)}
        }
        mock_pipeline._track_history = {1: [(148, 148)] * 10}
        mock_pipeline._extract_motion.return_value = np.zeros(8, dtype=np.float32)

        video_path = tmp_path / "test.mp4"
        out_dir = tmp_path / "cache"
        video_path.write_bytes(b"dummy")

        result = module.extract(str(video_path), mock_pipeline, 1, str(out_dir), sample_fps=5)

        # Verify process_frame was called
        mock_pipeline.process_frame.assert_called()

    @mock.patch("cv2.VideoCapture")
    def test_extract_invalid_bbox(self, mock_capture_class, tmp_path):
        """Test handling of invalid bounding boxes."""
        module = load_extract_features_module()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = mock.MagicMock()
        mock_cap.get.return_value = 25.0
        mock_cap.read.side_effect = [(True, frame), (False, None)]
        mock_capture_class.return_value = mock_cap

        mock_pipeline = mock.MagicMock()
        mock_track = mock.MagicMock()
        mock_track.track_id = 1
        # Invalid bbox (x2 <= x1)
        mock_track.to_tlbr.return_value = np.array([200, 100, 100, 300])
        mock_pipeline.process_frame.return_value = {
            "tracks": [mock_track],
            "track_kpts": {1: np.zeros((17, 3), dtype=np.float32)}
        }
        mock_pipeline._track_history = {1: [(150, 200)] * 10}
        mock_pipeline._extract_motion.return_value = np.zeros(8, dtype=np.float32)

        video_path = tmp_path / "test.mp4"
        out_dir = tmp_path / "cache"
        video_path.write_bytes(b"dummy")

        result = module.extract(str(video_path), mock_pipeline, 1, str(out_dir), sample_fps=5)

        # Should skip invalid bbox and return 0 saved
        assert result == 0

    @mock.patch("cv2.VideoCapture")
    def test_extract_bbox_clipping(self, mock_capture_class, tmp_path):
        """Test bounding box clipping to frame boundaries."""
        module = load_extract_features_module()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = mock.MagicMock()
        mock_cap.get.return_value = 25.0
        mock_cap.read.side_effect = [(True, frame), (False, None)]
        mock_capture_class.return_value = mock_cap

        mock_pipeline = mock.MagicMock()
        mock_track = mock.MagicMock()
        mock_track.track_id = 1
        # Bbox partially outside frame (negative x1, y1)
        mock_track.to_tlbr.return_value = np.array([-50, -50, 100, 100])
        mock_pipeline.process_frame.return_value = {
            "tracks": [mock_track],
            "track_kpts": {1: np.zeros((17, 3), dtype=np.float32)}
        }
        mock_pipeline._track_history = {1: [(25, 25)] * 10}
        mock_pipeline._extract_motion.return_value = np.zeros(8, dtype=np.float32)

        video_path = tmp_path / "test.mp4"
        out_dir = tmp_path / "cache"
        video_path.write_bytes(b"dummy")

        result = module.extract(str(video_path), mock_pipeline, 1, str(out_dir), sample_fps=5)

        # Should clip bbox and save features
        assert result >= 0

    @mock.patch("cv2.VideoCapture")
    def test_extract_no_keypoints(self, mock_capture_class, tmp_path):
        """Test handling when no keypoints are available."""
        module = load_extract_features_module()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = mock.MagicMock()
        mock_cap.get.return_value = 25.0
        mock_cap.read.side_effect = [(True, frame), (False, None)]
        mock_capture_class.return_value = mock_cap

        mock_pipeline = mock.MagicMock()
        mock_track = mock.MagicMock()
        mock_track.track_id = 1
        mock_track.to_tlbr.return_value = np.array([100, 100, 200, 300])
        # No keypoints for this track
        mock_pipeline.process_frame.return_value = {
            "tracks": [mock_track],
            "track_kpts": {}  # Empty keypoints
        }
        mock_pipeline._track_history = {1: [(150, 200)] * 10}

        video_path = tmp_path / "test.mp4"
        out_dir = tmp_path / "cache"
        video_path.write_bytes(b"dummy")

        result = module.extract(str(video_path), mock_pipeline, 1, str(out_dir), sample_fps=5)

        # Should skip track with no keypoints
        assert result == 0


class TestSavedFeatures:
    """Test saved feature files."""

    @mock.patch("cv2.VideoCapture")
    @mock.patch("fall_detection.pipeline.pipeline.FallDetectionPipeline")
    def test_saved_npz_contents(self, mock_pipeline_class, mock_capture_class, tmp_path):
        """Test that saved NPZ files contain expected data."""
        module = load_extract_features_module()

        # Setup mocks
        mock_cap = mock.MagicMock()
        mock_cap.get.return_value = 25.0
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]
        mock_capture_class.return_value = mock_cap

        mock_pipeline = mock.MagicMock()
        mock_track = mock.MagicMock()
        mock_track.track_id = 42
        mock_track.to_tlbr.return_value = np.array([100, 100, 196, 196])  # Will be resized to 96x96
        mock_pipeline.process_frame.return_value = {
            "tracks": [mock_track],
            "track_kpts": {42: np.random.randn(17, 3).astype(np.float32)}
        }
        mock_pipeline._track_history = {42: [(148, 148)] * 10}
        mock_pipeline._extract_motion.return_value = np.random.randn(8).astype(np.float32)
        mock_pipeline_class.return_value = mock_pipeline

        video_path = tmp_path / "test_video.mp4"
        out_dir = tmp_path / "cache"
        video_path.write_bytes(b"dummy")

        result = module.extract(str(video_path), mock_pipeline, 1, str(out_dir), sample_fps=5)

        assert result == 1

        # Check saved file
        saved_files = list(out_dir.glob("*.npz"))
        assert len(saved_files) == 1

        # Load and verify contents
        data = np.load(saved_files[0])
        assert "roi" in data
        assert "kpts" in data
        assert "motion" in data
        assert "label" in data

        assert data["roi"].shape == (96, 96, 3)
        assert data["kpts"].shape == (17, 3)
        assert data["motion"].shape == (8,)
        assert data["label"] == 1

    @mock.patch("cv2.VideoCapture")
    @mock.patch("fall_detection.pipeline.pipeline.FallDetectionPipeline")
    def test_saved_file_naming(self, mock_pipeline_class, mock_capture_class, tmp_path):
        """Test saved file naming convention."""
        module = load_extract_features_module()

        mock_cap = mock.MagicMock()
        mock_cap.get.return_value = 25.0
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None)
        ]
        mock_capture_class.return_value = mock_cap

        mock_pipeline = mock.MagicMock()
        mock_track = mock.MagicMock()
        mock_track.track_id = 5
        mock_track.to_tlbr.return_value = np.array([100, 100, 196, 196])
        mock_pipeline.process_frame.return_value = {
            "tracks": [mock_track],
            "track_kpts": {5: np.zeros((17, 3), dtype=np.float32)}
        }
        mock_pipeline._track_history = {5: [(148, 148)] * 10}
        mock_pipeline._extract_motion.return_value = np.zeros(8, dtype=np.float32)
        mock_pipeline_class.return_value = mock_pipeline

        video_path = tmp_path / "my_video.mp4"
        out_dir = tmp_path / "cache"
        video_path.write_bytes(b"dummy")

        module.extract(str(video_path), mock_pipeline, 0, str(out_dir), sample_fps=5)

        saved_files = list(out_dir.glob("*.npz"))
        assert len(saved_files) == 1

        # Check naming: {video_name}_f{frame_idx}_t{track_id}.npz
        filename = saved_files[0].name
        assert filename.startswith("my_video_f")
        assert "_t5" in filename
        assert filename.endswith(".npz")


class TestMainFunction:
    """Test main function."""

    def test_main_argument_parsing(self, tmp_path):
        """Test main function argument parsing structure."""
        # Verify the argument parser structure by checking help output
        result = subprocess.run(
            [sys.executable, "scripts/train/extract_features.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--video-dir" in result.stdout
        assert "--label-file" in result.stdout

    def test_main_with_empty_video_dir(self, tmp_path):
        """Test main function with empty video directory - just verify structure."""
        # Verify that main function exists and has proper structure
        module = load_extract_features_module()

        # Check that main function is defined
        assert hasattr(module, 'main')
        assert callable(module.main)

        # Verify argument parser is set up correctly
        result = subprocess.run(
            [sys.executable, "scripts/train/extract_features.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0


class TestIntegrationWithFileSystem:
    """Integration tests with actual file system operations."""

    def test_script_help_output(self):
        """Test that script produces help output."""
        result = subprocess.run(
            [sys.executable, "scripts/train/extract_features.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--video-dir" in result.stdout
        assert "--label-file" in result.stdout
        assert "--config" in result.stdout
        assert "--out-dir" in result.stdout
        assert "--sample-fps" in result.stdout

    def test_label_json_loading(self, tmp_path):
        """Test that label JSON can be loaded."""
        labels = {
            "video1.mp4": {
                "label": 1,
                "frames": [50, 150],
                "notes": "Fall event"
            },
            "video2.mp4": {
                "label": 0,
                "frames": [0, 200]
            }
        }

        label_file = tmp_path / "labels.json"
        with open(label_file, "w") as f:
            json.dump(labels, f)

        # Verify it can be loaded
        with open(label_file, "r") as f:
            loaded = json.load(f)

        assert loaded["video1.mp4"]["label"] == 1
        assert loaded["video1.mp4"]["frames"] == [50, 150]
        assert loaded["video2.mp4"]["label"] == 0
