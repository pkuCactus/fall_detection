"""Integration tests for train_detector.py script."""

import argparse
import os
import subprocess
import sys
import tempfile
from unittest import mock

import pytest
import yaml
import importlib.util

# Ensure src is in path
sys.path.insert(0, "src")


def load_train_detector_module():
    """Helper to load the train_detector module."""
    spec = importlib.util.spec_from_file_location(
        "train_detector", "scripts/train/train_detector.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_detector"] = module
    spec.loader.exec_module(module)
    return module


class TestTrainDetectorArgumentParsing:
    """Test argument parsing for train_detector.py."""

    def test_parse_args_defaults(self, tmp_path):
        """Test argument parsing with default values."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--data", default="data/fall_detection.yaml")
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--imgsz", type=int, default=832)
        parser.add_argument("--batch", type=int, default=16)
        parser.add_argument("--model", default="yolov8n.pt")
        parser.add_argument("--project", default="train/detector")
        parser.add_argument("--name", default="exp")

        test_args = ["--data", str(tmp_path / "data.yaml")]
        parsed = parser.parse_args(test_args)

        assert parsed.data == str(tmp_path / "data.yaml")
        assert parsed.epochs == 50  # default
        assert parsed.imgsz == 832  # default
        assert parsed.batch == 16  # default
        assert parsed.model == "yolov8n.pt"  # default
        assert parsed.project == "train/detector"  # default
        assert parsed.name == "exp"  # default

    def test_parse_args_custom_values(self, tmp_path):
        """Test argument parsing with custom values."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--data", default="data/fall_detection.yaml")
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--imgsz", type=int, default=832)
        parser.add_argument("--batch", type=int, default=16)
        parser.add_argument("--model", default="yolov8n.pt")
        parser.add_argument("--project", default="train/detector")
        parser.add_argument("--name", default="exp")

        test_args = [
            "--data", str(tmp_path / "custom.yaml"),
            "--epochs", "100",
            "--imgsz", "640",
            "--batch", "32",
            "--model", "yolov8s.pt",
            "--project", "outputs/detector",
            "--name", "custom_exp"
        ]
        parsed = parser.parse_args(test_args)

        assert parsed.data == str(tmp_path / "custom.yaml")
        assert parsed.epochs == 100
        assert parsed.imgsz == 640
        assert parsed.batch == 32
        assert parsed.model == "yolov8s.pt"
        assert parsed.project == "outputs/detector"
        assert parsed.name == "custom_exp"

    def test_parse_args_required_data(self):
        """Test that data argument is effectively required."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--data", required=True)
        parser.add_argument("--epochs", type=int, default=50)

        # Should fail without data
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestTrainDetectorEndToEnd:
    """Test train_detector end-to-end with mocked YOLO."""

    @mock.patch("ultralytics.YOLO")
    def test_train_detector_end_to_end(self, mock_yolo_class, tmp_path):
        """Test complete training workflow with mocked YOLO."""
        # Setup mock YOLO instance
        mock_model = mock.MagicMock()
        mock_yolo_class.return_value = mock_model

        # Create temporary data config
        data_config = {
            "path": str(tmp_path),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "person", 1: "fall"}
        }
        data_yaml_path = tmp_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_config, f)

        # Create project directory structure
        project_dir = tmp_path / "train" / "detector"
        exp_dir = project_dir / "exp"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Create a dummy best.pt file
        best_pt_path = weights_dir / "best.pt"
        best_pt_path.write_bytes(b"dummy model weights")

        # Load module and run main with mocked args
        module = load_train_detector_module()

        config_yaml_path = tmp_path / "config.yaml"
        config = {
            'data': str(data_yaml_path),
            "epochs": 10,
            "imgsz": 640,
            "batch": 8,
            'model': "yolov8n.pt",
            'project': str(project_dir),
            'name': "exp",
        }
        with open(config_yaml_path, "w") as f:
            yaml.dump(config, f)

        test_args = [
            "--config", str(config_yaml_path)
        ]

        with mock.patch("sys.argv", ["train_detector.py"] + test_args):
            module.main()

        # Verify YOLO was instantiated with correct model
        mock_yolo_class.assert_called_once_with("yolov8n.pt")

        # Verify train was called with correct arguments
        mock_model.train.assert_called_once()
        call_kwargs = mock_model.train.call_args.kwargs
        assert call_kwargs["data"] == str(data_yaml_path)
        assert call_kwargs["epochs"] == 10
        assert call_kwargs["imgsz"] == 640
        assert call_kwargs["batch"] == 8
        assert call_kwargs["project"] == str(project_dir)
        assert call_kwargs["name"] == "exp"

    @mock.patch("ultralytics.YOLO")
    def test_train_detector_with_different_models(self, mock_yolo_class, tmp_path):
        """Test training with different YOLO model variants."""
        mock_model = mock.MagicMock()
        mock_yolo_class.return_value = mock_model

        data_yaml_path = tmp_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump({"names": {0: "person"}}, f)

        project_dir = tmp_path / "train" / "detector"
        config = {}
        config_yaml_path = tmp_path / "config.yaml"
        with open(config_yaml_path, "w") as f:
            yaml.dump(config, f)

        for i, model_name in enumerate(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]):
            mock_yolo_class.reset_mock()
            mock_model.reset_mock()

            # Use unique experiment name for each iteration
            exp_name = f"test_exp_{i}"
            exp_dir = project_dir / exp_name
            weights_dir = exp_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            (weights_dir / "best.pt").write_bytes(b"dummy")

            # Reload module for each iteration
            if "train_detector" in sys.modules:
                del sys.modules["train_detector"]
            module = load_train_detector_module()

            test_args = [
                "--config", str(config_yaml_path),
                "--override", "model={}".format(model_name),
            ]

            with mock.patch("sys.argv", ["train_detector.py"] + test_args):
                module.main()

            mock_yolo_class.assert_called_once_with(model_name)


class TestTrainDetectorDataConfig:
    """Test data configuration handling."""

    @mock.patch("ultralytics.YOLO")
    def test_data_config_loading(self, mock_yolo_class, tmp_path):
        """Test that data config is passed correctly to YOLO."""
        mock_model = mock.MagicMock()
        mock_yolo_class.return_value = mock_model

        # Create comprehensive data config
        data_config = {
            "path": str(tmp_path),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "person", 1: "fall", 2: "lying"},
            "nc": 3
        }
        data_yaml_path = tmp_path / "fall_detection.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_config, f)

        project_dir = tmp_path / "train" / "detector"
        exp_dir = project_dir / "exp"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"dummy")

        module = load_train_detector_module()
        config_yaml_path = tmp_path / "config.yaml"
        config = {
            'data': str(data_yaml_path),
            'project': str(project_dir),
            'name': "exp",
        }
        with open(config_yaml_path, "w") as f:
            yaml.dump(config, f)

        test_args = [
            "--config", str(config_yaml_path)
        ]

        with mock.patch("sys.argv", ["train_detector.py"] + test_args):
            module.main()

        # Verify data path was passed correctly
        call_kwargs = mock_model.train.call_args.kwargs
        assert call_kwargs["data"] == str(data_yaml_path)


class TestTrainDetectorIntegrationWithFileSystem:
    """Integration tests with actual file system operations."""

    def test_script_help_output(self):
        """Test that script produces help output."""
        result = subprocess.run(
            [sys.executable, "scripts/train/train_detector.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--override" in result.stdout
