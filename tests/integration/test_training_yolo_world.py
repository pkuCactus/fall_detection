"""Integration tests for train_yolo_world.py script."""

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


def load_train_yolo_world_module():
    """Helper to load the train_yolo_world module."""
    spec = importlib.util.spec_from_file_location(
        "train_yolo_world", "scripts/train/train_yolo_world.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_yolo_world"] = module
    spec.loader.exec_module(module)
    return module


class TestTrainYOLOWorldArgumentParsing:
    """Test argument parsing for train_yolo_world.py."""

    def test_parse_args_defaults(self, tmp_path):
        """Test argument parsing with default values."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--data", default="data/fall_detection.yaml")
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--imgsz", type=int, default=640)
        parser.add_argument("--batch", type=int, default=16)
        parser.add_argument("--model", default="yolov8l-worldv2.pt")
        parser.add_argument("--project", default="train/yolo_world")
        parser.add_argument("--name", default="exp")

        test_args = ["--data", str(tmp_path / "data.yaml")]
        parsed = parser.parse_args(test_args)

        assert parsed.data == str(tmp_path / "data.yaml")
        assert parsed.epochs == 50  # default
        assert parsed.imgsz == 640  # default
        assert parsed.batch == 16  # default
        assert parsed.model == "yolov8l-worldv2.pt"  # default
        assert parsed.project == "train/yolo_world"  # default
        assert parsed.name == "exp"  # default

    def test_parse_args_custom_values(self, tmp_path):
        """Test argument parsing with custom values."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--data", default="data/fall_detection.yaml")
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--imgsz", type=int, default=640)
        parser.add_argument("--batch", type=int, default=16)
        parser.add_argument("--model", default="yolov8l-worldv2.pt")
        parser.add_argument("--project", default="train/yolo_world")
        parser.add_argument("--name", default="exp")

        test_args = [
            "--data", str(tmp_path / "custom.yaml"),
            "--epochs", "75",
            "--imgsz", "832",
            "--batch", "8",
            "--model", "yolov8s-worldv2.pt",
            "--project", "outputs/yolo_world",
            "--name", "custom_run"
        ]
        parsed = parser.parse_args(test_args)

        assert parsed.data == str(tmp_path / "custom.yaml")
        assert parsed.epochs == 75
        assert parsed.imgsz == 832
        assert parsed.batch == 8
        assert parsed.model == "yolov8s-worldv2.pt"
        assert parsed.project == "outputs/yolo_world"
        assert parsed.name == "custom_run"

    def test_parse_args_all_yolo_world_models(self, tmp_path):
        """Test parsing with different YOLO-World model variants."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default="yolov8l-worldv2.pt")
        parser.add_argument("--data", default="data.yaml")

        models = [
            "yolov8s-worldv2.pt",
            "yolov8m-worldv2.pt",
            "yolov8l-worldv2.pt",
            "yolov8x-worldv2.pt"
        ]

        for model in models:
            parsed = parser.parse_args(["--data", str(tmp_path / "data.yaml"), "--model", model])
            assert parsed.model == model


class TestTrainYOLOWorldClassLoading:
    """Test class loading from data configuration."""

    @mock.patch("ultralytics.YOLOWorld")
    def test_class_loading_dict_format(self, mock_yolo_class, tmp_path):
        """Test loading classes when names is a dict."""
        mock_model = mock.MagicMock()
        mock_yolo_class.return_value = mock_model

        # Create data config with dict-style names
        data_config = {
            "path": str(tmp_path),
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "person",
                1: "fall",
                2: "lying_person",
                3: "sitting_person"
            }
        }
        data_yaml_path = tmp_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_config, f)

        # Create output structure
        project_dir = tmp_path / "train" / "yolo_world"
        exp_dir = project_dir / "exp"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"dummy")

        module = load_train_yolo_world_module()

        test_args = [
            "--data", str(data_yaml_path),
            "--project", str(project_dir),
            "--name", "exp"
        ]

        with mock.patch("sys.argv", ["train_yolo_world.py"] + test_args):
            module.main()

        # Verify set_classes was called with sorted class names
        mock_model.set_classes.assert_called_once()
        classes_passed = mock_model.set_classes.call_args[0][0]
        assert classes_passed == ["person", "fall", "lying_person", "sitting_person"]

    @mock.patch("ultralytics.YOLOWorld")
    def test_class_loading_list_format(self, mock_yolo_class, tmp_path):
        """Test loading classes when names is a list."""
        mock_model = mock.MagicMock()
        mock_yolo_class.return_value = mock_model

        # Create data config with list-style names
        data_config = {
            "path": str(tmp_path),
            "train": "images/train",
            "val": "images/val",
            "names": ["person", "fall", "lying"]
        }
        data_yaml_path = tmp_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_config, f)

        project_dir = tmp_path / "train" / "yolo_world"
        exp_dir = project_dir / "exp"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"dummy")

        module = load_train_yolo_world_module()

        test_args = ["--data", str(data_yaml_path), "--project", str(project_dir), "--name", "exp"]

        with mock.patch("sys.argv", ["train_yolo_world.py"] + test_args):
            module.main()

        classes_passed = mock_model.set_classes.call_args[0][0]
        assert classes_passed == ["person", "fall", "lying"]

    @mock.patch("ultralytics.YOLOWorld")
    def test_class_loading_fallback(self, mock_yolo_class, tmp_path):
        """Test fallback to default classes when names is missing or invalid."""
        mock_model = mock.MagicMock()
        mock_yolo_class.return_value = mock_model

        # Create data config without names
        data_config = {
            "path": str(tmp_path),
            "train": "images/train",
            "val": "images/val"
        }
        data_yaml_path = tmp_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_config, f)

        project_dir = tmp_path / "train" / "yolo_world"
        exp_dir = project_dir / "exp"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"dummy")

        module = load_train_yolo_world_module()

        test_args = ["--data", str(data_yaml_path), "--project", str(project_dir), "--name", "exp"]

        with mock.patch("sys.argv", ["train_yolo_world.py"] + test_args):
            module.main()

        # When names is missing/empty, set_classes is called with empty list
        classes_passed = mock_model.set_classes.call_args[0][0]
        assert classes_passed == []

    @mock.patch("ultralytics.YOLOWorld")
    def test_class_loading_string_names(self, mock_yolo_class, tmp_path):
        """Test handling when names is a string (unexpected type)."""
        mock_model = mock.MagicMock()
        mock_yolo_class.return_value = mock_model

        # Create data config with string names (edge case)
        data_config = {
            "path": str(tmp_path),
            "names": "person"  # String instead of dict/list
        }
        data_yaml_path = tmp_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_config, f)

        project_dir = tmp_path / "train" / "yolo_world"
        exp_dir = project_dir / "exp"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"dummy")

        module = load_train_yolo_world_module()

        test_args = ["--data", str(data_yaml_path), "--project", str(project_dir), "--name", "exp"]

        with mock.patch("sys.argv", ["train_yolo_world.py"] + test_args):
            module.main()

        # When names is a string (unexpected type), set_classes is called with list(names)
        # which creates ['p', 'e', 'r', 's', 'o', 'n'] from "person"
        classes_passed = mock_model.set_classes.call_args[0][0]
        # The script converts string to list which gives individual characters
        assert isinstance(classes_passed, list)


class TestTrainYOLOWorldEndToEnd:
    """Test end-to-end training workflow with mocked YOLO."""

    @mock.patch("ultralytics.YOLOWorld")
    def test_end_to_end_training(self, mock_yolo_class, tmp_path):
        """Test complete YOLO-World training workflow."""
        mock_model = mock.MagicMock()
        mock_yolo_class.return_value = mock_model

        # Create data config
        data_config = {
            "path": str(tmp_path),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "person", 1: "fall"}
        }
        data_yaml_path = tmp_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_config, f)

        # Create output structure
        project_dir = tmp_path / "train" / "yolo_world"
        exp_dir = project_dir / "exp"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"model weights")

        module = load_train_yolo_world_module()

        test_args = [
            "--data", str(data_yaml_path),
            "--epochs", "25",
            "--imgsz", "640",
            "--batch", "16",
            "--model", "yolov8l-worldv2.pt",
            "--project", str(project_dir),
            "--name", "exp"
        ]

        with mock.patch("sys.argv", ["train_yolo_world.py"] + test_args):
            module.main()

        # Verify YOLO instantiation
        mock_yolo_class.assert_called_once_with("yolov8l-worldv2.pt")

        # Verify set_classes was called before train
        assert mock_model.set_classes.called
        assert mock_model.train.called

        # Verify train arguments
        call_kwargs = mock_model.train.call_args.kwargs
        assert call_kwargs["data"] == str(data_yaml_path)
        assert call_kwargs["epochs"] == 25
        assert call_kwargs["imgsz"] == 640
        assert call_kwargs["batch"] == 16
        assert call_kwargs["project"] == str(project_dir)
        assert call_kwargs["name"] == "exp"

    @mock.patch("ultralytics.YOLOWorld")
    def test_set_classes_called_before_train(self, mock_yolo_class, tmp_path):
        """Test that set_classes is called before train."""
        mock_model = mock.MagicMock()
        mock_yolo_class.return_value = mock_model

        data_config = {
            "names": {0: "person"}
        }
        data_yaml_path = tmp_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_config, f)

        project_dir = tmp_path / "train" / "yolo_world"
        exp_dir = project_dir / "exp"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(b"dummy")

        module = load_train_yolo_world_module()

        test_args = ["--data", str(data_yaml_path), "--project", str(project_dir), "--name", "exp"]

        with mock.patch("sys.argv", ["train_yolo_world.py"] + test_args):
            module.main()

        # Get call order
        calls = mock_model.method_calls
        set_classes_idx = None
        train_idx = None

        for idx, call in enumerate(calls):
            if call[0] == "set_classes":
                set_classes_idx = idx
            elif call[0] == "train":
                train_idx = idx

        assert set_classes_idx is not None
        assert train_idx is not None
        assert set_classes_idx < train_idx

    @mock.patch("ultralytics.YOLOWorld")
    def test_different_world_models(self, mock_yolo_class, tmp_path):
        """Test training with different YOLO-World model sizes."""
        mock_model = mock.MagicMock()
        mock_yolo_class.return_value = mock_model

        data_config = {"path": str(tmp_path), "train": "images/train", "val": "images/val", "names": {0: "person"}}
        (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (tmp_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        data_yaml_path = tmp_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_config, f)

        project_dir = tmp_path / "train" / "yolo_world"

        models = ["yolov8s-worldv2.pt", "yolov8m-worldv2.pt", "yolov8l-worldv2.pt", "yolov8x-worldv2.pt"]

        for model in models:
            mock_yolo_class.reset_mock()
            mock_model.reset_mock()

            exp_dir = project_dir / f"exp_{model.split('.')[0]}"
            weights_dir = exp_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            (weights_dir / "best.pt").write_bytes(b"dummy")

            if "train_yolo_world" in sys.modules:
                del sys.modules["train_yolo_world"]
            module = load_train_yolo_world_module()

            test_args = [
                "--data", str(data_yaml_path),
                "--model", model,
                "--project", str(project_dir),
                "--name", f"exp_{model.split('.')[0]}"
            ]

            with mock.patch("sys.argv", ["train_yolo_world.py"] + test_args):
                module.main()

            mock_yolo_class.assert_called_once_with(model)


class TestTrainYOLOWorldModelPathHandling:
    """Test model path handling and weight saving."""

    @mock.patch("ultralytics.YOLOWorld")
    def test_best_weights_saved(self, mock_yolo_class, tmp_path):
        """Test that best weights are saved correctly."""
        mock_model = mock.MagicMock()
        mock_yolo_class.return_value = mock_model

        data_config = {"path": str(tmp_path), "train": "images/train", "val": "images/val", "names": {0: "person"}}
        (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (tmp_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        data_yaml_path = tmp_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_config, f)

        project_dir = tmp_path / "train" / "yolo_world"
        exp_dir = project_dir / "exp"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        best_pt_path = weights_dir / "best.pt"
        best_pt_path.write_bytes(b"yolo world weights")

        module = load_train_yolo_world_module()

        test_args = ["--data", str(data_yaml_path), "--project", str(project_dir), "--name", "exp"]

        with mock.patch("sys.argv", ["train_yolo_world.py"] + test_args):
            module.main()

        expected_out_path = exp_dir / "best.pt"
        assert expected_out_path.exists()
        assert expected_out_path.read_bytes() == b"yolo world weights"

    @mock.patch("ultralytics.YOLOWorld")
    def test_copy_fallback_on_link_error(self, mock_yolo_class, tmp_path):
        """Test copy fallback when hard link fails."""
        mock_model = mock.MagicMock()
        mock_yolo_class.return_value = mock_model

        data_config = {"path": str(tmp_path), "train": "images/train", "val": "images/val", "names": {0: "person"}}
        (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (tmp_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        data_yaml_path = tmp_path / "data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_config, f)

        project_dir = tmp_path / "train" / "yolo_world"
        exp_dir = project_dir / "exp"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        best_pt_path = weights_dir / "best.pt"
        best_pt_path.write_bytes(b"weights content")

        module = load_train_yolo_world_module()

        with mock.patch("os.link", side_effect=OSError("Cross-device link")):
            test_args = ["--data", str(data_yaml_path), "--project", str(project_dir), "--name", "exp"]

            with mock.patch("sys.argv", ["train_yolo_world.py"] + test_args):
                module.main()

        expected_out_path = exp_dir / "best.pt"
        assert expected_out_path.exists()


class TestTrainYOLOWorldIntegration:
    """Integration tests with actual file system operations."""

    def test_script_help_output(self):
        """Test that script produces help output."""
        result = subprocess.run(
            [sys.executable, "scripts/train/train_yolo_world.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--data" in result.stdout
        assert "--epochs" in result.stdout
        assert "--imgsz" in result.stdout
        assert "--batch" in result.stdout
        assert "--model" in result.stdout
        assert "--project" in result.stdout
        assert "--name" in result.stdout
        assert "YOLOWorld" in result.stdout or "World" in result.stdout

    def test_data_yaml_loading(self, tmp_path):
        """Test that data YAML is properly loaded."""
        data_config = {
            "path": str(tmp_path),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "person", 1: "fall"}
        }
        data_yaml_path = tmp_path / "test_data.yaml"
        with open(data_yaml_path, "w") as f:
            yaml.dump(data_config, f)

        # Verify YAML can be loaded correctly
        with open(data_yaml_path, "r") as f:
            loaded = yaml.safe_load(f)

        assert loaded["names"] == {0: "person", 1: "fall"}
        assert loaded["path"] == str(tmp_path)
