import importlib.util
import os
import pytest
import subprocess
import sys
import tempfile


def test_train_detector_help():
    result = subprocess.run([sys.executable, "scripts/train/train_detector.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--data" in result.stdout


def test_train_pose_help():
    result = subprocess.run([sys.executable, "scripts/train/train_pose.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--data" in result.stdout


def test_tune_tracker_help():
    pytest.skip("tune_tracker.py has been removed from the project")


def test_extract_features_help():
    result = subprocess.run([sys.executable, "scripts/train/extract_features.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--video-dir" in result.stdout


def test_train_classifier_help():
    result = subprocess.run([sys.executable, "scripts/train/train_classifier.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--cache-dir" in result.stdout


def test_train_yolo_world_help():
    result = subprocess.run([sys.executable, "scripts/train/train_yolo_world.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--data" in result.stdout


def test_evaluate_pipeline_help():
    pytest.skip("evaluate_pipeline.py has been removed from the project")


def test_feature_dataset_empty():
    script_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train", "train_classifier.py")
    spec = importlib.util.spec_from_file_location("train_classifier", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_classifier"] = module
    spec.loader.exec_module(module)
    FeatureDataset = module.FeatureDataset
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = FeatureDataset(tmpdir)
        assert len(ds) == 0
