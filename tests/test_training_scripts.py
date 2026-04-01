import subprocess
import sys


def test_train_detector_help():
    result = subprocess.run([sys.executable, "scripts/train_detector.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--data" in result.stdout


def test_train_pose_help():
    result = subprocess.run([sys.executable, "scripts/train_pose.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--data" in result.stdout


def test_tune_tracker_help():
    result = subprocess.run([sys.executable, "scripts/tune_tracker.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--video-dir" in result.stdout


def test_extract_features_help():
    result = subprocess.run([sys.executable, "scripts/extract_features.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--video-dir" in result.stdout


def test_train_classifier_help():
    result = subprocess.run([sys.executable, "scripts/train_classifier.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--cache-dir" in result.stdout


def test_evaluate_pipeline_help():
    result = subprocess.run([sys.executable, "scripts/evaluate_pipeline.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--video-dir" in result.stdout
