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
