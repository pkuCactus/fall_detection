import subprocess
import sys


def test_train_detector_help():
    result = subprocess.run([sys.executable, "scripts/train_detector.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--data" in result.stdout
