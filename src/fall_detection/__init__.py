"""边缘AI跌倒检测系统.

专为海思 HiSilicon 3516C 平台设计的纯视觉边缘跌倒检测系统。
"""

__version__ = "1.0.0"

# 便捷导入
from .core import (
    PersonDetector,
    ByteTrackLite,
    Detection,
    Track,
    PoseEstimator,
    RuleEngine,
    FusionDecision,
    FallState,
)
from .models import FallClassifier, SimpleFallClassifier
from .pipeline import FallDetectionPipeline

__all__ = [
    "PersonDetector",
    "ByteTrackLite",
    "Detection",
    "Track",
    "PoseEstimator",
    "RuleEngine",
    "FusionDecision",
    "FallState",
    "FallClassifier",
    "SimpleFallClassifier",
    "FallDetectionPipeline",
]
