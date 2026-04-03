from fall_detection.models.classifier import FallClassifier
import numpy as np
import torch


def test_classifier_forward():
    clf = FallClassifier()
    roi = np.zeros((3, 96, 96), dtype=np.float32)
    kpts = np.zeros((17, 3), dtype=np.float32)
    motion = np.zeros(8, dtype=np.float32)
    prob = clf(roi, kpts, motion)
    assert 0.0 <= prob <= 1.0


def test_classifier_forward_batch():
    clf = FallClassifier()
    roi = torch.zeros(2, 3, 96, 96, dtype=torch.float32)
    kpts = torch.zeros(2, 17, 3, dtype=torch.float32)
    motion = torch.zeros(2, 8, dtype=torch.float32)
    probs = clf.forward(roi, kpts, motion)
    assert probs.shape == (2, 1)
    assert torch.all((probs >= 0) & (probs <= 1))
