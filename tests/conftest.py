"""Shared pytest fixtures for fall detection tests."""

import pytest
import numpy as np
import torch


@pytest.fixture
def sample_bbox():
    """Return a sample bounding box."""
    return [100, 100, 200, 300]


@pytest.fixture
def sample_keypoints():
    """Return sample COCO keypoints."""
    kpts = np.random.rand(17, 3) * 100
    kpts[:, 2] = 1.0  # All visible
    return kpts


@pytest.fixture
def sample_frame():
    """Return a sample image frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_detection():
    """Return a sample detection."""
    return {
        'bbox': [100, 100, 200, 300],
        'conf': 0.9,
        'class_id': 0
    }


@pytest.fixture
def device():
    """Return available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
