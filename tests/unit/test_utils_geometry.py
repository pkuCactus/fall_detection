"""Tests for geometry utility functions."""

import pytest
import numpy as np

from fall_detection.utils.geometry import iou


class TestIoU:
    """Test cases for the IoU (Intersection over Union) function."""

    def test_identical_boxes(self):
        """IoU of identical boxes should be 1.0."""
        bbox = [0, 0, 10, 10]
        result = iou(bbox, bbox)
        assert result == 1.0

    def test_no_overlap(self):
        """IoU of non-overlapping boxes should be 0.0."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [20, 20, 30, 30]
        result = iou(bbox1, bbox2)
        assert result == 0.0

    def test_partial_overlap(self):
        """IoU of partially overlapping boxes should be between 0 and 1."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [5, 5, 15, 15]
        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        # IoU: 25 / 175 = 0.142857...
        result = iou(bbox1, bbox2)
        expected = 25.0 / 175.0
        assert abs(result - expected) < 1e-6

    def test_complete_containment(self):
        """IoU when one box is completely inside another."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [2, 2, 8, 8]
        # Intersection: 6x6 = 36
        # Union: 100
        # IoU: 36 / 100 = 0.36
        result = iou(bbox1, bbox2)
        expected = 36.0 / 100.0
        assert abs(result - expected) < 1e-6

    def test_edge_touching(self):
        """IoU of boxes that only touch at edges should be 0."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [10, 0, 20, 10]  # touches at x=10
        result = iou(bbox1, bbox2)
        assert result == 0.0

    def test_corner_touching(self):
        """IoU of boxes that only touch at corners should be 0."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [10, 10, 20, 20]  # touches at corner (10, 10)
        result = iou(bbox1, bbox2)
        assert result == 0.0

    def test_zero_area_box(self):
        """IoU with a zero-area box should be 0."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [5, 5, 5, 5]  # zero area
        result = iou(bbox1, bbox2)
        assert result == 0.0

    def test_both_zero_area_boxes(self):
        """IoU of two zero-area boxes should be 0."""
        bbox1 = [0, 0, 0, 0]
        bbox2 = [5, 5, 5, 5]
        result = iou(bbox1, bbox2)
        assert result == 0.0

    def test_negative_coordinates(self):
        """IoU should work with negative coordinates."""
        bbox1 = [-10, -10, 0, 0]
        bbox2 = [-5, -5, 5, 5]
        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        result = iou(bbox1, bbox2)
        expected = 25.0 / 175.0
        assert abs(result - expected) < 1e-6

    def test_floating_point_coordinates(self):
        """IoU should work with floating point coordinates."""
        bbox1 = [0.5, 0.5, 10.5, 10.5]
        bbox2 = [5.0, 5.0, 15.0, 15.0]
        # Intersection: 5.5 x 5.5 = 30.25
        # Area1: 10 x 10 = 100
        # Area2: 10 x 10 = 100
        # Union: 100 + 100 - 30.25 = 169.75
        result = iou(bbox1, bbox2)
        expected = 30.25 / 169.75
        assert abs(result - expected) < 1e-6

    def test_commutative(self):
        """IoU should be commutative: iou(a, b) == iou(b, a)."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [5, 5, 15, 15]
        result1 = iou(bbox1, bbox2)
        result2 = iou(bbox2, bbox1)
        assert result1 == result2

    def test_with_numpy_arrays(self):
        """IoU should work with numpy arrays as input."""
        bbox1 = np.array([0, 0, 10, 10])
        bbox2 = np.array([5, 5, 15, 15])
        result = iou(bbox1, bbox2)
        expected = 25.0 / 175.0
        assert abs(result - expected) < 1e-6

    def test_with_lists(self):
        """IoU should work with lists as input."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [5, 5, 15, 15]
        result = iou(bbox1, bbox2)
        expected = 25.0 / 175.0
        assert abs(result - expected) < 1e-6

    def test_with_tuples(self):
        """IoU should work with tuples as input."""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (5, 5, 15, 15)
        result = iou(bbox1, bbox2)
        expected = 25.0 / 175.0
        assert abs(result - expected) < 1e-6

    def test_large_coordinates(self):
        """IoU should work with large coordinate values."""
        bbox1 = [0, 0, 10000, 10000]
        bbox2 = [5000, 5000, 15000, 15000]
        # Intersection: 5000 x 5000 = 25,000,000
        # Union: 100M + 100M - 25M = 175M
        result = iou(bbox1, bbox2)
        expected = 25_000_000.0 / 175_000_000.0
        assert abs(result - expected) < 1e-6

    def test_horizontal_overlap_only(self):
        """IoU for boxes overlapping only horizontally."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [5, 20, 15, 30]  # y ranges don't overlap
        result = iou(bbox1, bbox2)
        assert result == 0.0

    def test_vertical_overlap_only(self):
        """IoU for boxes overlapping only vertically."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [20, 5, 30, 15]  # x ranges don't overlap
        result = iou(bbox1, bbox2)
        assert result == 0.0
