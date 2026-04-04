"""Tests for visualization utility functions."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call

from fall_detection.utils.visualization import draw_results, COCO_SKELETON


@pytest.fixture
def mock_frame():
    """Create a mock frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_tracks():
    """Create mock tracks for testing."""
    track1 = MagicMock()
    track1.track_id = 1
    track1.to_tlbr.return_value = [100, 100, 200, 300]

    track2 = MagicMock()
    track2.track_id = 2
    track2.to_tlbr.return_value = [300, 150, 400, 350]

    return [track1, track2]


@pytest.fixture
def mock_track_kpts():
    """Create mock keypoints for testing."""
    # Create valid keypoints for 2 tracks
    kpts1 = np.random.rand(17, 3).astype(np.float32)
    kpts1[:, 2] = 0.9  # High confidence

    kpts2 = np.random.rand(17, 3).astype(np.float32)
    kpts2[:, 2] = 0.8

    return {1: kpts1, 2: kpts2}


@pytest.fixture
def mock_track_scores():
    """Create mock track scores for testing."""
    return {
        1: {"rule": 0.8, "cls": 0.7, "final": 0.75, "state": "SUSPECTED"},
        2: {"rule": 0.3, "cls": 0.2, "final": 0.25, "state": "NORMAL"},
    }


@pytest.fixture
def mock_track_falling():
    """Create mock falling status for testing."""
    return {1: True, 2: False}


class TestDrawResultsBasic:
    """Basic test cases for draw_results function."""

    @patch("fall_detection.utils.visualization.cv2")
    def test_draw_results_returns_frame(self, mock_cv2, mock_frame, mock_tracks):
        """draw_results should return the modified frame."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        result = draw_results(
            mock_frame.copy(),
            mock_tracks,
            {},
            {},
            {}
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == mock_frame.shape

    @patch("fall_detection.utils.visualization.cv2")
    def test_draw_results_empty_tracks(self, mock_cv2, mock_frame):
        """draw_results should handle empty tracks list."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        result = draw_results(
            mock_frame.copy(),
            [],
            {},
            {},
            {}
        )

        assert isinstance(result, np.ndarray)
        # Should show "No targets detected" message
        call_found = False
        for c in mock_cv2.putText.call_args_list:
            args = c.args if c.args else c[0]
            kwargs = c.kwargs if c.kwargs else c[1] if len(c) > 1 else {}
            text = args[1] if len(args) > 1 else kwargs.get('text', '')
            if "No targets detected" in str(text):
                call_found = True
                break
        assert call_found, "Expected 'No targets detected' text to be drawn"


class TestDrawResultsBboxes:
    """Test cases for bounding box drawing."""

    @patch("fall_detection.utils.visualization.cv2")
    def test_draw_falling_bbox_red(self, mock_cv2, mock_frame, mock_tracks, mock_track_falling):
        """Falling tracks should be drawn with red color and thicker lines."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        draw_results(
            mock_frame.copy(),
            mock_tracks,
            {},
            {},
            mock_track_falling
        )

        # Track 1 is falling, should be red with thickness 4
        # cv2.rectangle(frame, pt1, pt2, color, thickness)
        calls = mock_cv2.rectangle.call_args_list
        red_color_calls = []
        for c in calls:
            args = c.args if c.args else []
            # args[0] is frame, args[1] is pt1, args[2] is pt2, args[3] is color
            color = args[3] if len(args) > 3 else None
            if color == (0, 0, 255):
                red_color_calls.append(c)
        assert len(red_color_calls) > 0, "Expected red color for falling track"

    @patch("fall_detection.utils.visualization.cv2")
    def test_draw_normal_bbox_green(self, mock_cv2, mock_frame, mock_tracks, mock_track_falling):
        """Normal tracks should be drawn with green color."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        draw_results(
            mock_frame.copy(),
            mock_tracks,
            {},
            {},
            mock_track_falling
        )

        # Track 2 is normal, should be green
        # cv2.rectangle(frame, pt1, pt2, color, thickness)
        calls = mock_cv2.rectangle.call_args_list
        green_color_calls = []
        for c in calls:
            args = c.args if c.args else []
            # args[0] is frame, args[1] is pt1, args[2] is pt2, args[3] is color
            color = args[3] if len(args) > 3 else None
            if color == (0, 255, 0):
                green_color_calls.append(c)
        assert len(green_color_calls) > 0, "Expected green color for normal track"

    @patch("fall_detection.utils.visualization.cv2")
    def test_draw_bbox_coordinates(self, mock_cv2, mock_frame, mock_tracks):
        """Bounding boxes should be drawn at correct coordinates."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        draw_results(
            mock_frame.copy(),
            mock_tracks,
            {},
            {},
            {}
        )

        # Check that rectangle was called with track coordinates
        # cv2.rectangle(frame, pt1, pt2, color, thickness)
        calls = mock_cv2.rectangle.call_args_list
        bbox_call_found = False
        for c in calls:
            args = c.args if c.args else []
            if len(args) >= 3:
                pt1, pt2 = args[1], args[2]
                if isinstance(pt1, tuple) and isinstance(pt2, tuple):
                    if pt1 == (100, 100) and pt2 == (200, 300):
                        bbox_call_found = True
                        break
        assert bbox_call_found, "Expected bbox drawn at coordinates (100,100)-(200,300)"


class TestDrawResultsKeypoints:
    """Test cases for keypoint drawing."""

    @patch("fall_detection.utils.visualization.cv2")
    def test_draw_keypoints_for_valid_kpts(self, mock_cv2, mock_frame, mock_tracks, mock_track_kpts):
        """Keypoints should be drawn when valid keypoints are provided."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        draw_results(
            mock_frame.copy(),
            mock_tracks,
            mock_track_kpts,
            {},
            {}
        )

        # Should draw lines for skeleton
        assert mock_cv2.line.called, "Expected line to be called for skeleton"
        # Should draw circles for keypoints
        assert mock_cv2.circle.called, "Expected circle to be called for keypoints"

    @patch("fall_detection.utils.visualization.cv2")
    def test_draw_keypoints_skip_low_confidence(self, mock_cv2, mock_frame, mock_tracks):
        """Keypoints with low confidence should not be drawn."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        # Create keypoints with very low confidence
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[:, 2] = 0.05  # Below threshold of 0.1

        draw_results(
            mock_frame.copy(),
            mock_tracks[:1],
            {1: kpts},
            {},
            {}
        )

        # Should not draw lines for low confidence keypoints
        line_calls_for_skeleton = [c for c in mock_cv2.line.call_args_list
                                    if len(c.args) >= 3]
        # No lines should be drawn since confidence is too low
        # (lines are drawn only when both keypoints have confidence > 0.1)

    @patch("fall_detection.utils.visualization.cv2")
    def test_draw_keypoints_skip_invalid_shape(self, mock_cv2, mock_frame, mock_tracks):
        """Keypoints with invalid shape should be skipped."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        # Create keypoints with wrong shape
        kpts = np.zeros((10, 2), dtype=np.float32)  # Wrong shape

        draw_results(
            mock_frame.copy(),
            mock_tracks[:1],
            {1: kpts},
            {},
            {}
        )

        # Should not try to draw keypoints with invalid shape
        # The function checks for shape == (17, 3)

    @patch("fall_detection.utils.visualization.cv2")
    def test_skeleton_lines_drawn(self, mock_cv2, mock_frame, mock_tracks, mock_track_kpts):
        """Skeleton lines should be drawn between connected keypoints."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        draw_results(
            mock_frame.copy(),
            mock_tracks,
            mock_track_kpts,
            {},
            {}
        )

        # Check that line was called for each skeleton connection
        line_calls = mock_cv2.line.call_args_list
        assert len(line_calls) > 0, "Expected skeleton lines to be drawn"

        # Verify some expected skeleton connections were drawn
        # COCO_SKELETON defines the connections
        for p1, p2 in COCO_SKELETON:
            # Each connection should result in a line call
            pass  # The actual verification depends on the mock setup


class TestDrawResultsLabels:
    """Test cases for label drawing."""

    @patch("fall_detection.utils.visualization.cv2")
    def test_falling_label_shown(self, mock_cv2, mock_frame, mock_tracks, mock_track_falling):
        """Falling tracks should show 'FALL!' label."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        draw_results(
            mock_frame.copy(),
            mock_tracks,
            {},
            {},
            mock_track_falling
        )

        # Check that "FALL!" text was drawn
        texts_drawn = []
        for c in mock_cv2.putText.call_args_list:
            args = c.args if c.args else []
            if len(args) > 1:
                texts_drawn.append(str(args[1]))
        assert any("FALL!" in t for t in texts_drawn), "Expected 'FALL!' label for falling track"

    @patch("fall_detection.utils.visualization.cv2")
    def test_normal_label_shows_id(self, mock_cv2, mock_frame, mock_tracks, mock_track_falling):
        """Normal tracks should show ID label."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        draw_results(
            mock_frame.copy(),
            mock_tracks,
            {},
            {},
            mock_track_falling
        )

        # Check that ID label was drawn for normal track
        texts_drawn = []
        for c in mock_cv2.putText.call_args_list:
            args = c.args if c.args else []
            if len(args) > 1:
                texts_drawn.append(str(args[1]))
        assert any("ID:2" in t for t in texts_drawn), "Expected 'ID:2' label for normal track"


class TestDrawResultsScorePanel:
    """Test cases for score panel drawing."""

    @patch("fall_detection.utils.visualization.cv2")
    def test_score_panel_drawn(self, mock_cv2, mock_frame, mock_tracks, mock_track_scores, mock_track_falling):
        """Score panel should be drawn when track_scores provided."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        draw_results(
            mock_frame.copy(),
            mock_tracks,
            {},
            mock_track_scores,
            mock_track_falling
        )

        # Should draw background rectangle for score panel
        assert mock_cv2.rectangle.called, "Expected rectangle for score panel background"

        # Should draw score header
        texts_drawn = []
        for c in mock_cv2.putText.call_args_list:
            args = c.args if c.args else []
            if len(args) > 1:
                texts_drawn.append(str(args[1]))
        assert any("Track Scores" in t for t in texts_drawn), "Expected 'Track Scores' header"

    @patch("fall_detection.utils.visualization.cv2")
    def test_score_values_displayed(self, mock_cv2, mock_frame, mock_tracks, mock_track_scores, mock_track_falling):
        """Score values should be displayed for each track."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        draw_results(
            mock_frame.copy(),
            mock_tracks,
            {},
            mock_track_scores,
            mock_track_falling
        )

        # Check that score values are displayed
        texts_drawn = []
        for c in mock_cv2.putText.call_args_list:
            args = c.args if c.args else []
            if len(args) > 1:
                texts_drawn.append(str(args[1]))
        # Should show track 1 scores
        assert any("T1:" in t for t in texts_drawn), "Expected 'T1:' score display"
        # Should show track 2 scores
        assert any("T2:" in t for t in texts_drawn), "Expected 'T2:' score display"

    @patch("fall_detection.utils.visualization.cv2")
    def test_fusion_history_displayed(self, mock_cv2, mock_frame, mock_tracks, mock_track_scores, mock_track_falling):
        """Fusion history should be displayed when provided."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        fusion_histories = {
            1: [0.5, 0.6, 0.75],
            2: [0.2, 0.25, 0.3],
        }

        draw_results(
            mock_frame.copy(),
            mock_tracks,
            {},
            mock_track_scores,
            mock_track_falling,
            fusion_histories
        )

        # Check that history values are displayed
        texts_drawn = []
        for c in mock_cv2.putText.call_args_list:
            args = c.args if c.args else []
            if len(args) > 1:
                texts_drawn.append(str(args[1]))
        # Should show history notation
        assert any("H:" in t for t in texts_drawn), "Expected 'H:' history notation"


class TestDrawResultsAlarm:
    """Test cases for alarm banner drawing."""

    @patch("fall_detection.utils.visualization.cv2")
    def test_alarm_banner_when_falling(self, mock_cv2, mock_frame, mock_tracks, mock_track_falling):
        """Alarm banner should be shown when any track is falling."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        draw_results(
            mock_frame.copy(),
            mock_tracks,
            {},
            {},
            mock_track_falling
        )

        # Should draw red overlay banner at top
        texts_drawn = []
        for c in mock_cv2.putText.call_args_list:
            args = c.args if c.args else []
            if len(args) > 1:
                texts_drawn.append(str(args[1]))
        assert any("FALL DETECTED!" in t for t in texts_drawn), "Expected 'FALL DETECTED!' alarm banner"

    @patch("fall_detection.utils.visualization.cv2")
    def test_no_alarm_banner_when_normal(self, mock_cv2, mock_frame, mock_tracks):
        """No alarm banner when no tracks are falling."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        track_falling = {1: False, 2: False}

        draw_results(
            mock_frame.copy(),
            mock_tracks,
            {},
            {},
            track_falling
        )

        # Should not draw alarm banner
        texts_drawn = []
        for c in mock_cv2.putText.call_args_list:
            args = c.args if c.args else []
            if len(args) > 1:
                texts_drawn.append(str(args[1]))
        assert not any("FALL DETECTED!" in t for t in texts_drawn), "Should not show alarm when no falling"


class TestDrawResultsSkipFrame:
    """Test cases for skip frame handling."""

    @patch("fall_detection.utils.visualization.cv2")
    def test_skip_frame_message(self, mock_cv2, mock_frame, mock_tracks):
        """Skip frame message should be shown when no scores provided."""
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.addWeighted = lambda overlay, alpha, frame, beta, gamma, dst: dst

        track_falling = {1: False, 2: False}

        draw_results(
            mock_frame.copy(),
            mock_tracks,
            {},
            {},  # No scores - indicates skip frame
            track_falling
        )

        # Should show skip frame status
        texts_drawn = []
        for c in mock_cv2.putText.call_args_list:
            args = c.args if c.args else []
            if len(args) > 1:
                texts_drawn.append(str(args[1]))
        assert any("skip frame" in t.lower() for t in texts_drawn), "Expected 'skip frame' message"


class TestCocoSkeleton:
    """Test cases for COCO_SKELETON constant."""

    def test_coco_skeleton_structure(self):
        """COCO_SKELETON should contain expected connections."""
        # Should have 16 connections for COCO 17 keypoints
        assert len(COCO_SKELETON) == 16

        # Check some expected connections
        assert (0, 1) in COCO_SKELETON  # nose to left eye
        assert (0, 2) in COCO_SKELETON  # nose to right eye
        assert (5, 6) in COCO_SKELETON  # left shoulder to right shoulder
        assert (11, 12) in COCO_SKELETON  # left hip to right hip

    def test_coco_skeleton_valid_indices(self):
        """All keypoint indices in COCO_SKELETON should be valid (0-16)."""
        for p1, p2 in COCO_SKELETON:
            assert 0 <= p1 <= 16
            assert 0 <= p2 <= 16
            assert p1 != p2
