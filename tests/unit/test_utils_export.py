"""Tests for export utility functions."""

import pytest
from unittest.mock import patch, MagicMock

from fall_detection.utils.export import export_classifier_onnx, export_simple_classifier_onnx


class TestExportClassifierOnnx:
    """Test cases for the export_classifier_onnx function."""

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.FallClassifier")
    def test_export_classifier_creates_model(self, mock_classifier_class, mock_export):
        """Export should create a FallClassifier instance."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_classifier_onnx("test.onnx")

        mock_classifier_class.assert_called_once()
        mock_model.eval.assert_called_once()

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.FallClassifier")
    def test_export_classifier_calls_torch_export(self, mock_classifier_class, mock_export):
        """Export should call torch.onnx.export with correct arguments."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_classifier_onnx("test.onnx")

        mock_export.assert_called_once()
        args, kwargs = mock_export.call_args

        # Check model is passed
        assert args[0] == mock_model

        # Check input tensors have correct shapes
        roi, kpts, motion = args[1]
        assert roi.shape == (1, 3, 96, 96)
        assert kpts.shape == (1, 17, 3)
        assert motion.shape == (1, 8)

        # Check output path
        assert args[2] == "test.onnx"

        # Check kwargs
        assert kwargs["input_names"] == ["roi", "kpts", "motion"]
        assert kwargs["output_names"] == ["prob"]
        assert kwargs["opset_version"] == 11
        assert "dynamic_axes" in kwargs

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.FallClassifier")
    def test_export_classifier_default_path(self, mock_classifier_class, mock_export):
        """Export should use default path when not specified."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_classifier_onnx()

        args, _ = mock_export.call_args
        assert args[2] == "fall_classifier.onnx"

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.FallClassifier")
    def test_export_classifier_dynamic_axes(self, mock_classifier_class, mock_export):
        """Export should set up dynamic axes for batch dimension."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_classifier_onnx("test.onnx")

        args, kwargs = mock_export.call_args
        dynamic_axes = kwargs["dynamic_axes"]

        assert dynamic_axes["roi"] == {0: "batch_size"}
        assert dynamic_axes["kpts"] == {0: "batch_size"}
        assert dynamic_axes["motion"] == {0: "batch_size"}
        assert dynamic_axes["prob"] == {0: "batch_size"}

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.FallClassifier")
    def test_export_classifier_tensor_dtypes(self, mock_classifier_class, mock_export):
        """Export should create tensors with float32 dtype."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_classifier_onnx("test.onnx")

        args, _ = mock_export.call_args
        roi, kpts, motion = args[1]

        assert roi.dtype == pytest.approx(0, abs=1) or str(roi.dtype) == "torch.float32"
        assert kpts.dtype == pytest.approx(0, abs=1) or str(kpts.dtype) == "torch.float32"
        assert motion.dtype == pytest.approx(0, abs=1) or str(motion.dtype) == "torch.float32"

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.FallClassifier")
    def test_export_classifier_prints_message(self, mock_classifier_class, mock_export, capsys):
        """Export should print success message."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_classifier_onnx("test.onnx")

        captured = capsys.readouterr()
        assert "ONNX exported to test.onnx" in captured.out

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.FallClassifier")
    def test_export_classifier_custom_path(self, mock_classifier_class, mock_export):
        """Export should use custom path when specified."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_classifier_onnx("custom/path/model.onnx")

        args, _ = mock_export.call_args
        assert args[2] == "custom/path/model.onnx"
        captured = capsys.readouterr() if hasattr(pytest, 'capsys') else None


class TestExportSimpleClassifierOnnx:
    """Test cases for the export_simple_classifier_onnx function."""

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.SimpleFallClassifier")
    def test_export_simple_classifier_creates_model(self, mock_classifier_class, mock_export):
        """Export should create a SimpleFallClassifier instance."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_simple_classifier_onnx("test.onnx")

        mock_classifier_class.assert_called_once()
        mock_model.eval.assert_called_once()

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.SimpleFallClassifier")
    def test_export_simple_classifier_calls_torch_export(self, mock_classifier_class, mock_export):
        """Export should call torch.onnx.export with correct arguments."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_simple_classifier_onnx("test.onnx")

        mock_export.assert_called_once()
        args, kwargs = mock_export.call_args

        # Check model is passed
        assert args[0] == mock_model

        # Check input tensor has correct shape
        x = args[1]
        assert x.shape == (1, 3, 96, 96)

        # Check output path
        assert args[2] == "test.onnx"

        # Check kwargs
        assert kwargs["input_names"] == ["input"]
        assert kwargs["output_names"] == ["logits"]
        assert kwargs["opset_version"] == 11
        assert "dynamic_axes" in kwargs

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.SimpleFallClassifier")
    def test_export_simple_classifier_default_path(self, mock_classifier_class, mock_export):
        """Export should use default path when not specified."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_simple_classifier_onnx()

        args, _ = mock_export.call_args
        assert args[2] == "simple_fall_classifier.onnx"

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.SimpleFallClassifier")
    def test_export_simple_classifier_dynamic_axes(self, mock_classifier_class, mock_export):
        """Export should set up dynamic axes for batch dimension."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_simple_classifier_onnx("test.onnx")

        args, kwargs = mock_export.call_args
        dynamic_axes = kwargs["dynamic_axes"]

        assert dynamic_axes["input"] == {0: "batch_size"}
        assert dynamic_axes["logits"] == {0: "batch_size"}

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.SimpleFallClassifier")
    def test_export_simple_classifier_tensor_dtype(self, mock_classifier_class, mock_export):
        """Export should create tensor with float32 dtype."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_simple_classifier_onnx("test.onnx")

        args, _ = mock_export.call_args
        x = args[1]

        assert str(x.dtype) == "torch.float32"

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.SimpleFallClassifier")
    def test_export_simple_classifier_prints_message(self, mock_classifier_class, mock_export, capsys):
        """Export should print success message."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_simple_classifier_onnx("test.onnx")

        captured = capsys.readouterr()
        assert "ONNX exported to test.onnx" in captured.out

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.SimpleFallClassifier")
    def test_export_simple_classifier_custom_path(self, mock_classifier_class, mock_export):
        """Export should use custom path when specified."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model

        export_simple_classifier_onnx("models/simple.onnx")

        args, _ = mock_export.call_args
        assert args[2] == "models/simple.onnx"


class TestExportErrorHandling:
    """Test error handling in export functions."""

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.FallClassifier")
    def test_export_classifier_handles_export_error(self, mock_classifier_class, mock_export):
        """Export should propagate torch.onnx.export errors."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model
        mock_export.side_effect = RuntimeError("Export failed")

        with pytest.raises(RuntimeError, match="Export failed"):
            export_classifier_onnx("test.onnx")

    @patch("fall_detection.utils.export.torch.onnx.export")
    @patch("fall_detection.utils.export.SimpleFallClassifier")
    def test_export_simple_classifier_handles_export_error(self, mock_classifier_class, mock_export):
        """Export should propagate torch.onnx.export errors."""
        mock_model = MagicMock()
        mock_classifier_class.return_value = mock_model
        mock_export.side_effect = RuntimeError("Export failed")

        with pytest.raises(RuntimeError, match="Export failed"):
            export_simple_classifier_onnx("test.onnx")
