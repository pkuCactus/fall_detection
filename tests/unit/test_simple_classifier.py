"""Unit tests for SimpleFallClassifier model."""

import os
import tempfile
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from fall_detection.models.simple_classifier import SimpleFallClassifier, SimpleBasicBlock, SimpleResBlock


class TestSimpleBasicBlock:
    """Tests for SimpleBasicBlock."""

    def test_initialization(self):
        """Test block initialization with various parameters."""
        block = SimpleBasicBlock(in_channels=64, out_channels=64, stride=1)
        assert isinstance(block, nn.Module)

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        block = SimpleBasicBlock(in_channels=64, out_channels=64, stride=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_forward_pass_same_channels(self):
        """Test forward pass requires same input/output channels due to residual."""
        # SimpleBasicBlock uses x + self.model(x), so channels must match
        block = SimpleBasicBlock(in_channels=64, out_channels=64, stride=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_residual_connection(self):
        """Test that residual connection preserves information."""
        block = SimpleBasicBlock(in_channels=64, out_channels=64, stride=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        # Output should be close to input due to residual (after BN may differ)
        assert out.shape == x.shape


class TestSimpleResBlock:
    """Tests for SimpleResBlock."""

    def test_initialization(self):
        """Test block initialization."""
        block = SimpleResBlock(in_channels=64, out_channels=128, stride=2)
        assert isinstance(block, nn.Module)
        assert isinstance(block.model, nn.Sequential)
        assert isinstance(block.model2, nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass with downsampling."""
        block = SimpleResBlock(in_channels=64, out_channels=128, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 16, 16)

    def test_forward_pass_same_size(self):
        """Test forward pass without downsampling."""
        block = SimpleResBlock(in_channels=64, out_channels=64, stride=1)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)

    def test_skip_connection_shape_match(self):
        """Test skip connection with 1x1 conv for shape matching."""
        block = SimpleResBlock(in_channels=64, out_channels=128, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        # The 1x1 conv in model2 should handle the channel and spatial downsampling
        assert out.shape == (2, 128, 16, 16)


class TestSimpleFallClassifierInitialization:
    """Tests for SimpleFallClassifier initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        model = SimpleFallClassifier()
        assert model.num_classes == 2
        assert model.fall_class_idx == 1
        assert isinstance(model.backbone, nn.Sequential)
        assert isinstance(model.classifier_head, nn.Sequential)

    def test_custom_num_classes(self):
        """Test initialization with custom num_classes."""
        # Mock the default model loading to avoid file not found error
        with mock.patch('fall_detection.models.simple_classifier.torch.load', side_effect=FileNotFoundError()):
            model = SimpleFallClassifier(num_classes=5)
        assert model.num_classes == 5

    def test_custom_dropout(self):
        """Test initialization with custom dropout value."""
        model = SimpleFallClassifier(dropout=0.5)
        # Check dropout layer exists in classifier head
        dropout_layers = [m for m in model.classifier_head.modules() if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) == 1
        assert dropout_layers[0].p == 0.5

    def test_custom_fall_class_idx(self):
        """Test initialization with custom fall_class_idx."""
        model = SimpleFallClassifier(fall_class_idx=0)
        assert model.fall_class_idx == 0

    def test_zero_dropout(self):
        """Test initialization with zero dropout."""
        model = SimpleFallClassifier(dropout=0.0)
        dropout_layers = [m for m in model.classifier_head.modules() if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) == 1
        assert dropout_layers[0].p == 0.0

    def test_backbone_structure(self):
        """Test backbone contains expected layers."""
        model = SimpleFallClassifier()
        backbone_modules = list(model.backbone.children())
        # Should have Conv, BN, ReLU, Conv, BN, ReLU, ResBlock, ReLU, BasicBlock, ReLU, ResBlock, ReLU, BasicBlock, Pool, Flatten
        assert len(backbone_modules) >= 10

    def test_classifier_head_structure(self):
        """Test classifier head contains expected layers."""
        model = SimpleFallClassifier()
        head_modules = list(model.classifier_head.children())
        # Should have Linear, BN, Dropout, ReLU, Linear
        assert len(head_modules) == 5
        assert isinstance(head_modules[0], nn.Linear)
        assert isinstance(head_modules[1], nn.BatchNorm1d)
        assert isinstance(head_modules[2], nn.Dropout)
        assert isinstance(head_modules[3], nn.ReLU)
        assert isinstance(head_modules[4], nn.Linear)


class TestSimpleFallClassifierForwardPass:
    """Tests for SimpleFallClassifier forward pass."""

    def test_forward_single_sample(self):
        """Test forward pass with batch size 1."""
        model = SimpleFallClassifier()
        model.eval()  # BatchNorm requires eval mode for batch_size=1
        x = torch.randn(1, 3, 96, 96)
        out = model(x)
        assert out.shape == (1, 2)

    def test_forward_batch(self):
        """Test forward pass with batch size > 1."""
        model = SimpleFallClassifier()
        x = torch.randn(4, 3, 96, 96)
        out = model(x)
        assert out.shape == (4, 2)

    def test_forward_large_batch(self):
        """Test forward pass with large batch size."""
        model = SimpleFallClassifier()
        x = torch.randn(32, 3, 96, 96)
        out = model(x)
        assert out.shape == (32, 2)

    def test_forward_different_num_classes(self):
        """Test forward pass with different num_classes."""
        # Mock torch.load to prevent loading incompatible weights
        with mock.patch('fall_detection.models.simple_classifier.torch.load', side_effect=FileNotFoundError()):
            model = SimpleFallClassifier(num_classes=10)
        x = torch.randn(2, 3, 96, 96)
        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_output_type(self):
        """Test forward pass returns tensor."""
        model = SimpleFallClassifier()
        x = torch.randn(2, 3, 96, 96)
        out = model(x)
        assert isinstance(out, torch.Tensor)

    def test_forward_gradients(self):
        """Test gradients flow through the model."""
        model = SimpleFallClassifier()
        x = torch.randn(2, 3, 96, 96, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestSimpleFallClassifierFeatureExtraction:
    """Tests for feature extraction behavior."""

    def test_backbone_output_shape(self):
        """Test backbone produces correct feature shape."""
        model = SimpleFallClassifier()
        x = torch.randn(2, 3, 96, 96)
        features = model.backbone(x)
        # After AdaptiveAvgPool2d((1, 1)) and Flatten, should be (B, 128)
        assert features.shape == (2, 128)

    def test_backbone_output_batch_sizes(self):
        """Test backbone with various batch sizes."""
        model = SimpleFallClassifier()
        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 3, 96, 96)
            features = model.backbone(x)
            assert features.shape == (batch_size, 128)

    def test_feature_dimension(self):
        """Test feature dimension is 128."""
        model = SimpleFallClassifier()
        x = torch.randn(1, 3, 96, 96)
        features = model.backbone(x)
        assert features.shape[-1] == 128


class TestSimpleFallClassifierModelLoading:
    """Tests for model weight loading."""

    def test_load_from_temp_file(self):
        """Test loading weights from a temporary file."""
        model = SimpleFallClassifier()
        # Save current state
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
            torch.save(model.state_dict(), temp_path)

        try:
            # Load into new model
            model2 = SimpleFallClassifier(model_path=temp_path)
            # Verify weights loaded by comparing parameters
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file uses random initialization."""
        # Provide a non-existent path - model should still initialize
        model = SimpleFallClassifier(model_path="/nonexistent/path/model.pt")
        # Model should still be initialized
        assert model.num_classes == 2
        # Use eval mode and batch size >= 2 to avoid batch norm issues in train mode
        model.eval()
        x = torch.randn(1, 3, 96, 96)
        out = model(x)
        assert out.shape == (1, 2)

    def test_model_in_eval_mode_after_loading(self):
        """Test model is set to eval mode after loading weights."""
        model = SimpleFallClassifier()
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
            torch.save(model.state_dict(), temp_path)

        try:
            model2 = SimpleFallClassifier(model_path=temp_path)
            assert not model2.training
        finally:
            os.unlink(temp_path)


class TestSimpleFallClassifierDropout:
    """Tests for dropout behavior."""

    def test_dropout_training_mode(self):
        """Test dropout is active in training mode."""
        model = SimpleFallClassifier(dropout=0.5)
        model.train()
        x = torch.randn(2, 3, 96, 96)

        # Run multiple forward passes, outputs should differ due to dropout
        torch.manual_seed(42)
        out1 = model(x)
        out2 = model(x)
        # Due to dropout randomness, outputs should be different
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_dropout_eval_mode(self):
        """Test dropout is inactive in eval mode."""
        model = SimpleFallClassifier(dropout=0.5)
        model.eval()
        x = torch.randn(2, 3, 96, 96)

        # Run multiple forward passes, outputs should be identical
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_different_dropout_values(self):
        """Test model works with different dropout values."""
        for dropout in [0.0, 0.1, 0.3, 0.5, 0.8]:
            model = SimpleFallClassifier(dropout=dropout)
            model.eval()  # BatchNorm requires eval mode for batch_size=1
            x = torch.randn(1, 3, 96, 96)
            out = model(x)
            assert out.shape == (1, 2)


class TestSimpleFallClassifierEdgeCases:
    """Tests for edge cases."""

    def test_single_class(self):
        """Test with single class (edge case)."""
        with mock.patch('fall_detection.models.simple_classifier.torch.load', side_effect=FileNotFoundError()):
            model = SimpleFallClassifier(num_classes=1)
        model.eval()
        x = torch.randn(1, 3, 96, 96)
        out = model(x)
        assert out.shape == (1, 1)

    def test_large_num_classes(self):
        """Test with large number of classes."""
        with mock.patch('fall_detection.models.simple_classifier.torch.load', side_effect=FileNotFoundError()):
            model = SimpleFallClassifier(num_classes=1000)
        model.eval()
        x = torch.randn(1, 3, 96, 96)
        out = model(x)
        assert out.shape == (1, 1000)

    def test_input_channels(self):
        """Test that model expects 3 channel input."""
        model = SimpleFallClassifier()
        # First conv layer should expect 3 channels
        first_conv = model.backbone[0]
        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == 3

    def test_parameter_count(self):
        """Test model has trainable parameters."""
        model = SimpleFallClassifier()
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0

    def test_batch_norm_tracking(self):
        """Test batch norm tracks running statistics."""
        model = SimpleFallClassifier()
        model.train()
        x = torch.randn(4, 3, 96, 96)

        # Get first batch norm in backbone
        bn_layers = [m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_layers) > 0

        initial_running_mean = bn_layers[0].running_mean.clone()
        for _ in range(10):
            out = model(x)

        # Running mean should have been updated
        assert not torch.allclose(bn_layers[0].running_mean, initial_running_mean)


class TestSimpleFallClassifierDevice:
    """Tests for device compatibility."""

    def test_cpu_inference(self):
        """Test inference on CPU."""
        model = SimpleFallClassifier()
        model.eval()  # BatchNorm requires eval mode for batch_size=1
        x = torch.randn(1, 3, 96, 96)
        out = model(x)
        assert out.shape == (1, 2)
        assert out.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_inference(self):
        """Test inference on CUDA if available."""
        model = SimpleFallClassifier().cuda()
        model.eval()  # BatchNorm requires eval mode for batch_size=1
        x = torch.randn(1, 3, 96, 96).cuda()
        out = model(x)
        assert out.shape == (1, 2)
        assert out.device.type == 'cuda'


class TestSimpleFallClassifierStateDict:
    """Tests for model state dict operations."""

    def test_state_dict_keys(self):
        """Test state dict contains expected keys."""
        model = SimpleFallClassifier()
        state_dict = model.state_dict()

        # Should contain backbone and classifier_head parameters
        backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.')]
        head_keys = [k for k in state_dict.keys() if k.startswith('classifier_head.')]

        assert len(backbone_keys) > 0
        assert len(head_keys) > 0

    def test_load_state_dict(self):
        """Test loading state dict works correctly."""
        model1 = SimpleFallClassifier()
        model2 = SimpleFallClassifier()

        # Manually change model2 weights (only trainable params with actual values)
        with torch.no_grad():
            for p in model2.parameters():
                if p.numel() > 0 and p.dtype == torch.float32:
                    p.fill_(0.0)

        # Verify at least some weights are now different (skip BatchNorm defaults)
        different_count = sum(
            1 for p1, p2 in zip(model1.parameters(), model2.parameters())
            if not torch.allclose(p1, p2)
        )
        assert different_count > 0, "At least some parameters should be different"

        # Load state dict from model1 to model2
        model2.load_state_dict(model1.state_dict())

        # Now they should match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
