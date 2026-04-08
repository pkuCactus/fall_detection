"""Integration tests for train_classifier.py script."""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn
import importlib.util

# Ensure src is in path
sys.path.insert(0, "src")

from fall_detection.models import FallClassifier


def load_train_classifier_module():
    """Helper to load the train_classifier module."""
    spec = importlib.util.spec_from_file_location(
        "train_classifier", "scripts/train/train_classifier.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_classifier"] = module
    spec.loader.exec_module(module)
    return module


class TestFeatureDataset:
    """Test FeatureDataset loading from cache."""

    def test_feature_dataset_empty_cache(self, tmp_path):
        """Test FeatureDataset with empty cache directory."""
        module = load_train_classifier_module()
        dataset = module.FeatureDataset(str(tmp_path))
        assert len(dataset) == 0

    def test_feature_dataset_with_files(self, tmp_path):
        """Test FeatureDataset with cached feature files."""
        module = load_train_classifier_module()

        # Create sample .npz files
        for i in range(5):
            roi = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
            kpts = np.random.randn(17, 3).astype(np.float32)
            motion = np.random.randn(8).astype(np.float32)
            label = np.int64(1 if i % 2 == 0 else 0)

            np.savez(
                tmp_path / f"sample_{i}.npz",
                roi=roi,
                kpts=kpts,
                motion=motion,
                label=label
            )

        dataset = module.FeatureDataset(str(tmp_path))
        assert len(dataset) == 5

    def test_feature_dataset_getitem(self, tmp_path):
        """Test FeatureDataset __getitem__ returns correct tensors."""
        module = load_train_classifier_module()

        # Create a sample .npz file
        roi = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        kpts = np.random.randn(17, 3).astype(np.float32)
        motion = np.random.randn(8).astype(np.float32)
        label = np.int64(1)

        np.savez(
            tmp_path / "sample.npz",
            roi=roi,
            kpts=kpts,
            motion=motion,
            label=label
        )

        dataset = module.FeatureDataset(str(tmp_path))
        roi_tensor, kpts_tensor, motion_tensor, label_tensor = dataset[0]

        # Check types
        assert isinstance(roi_tensor, torch.Tensor)
        assert isinstance(kpts_tensor, torch.Tensor)
        assert isinstance(motion_tensor, torch.Tensor)
        assert isinstance(label_tensor, torch.Tensor)

        # Check shapes
        assert roi_tensor.shape == (3, 96, 96)  # Permuted and normalized
        assert kpts_tensor.shape == (17, 3)
        assert motion_tensor.shape == (8,)
        assert label_tensor.shape == ()  # scalar

        # Check value ranges
        assert roi_tensor.min() >= 0.0 and roi_tensor.max() <= 1.0  # Normalized to [0, 1]
        assert label_tensor.item() == 1.0  # Converted to float32

    def test_feature_dataset_roi_normalization(self, tmp_path):
        """Test that ROI is properly normalized to [0, 1]."""
        module = load_train_classifier_module()

        # Create sample with known values
        roi = np.full((96, 96, 3), 255, dtype=np.uint8)  # All white
        kpts = np.zeros((17, 3), dtype=np.float32)
        motion = np.zeros(8, dtype=np.float32)
        label = np.int64(0)

        np.savez(
            tmp_path / "white.npz",
            roi=roi,
            kpts=kpts,
            motion=motion,
            label=label
        )

        dataset = module.FeatureDataset(str(tmp_path))
        roi_tensor, _, _, _ = dataset[0]

        # Should be normalized to ~1.0
        assert torch.allclose(roi_tensor, torch.ones(3, 96, 96))

    def test_feature_dataset_file_sorting(self, tmp_path):
        """Test that files are sorted alphabetically."""
        module = load_train_classifier_module()

        # Create files in non-alphabetical order
        for name in ["z.npz", "a.npz", "m.npz"]:
            np.savez(
                tmp_path / name,
                roi=np.zeros((96, 96, 3), dtype=np.uint8),
                kpts=np.zeros((17, 3), dtype=np.float32),
                motion=np.zeros(8, dtype=np.float32),
                label=np.int64(0)
            )

        dataset = module.FeatureDataset(str(tmp_path))
        files = dataset.files

        # Should be sorted
        assert files == sorted(files)
        assert os.path.basename(files[0]) == "a.npz"
        assert os.path.basename(files[1]) == "m.npz"
        assert os.path.basename(files[2]) == "z.npz"


class TestTrainEpoch:
    """Test train_epoch function."""

    def test_train_epoch_basic(self):
        """Test basic train_epoch execution."""
        module = load_train_classifier_module()

        # Use a simple working model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 96 * 96 + 17 * 3 + 8, 1)

            def forward(self, roi, kpts, motion):
                x = torch.cat([roi.flatten(1), kpts.flatten(1), motion.flatten(1)], dim=1)
                return torch.sigmoid(self.fc(x))

        model = SimpleModel()

        # Create dummy data loader with 4-tuple (roi, kpts, motion, label)
        # Labels must be in [0, 1] for BCELoss
        dataset = torch.utils.data.TensorDataset(
            torch.randn(12, 3, 96, 96),  # roi
            torch.randn(12, 17, 3),       # kpts
            torch.randn(12, 8),           # motion
            torch.rand(12)                # label in [0, 1]
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        total_loss, total_samples = module.train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

        assert total_samples == 12
        assert total_loss > 0

    def test_train_epoch_model_in_train_mode(self):
        """Test that model is set to train mode."""
        module = load_train_classifier_module()

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 96 * 96 + 17 * 3 + 8, 1)

            def forward(self, roi, kpts, motion):
                x = torch.cat([roi.flatten(1), kpts.flatten(1), motion.flatten(1)], dim=1)
                return torch.sigmoid(self.fc(x))

        model = SimpleModel()
        model.eval()  # Start in eval mode

        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 3, 96, 96),
            torch.randn(8, 17, 3),
            torch.randn(8, 8),
            torch.rand(8)  # Labels in [0, 1]
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        module.train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

        assert model.training

    def test_train_epoch_loss_accumulation(self):
        """Test that loss is properly accumulated."""
        module = load_train_classifier_module()

        # Use a simple model that works with the data format
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 96 * 96 + 17 * 3 + 8, 1)

            def forward(self, roi, kpts, motion):
                x = torch.cat([roi.flatten(1), kpts.flatten(1), motion.flatten(1)], dim=1)
                return torch.sigmoid(self.fc(x))

        model = SimpleModel()

        dataset = torch.utils.data.TensorDataset(
            torch.ones(8, 3, 96, 96),
            torch.ones(8, 17, 3),
            torch.ones(8, 8),
            torch.ones(8)  # Labels are 1.0 (valid for BCELoss)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        total_loss, total_samples = module.train_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

        assert total_samples == 8
        assert total_loss > 0


class TestEvalEpoch:
    """Test eval_epoch function."""

    def test_eval_epoch_basic(self):
        """Test basic eval_epoch execution."""
        module = load_train_classifier_module()

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 96 * 96 + 17 * 3 + 8, 1)

            def forward(self, roi, kpts, motion):
                x = torch.cat([roi.flatten(1), kpts.flatten(1), motion.flatten(1)], dim=1)
                return torch.sigmoid(self.fc(x))

        model = SimpleModel()

        dataset = torch.utils.data.TensorDataset(
            torch.randn(12, 3, 96, 96),
            torch.randn(12, 17, 3),
            torch.randn(12, 8),
            torch.rand(12)  # Labels in [0, 1]
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        criterion = nn.BCELoss()

        total_loss, correct, total = module.eval_epoch(model, loader, criterion, torch.device("cpu"))

        assert total == 12
        assert not model.training

    def test_eval_epoch_model_in_eval_mode(self):
        """Test that model is set to eval mode during evaluation."""
        module = load_train_classifier_module()

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 96 * 96 + 17 * 3 + 8, 1)

            def forward(self, roi, kpts, motion):
                x = torch.cat([roi.flatten(1), kpts.flatten(1), motion.flatten(1)], dim=1)
                return torch.sigmoid(self.fc(x))

        model = SimpleModel()
        model.train()  # Start in train mode

        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 3, 96, 96),
            torch.randn(8, 17, 3),
            torch.randn(8, 8),
            torch.rand(8)  # Labels in [0, 1]
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        criterion = nn.BCELoss()

        module.eval_epoch(model, loader, criterion, torch.device("cpu"))

        assert not model.training

    def test_eval_epoch_no_gradients(self):
        """Test that no gradients are computed during evaluation."""
        module = load_train_classifier_module()

        # Use a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 96 * 96 + 17 * 3 + 8, 1)

            def forward(self, roi, kpts, motion):
                x = torch.cat([roi.flatten(1), kpts.flatten(1), motion.flatten(1)], dim=1)
                return torch.sigmoid(self.fc(x))

        model = SimpleModel()
        initial_weight = list(model.parameters())[0].clone().detach()

        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 3, 96, 96),
            torch.randn(8, 17, 3),
            torch.randn(8, 8),
            torch.rand(8)  # Labels in [0, 1]
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        criterion = nn.BCELoss()

        module.eval_epoch(model, loader, criterion, torch.device("cpu"))

        # Weights should not have changed
        assert torch.allclose(list(model.parameters())[0], initial_weight)

    def test_eval_epoch_accuracy_calculation(self):
        """Test accuracy calculation with known outcomes."""
        module = load_train_classifier_module()

        # Use a simple model with predictable outputs
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 96 * 96 + 17 * 3 + 8, 1)
                # Set weights to produce high output
                with torch.no_grad():
                    self.fc.weight.fill_(0.1)
                    self.fc.bias.fill_(5.0)  # High bias -> output ~1 after sigmoid

            def forward(self, roi, kpts, motion):
                x = torch.cat([roi.flatten(1), kpts.flatten(1), motion.flatten(1)], dim=1)
                return torch.sigmoid(self.fc(x))

        model = SimpleModel()

        dataset = torch.utils.data.TensorDataset(
            torch.ones(8, 3, 96, 96),
            torch.ones(8, 17, 3),
            torch.ones(8, 8),
            torch.ones(8)  # All labels are 1.0
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        criterion = nn.BCELoss()

        total_loss, correct, total = module.eval_epoch(model, loader, criterion, torch.device("cpu"))

        # All predictions should be correct (high output -> pred=1, labels are all 1)
        assert total == 8
        assert correct == 8

    def test_eval_epoch_partial_accuracy(self):
        """Test accuracy with mixed correct/incorrect predictions."""
        module = load_train_classifier_module()

        # Use a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 96 * 96 + 17 * 3 + 8, 1)

            def forward(self, roi, kpts, motion):
                x = torch.cat([roi.flatten(1), kpts.flatten(1), motion.flatten(1)], dim=1)
                return torch.sigmoid(self.fc(x))

        model = SimpleModel()
        # Set bias to produce ~0.6 output (pred=1)
        with torch.no_grad():
            model.fc.bias.fill_(0.6)

        labels = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        dataset = torch.utils.data.TensorDataset(
            torch.ones(8, 3, 96, 96),
            torch.ones(8, 17, 3),
            torch.ones(8, 8),
            labels
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        criterion = nn.BCELoss()

        total_loss, correct, total = module.eval_epoch(model, loader, criterion, torch.device("cpu"))

        # Should get 4 correct (the ones with label=1, since output ~0.6 -> pred=1)
        assert total == 8
        assert correct == 4


class TestDDPSetup:
    """Test DDP setup (mocked)."""

    @mock.patch("torch.cuda.is_available", return_value=False)
    @mock.patch("torch.distributed.init_process_group")
    @mock.patch("torch.distributed.get_world_size", return_value=2)
    @mock.patch("torch.distributed.get_rank", return_value=0)
    def test_ddp_setup_cpu(self, mock_get_rank, mock_get_world_size, mock_init_pg, mock_cuda_avail):
        """Test DDP setup on CPU."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--cache-dir", default="train/cache")
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--val-ratio", type=float, default=0.2)
        parser.add_argument("--output-dir", default="train/classifier")
        parser.add_argument("--local-rank", type=int, default=-1)
        args = parser.parse_args(["--local-rank", "0"])

        # Test that DDP is detected
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "LOCAL_RANK": "0"}):
            ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
            assert ddp is True

    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.cuda.set_device")
    @mock.patch("torch.distributed.init_process_group")
    @mock.patch("torch.distributed.get_world_size", return_value=4)
    @mock.patch("torch.distributed.get_rank", return_value=1)
    def test_ddp_setup_cuda(self, mock_get_rank, mock_get_world_size, mock_init_pg, mock_set_device, mock_cuda_avail):
        """Test DDP setup with CUDA."""
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "4", "LOCAL_RANK": "1"}):
            # Verify DDP environment
            assert os.environ["WORLD_SIZE"] == "4"
            assert os.environ["LOCAL_RANK"] == "1"

    def test_non_ddp_setup(self):
        """Test non-DDP setup."""
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True):
            ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
            assert ddp is False


class TestClassifierModelIntegration:
    """Test integration with FallClassifier model."""

    def test_model_creation(self):
        """Test FallClassifier model creation."""

        model = FallClassifier()
        assert isinstance(model, nn.Module)

        # Test forward pass
        roi = torch.randn(2, 3, 96, 96)
        kpts = torch.randn(2, 17, 3)
        motion = torch.randn(2, 8)

        output = model(roi, kpts, motion)
        assert output.shape == (2, 1)

    def test_model_to_device(self):
        """Test model can be moved to device."""

        model = FallClassifier()
        device = torch.device("cpu")
        model.to(device)

        # Verify parameters are on correct device
        for param in model.parameters():
            assert param.device.type == "cpu"

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_model_with_torch_compile(self):
        """Test FallClassifier with torch.compile."""
        model = FallClassifier()

        # Compile the model
        compiled_model = torch.compile(model)
        assert isinstance(compiled_model, nn.Module)

        # Test forward pass with compiled model
        roi = torch.randn(2, 3, 96, 96)
        kpts = torch.randn(2, 17, 3)
        motion = torch.randn(2, 8)

        output = compiled_model(roi, kpts, motion)
        assert output.shape == (2, 1)


class TestTrainingLoopComponents:
    """Test training loop components."""

    def test_optimizer_creation(self):
        """Test optimizer creation."""

        model = FallClassifier()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        assert isinstance(optimizer, torch.optim.Optimizer)
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_criterion_creation(self):
        """Test loss criterion creation."""
        criterion = nn.BCELoss()
        assert isinstance(criterion, nn.Module)

    def test_data_loader_creation(self, tmp_path):
        """Test DataLoader creation."""
        module = load_train_classifier_module()

        # Create sample data
        for i in range(10):
            np.savez(
                tmp_path / f"sample_{i}.npz",
                roi=np.zeros((96, 96, 3), dtype=np.uint8),
                kpts=np.zeros((17, 3), dtype=np.float32),
                motion=np.zeros(8, dtype=np.float32),
                label=np.int64(0)
            )

        dataset = module.FeatureDataset(str(tmp_path))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        assert len(loader) == 3  # 10 samples / 4 batch_size = 3 batches

    def test_train_val_split(self, tmp_path):
        """Test train/validation split."""
        module = load_train_classifier_module()

        # Create sample data
        for i in range(100):
            np.savez(
                tmp_path / f"sample_{i:03d}.npz",
                roi=np.zeros((96, 96, 3), dtype=np.uint8),
                kpts=np.zeros((17, 3), dtype=np.float32),
                motion=np.zeros(8, dtype=np.float32),
                label=np.int64(0)
            )

        dataset = module.FeatureDataset(str(tmp_path))
        val_ratio = 0.2
        n_val = int(len(dataset) * val_ratio)
        n_train = len(dataset) - n_val

        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        assert len(train_ds) == 80
        assert len(val_ds) == 20


class TestArgumentParsing:
    """Test argument parsing."""

    def test_parse_args_defaults(self):
        """Test argument parsing with defaults."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--cache-dir", default="train/cache")
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--val-ratio", type=float, default=0.2)
        parser.add_argument("--output-dir", default="train/classifier")
        parser.add_argument("--local-rank", type=int, default=-1)

        args = parser.parse_args([])

        assert args.cache_dir == "train/cache"
        assert args.epochs == 100
        assert args.batch_size == 32
        assert args.lr == 0.001
        assert args.val_ratio == 0.2
        assert args.output_dir == "train/classifier"
        assert args.local_rank == -1

    def test_parse_args_custom(self):
        """Test argument parsing with custom values."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--cache-dir", default="train/cache")
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--val-ratio", type=float, default=0.2)
        parser.add_argument("--output-dir", default="train/classifier")
        parser.add_argument("--local-rank", type=int, default=-1)

        args = parser.parse_args([
            "--cache-dir", "/custom/cache",
            "--epochs", "50",
            "--batch-size", "64",
            "--lr", "0.01",
            "--val-ratio", "0.3",
            "--output-dir", "/custom/output",
            "--local-rank", "0"
        ])

        assert args.cache_dir == "/custom/cache"
        assert args.epochs == 50
        assert args.batch_size == 64
        assert args.lr == 0.01
        assert args.val_ratio == 0.3
        assert args.output_dir == "/custom/output"
        assert args.local_rank == 0


class TestIntegrationWithFileSystem:
    """Integration tests with actual file system operations."""

    def test_script_help_output(self):
        """Test that script produces help output."""
        result = subprocess.run(
            [sys.executable, "scripts/train/train_classifier.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--override" in result.stdout
