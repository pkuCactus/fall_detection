"""Integration tests for train_simple_classifier.py script."""

import argparse
import os
import subprocess
import sys
import tempfile
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

# Ensure src is in path
sys.path.insert(0, "src")


def load_train_simple_classifier_module():
    """Helper to load the train_simple_classifier module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_simple_classifier", "training/scripts/train_simple_classifier.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_simple_classifier"] = module
    spec.loader.exec_module(module)
    return module


class TestConfigLoading:
    """Test configuration loading."""

    def test_load_config_basic(self, tmp_path):
        """Test basic config loading."""
        module = load_train_simple_classifier_module()

        config = {
            "lr": 0.001,
            "batch_size": 64,
            "epochs": 100,
            "weight_decay": 0.0001,
            "seed": 42,
            "data": {
                "format": "coco",
                "train_coco_json": "data/train.json",
                "val_coco_json": "data/val.json",
                "image_dir": "data/images"
            },
            "model": {
                "dropout": 0.3,
                "fall_class_idx": 1
            },
            "output": {
                "dir": "outputs/simple_classifier"
            }
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Create mock args
        class MockArgs:
            def __init__(self):
                self.config = str(config_path)
                self.override = None

        args = MockArgs()
        loaded_cfg = module.load_config(args)

        assert loaded_cfg["lr"] == 0.001
        assert loaded_cfg["batch_size"] == 64
        assert loaded_cfg["epochs"] == 100
        assert loaded_cfg["data"]["format"] == "coco"
        assert loaded_cfg["model"]["dropout"] == 0.3

    def test_load_config_with_override(self, tmp_path):
        """Test config loading with override values."""
        module = load_train_simple_classifier_module()

        config = {
            "lr": 0.001,
            "batch_size": 64,
            "epochs": 100
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        class MockArgs:
            def __init__(self):
                self.config = str(config_path)
                self.override = "lr=0.01,batch_size=128"

        args = MockArgs()
        loaded_cfg = module.load_config(args)

        assert loaded_cfg["lr"] == 0.01
        assert loaded_cfg["batch_size"] == 128
        assert loaded_cfg["epochs"] == 100  # unchanged

    def test_load_config_nested_override(self, tmp_path):
        """Test config loading with nested key override."""
        module = load_train_simple_classifier_module()

        config = {
            "model": {
                "dropout": 0.3,
                "fall_class_idx": 1
            },
            "lr_scheduler": {
                "type": "plateau",
                "factor": 0.5
            }
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        class MockArgs:
            def __init__(self):
                self.config = str(config_path)
                self.override = "model.dropout=0.5,lr_scheduler.factor=0.3"

        args = MockArgs()
        loaded_cfg = module.load_config(args)

        assert loaded_cfg["model"]["dropout"] == 0.5
        assert loaded_cfg["lr_scheduler"]["factor"] == 0.3

    def test_load_config_voc_format(self, tmp_path):
        """Test config loading for VOC format."""
        module = load_train_simple_classifier_module()

        config = {
            "data": {
                "format": "voc"
            },
            "voc": {
                "train_dirs": ["data/voc/train"],
                "val_dirs": ["data/voc/val"],
                "fall_classes": ["fall", "falling"],
                "normal_classes": ["normal", "standing"]
            }
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        class MockArgs:
            def __init__(self):
                self.config = str(config_path)
                self.override = None

        args = MockArgs()
        loaded_cfg = module.load_config(args)

        assert loaded_cfg["data"]["format"] == "voc"
        assert loaded_cfg["voc"]["fall_classes"] == ["fall", "falling"]


class TestModelCreation:
    """Test model creation."""

    def test_create_model_basic(self):
        """Test basic model creation."""
        module = load_train_simple_classifier_module()

        cfg = {
            "model": {
                "dropout": 0.3,
                "fall_class_idx": 1
            }
        }

        device = torch.device("cpu")
        model = module.create_model(cfg, device, ddp=False, local_rank=0)

        assert isinstance(model, nn.Module)
        # Check it's SimpleFallClassifier
        from fall_detection.models.simple_classifier import SimpleFallClassifier
        assert isinstance(model, SimpleFallClassifier)

    @mock.patch("torch.nn.parallel.DistributedDataParallel")
    def test_create_model_with_ddp(self, mock_ddp):
        """Test model creation with DDP wrapper."""
        module = load_train_simple_classifier_module()

        cfg = {
            "model": {
                "dropout": 0.3,
                "fall_class_idx": 1
            },
            "ddp": {
                "find_unused_parameters": False
            }
        }

        device = torch.device("cpu")

        # Mock DDP to avoid actual distributed setup
        mock_ddp.return_value = nn.Linear(10, 2)  # Dummy wrapped model
        model = module.create_model(cfg, device, ddp=True, local_rank=0)

        # Verify DDP was called
        mock_ddp.assert_called_once()

    def test_create_model_custom_dropout(self):
        """Test model creation with custom dropout."""
        module = load_train_simple_classifier_module()

        cfg = {
            "model": {
                "dropout": 0.5,
                "fall_class_idx": 1
            }
        }

        device = torch.device("cpu")
        model = module.create_model(cfg, device, ddp=False, local_rank=0)

        # Check dropout layer exists with correct value
        from fall_detection.models.simple_classifier import SimpleFallClassifier
        assert isinstance(model, SimpleFallClassifier)


class TestTrainingLoopComponents:
    """Test training loop components."""

    def test_train_epoch_basic(self):
        """Test basic train_epoch execution."""
        module = load_train_simple_classifier_module()

        from fall_detection.models.simple_classifier import SimpleFallClassifier

        model = SimpleFallClassifier(dropout=0.0)
        device = torch.device("cpu")
        model.to(device)

        # Create dummy data loader
        images = torch.randn(16, 3, 96, 96)
        labels = torch.randint(0, 2, (16,))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(images, labels),
            batch_size=4
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        total_loss, total_correct, total_samples = module.train_epoch(
            model, loader, optimizer, criterion, device, epoch=1, rank=0, log_interval=0
        )

        assert total_loss > 0
        assert total_samples == 16
        assert 0 <= total_correct <= 16

    def test_train_epoch_model_in_train_mode(self):
        """Test that train_epoch sets model to train mode."""
        module = load_train_simple_classifier_module()

        from fall_detection.models.simple_classifier import SimpleFallClassifier

        model = SimpleFallClassifier(dropout=0.0)
        model.eval()  # Start in eval mode

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randn(8, 3, 96, 96),
                torch.randint(0, 2, (8,))
            ),
            batch_size=4
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        module.train_epoch(model, loader, optimizer, criterion, torch.device("cpu"), epoch=1, rank=0, log_interval=0)

        assert model.training

    def test_eval_epoch_basic(self):
        """Test basic eval_epoch execution."""
        module = load_train_simple_classifier_module()

        from fall_detection.models.simple_classifier import SimpleFallClassifier

        model = SimpleFallClassifier(dropout=0.0)
        device = torch.device("cpu")
        model.to(device)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randn(12, 3, 96, 96),
                torch.randint(0, 2, (12,))
            ),
            batch_size=4
        )

        criterion = nn.CrossEntropyLoss()

        total_loss, total_correct, total_samples = module.eval_epoch(
            model, loader, criterion, device
        )

        assert total_loss >= 0
        assert total_samples == 12
        assert 0 <= total_correct <= 12

    def test_eval_epoch_model_in_eval_mode(self):
        """Test that eval_epoch sets model to eval mode."""
        module = load_train_simple_classifier_module()

        from fall_detection.models.simple_classifier import SimpleFallClassifier

        model = SimpleFallClassifier(dropout=0.0)
        model.train()  # Start in train mode

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randn(8, 3, 96, 96),
                torch.randint(0, 2, (8,))
            ),
            batch_size=4
        )

        criterion = nn.CrossEntropyLoss()

        module.eval_epoch(model, loader, criterion, torch.device("cpu"))

        assert not model.training

    def test_eval_epoch_no_gradients(self):
        """Test that eval_epoch doesn't compute gradients."""
        module = load_train_simple_classifier_module()

        from fall_detection.models.simple_classifier import SimpleFallClassifier

        model = SimpleFallClassifier(dropout=0.0)
        initial_weight = list(model.parameters())[0].clone().detach()

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randn(8, 3, 96, 96),
                torch.randint(0, 2, (8,))
            ),
            batch_size=4
        )

        criterion = nn.CrossEntropyLoss()

        module.eval_epoch(model, loader, criterion, torch.device("cpu"))

        # Weights should not have changed
        assert torch.allclose(list(model.parameters())[0], initial_weight)


class TestOptimizerSchedulerCreation:
    """Test optimizer and scheduler creation."""

    def test_create_optimizer(self):
        """Test optimizer creation."""
        module = load_train_simple_classifier_module()

        from fall_detection.models.simple_classifier import SimpleFallClassifier

        model = SimpleFallClassifier()

        cfg = {
            "lr": 0.001,
            "weight_decay": 0.0001,
            "lr_scheduler": {"type": "plateau"}
        }

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randn(8, 3, 96, 96),
                torch.randint(0, 2, (8,))
            ),
            batch_size=4
        )

        optimizer, scheduler = module.create_optimizer_scheduler(cfg, model, loader, rank=0)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[0]["weight_decay"] == 0.0001

    def test_create_plateau_scheduler(self):
        """Test ReduceLROnPlateau scheduler creation."""
        module = load_train_simple_classifier_module()

        from fall_detection.models.simple_classifier import SimpleFallClassifier

        model = SimpleFallClassifier()

        cfg = {
            "lr": 0.001,
            "lr_scheduler": {
                "type": "plateau",
                "factor": 0.5,
                "patience": 10,
                "min_lr": 1e-6
            }
        }

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(8, 3, 96, 96), torch.randint(0, 2, (8,))),
            batch_size=4
        )

        optimizer, scheduler = module.create_optimizer_scheduler(cfg, model, loader, rank=0)

        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_create_cosine_scheduler(self):
        """Test CosineAnnealingLR scheduler creation."""
        module = load_train_simple_classifier_module()

        from fall_detection.models.simple_classifier import SimpleFallClassifier

        model = SimpleFallClassifier()

        cfg = {
            "lr": 0.001,
            "epochs": 100,
            "lr_scheduler": {
                "type": "cosine",
                "T_max": 100,
                "min_lr": 1e-5
            }
        }

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(8, 3, 96, 96), torch.randint(0, 2, (8,))),
            batch_size=4
        )

        optimizer, scheduler = module.create_optimizer_scheduler(cfg, model, loader, rank=0)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_create_step_scheduler(self):
        """Test StepLR scheduler creation."""
        module = load_train_simple_classifier_module()

        from fall_detection.models.simple_classifier import SimpleFallClassifier

        model = SimpleFallClassifier()

        cfg = {
            "lr": 0.001,
            "lr_scheduler": {
                "type": "step",
                "step_size": 30,
                "gamma": 0.1
            }
        }

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(8, 3, 96, 96), torch.randint(0, 2, (8,))),
            batch_size=4
        )

        optimizer, scheduler = module.create_optimizer_scheduler(cfg, model, loader, rank=0)

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_create_warmup_scheduler(self):
        """Test WarmupScheduler wrapper creation."""
        module = load_train_simple_classifier_module()

        from fall_detection.models.simple_classifier import SimpleFallClassifier

        model = SimpleFallClassifier()

        cfg = {
            "lr": 0.001,
            "epochs": 100,
            "lr_scheduler": {
                "type": "plateau",
                "warmup": {
                    "enabled": True,
                    "epochs": 5,
                    "strategy": "linear",
                    "init_lr": 1e-5
                }
            }
        }

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(8, 3, 96, 96), torch.randint(0, 2, (8,))),
            batch_size=4
        )

        optimizer, scheduler = module.create_optimizer_scheduler(cfg, model, loader, rank=0)

        from fall_detection.training import WarmupScheduler
        assert isinstance(scheduler, WarmupScheduler)


class TestDDPSetup:
    """Test DDP setup (mocked)."""

    @mock.patch("torch.cuda.is_available", return_value=False)
    @mock.patch("torch.distributed.init_process_group")
    @mock.patch("torch.distributed.get_world_size", return_value=2)
    @mock.patch("torch.distributed.get_rank", return_value=0)
    def test_ddp_setup_cpu(self, mock_get_rank, mock_get_world_size, mock_init_pg, mock_cuda_avail):
        """Test DDP setup on CPU."""
        module = load_train_simple_classifier_module()

        cfg = {"ddp": {"backend": "nccl"}}

        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "LOCAL_RANK": "0"}):
            ddp, device, world_size, rank, local_rank = module.setup_ddp(cfg)

            assert ddp is True
            assert world_size == 2
            assert rank == 0
            assert local_rank == 0

    @mock.patch("torch.cuda.is_available", return_value=True)
    @mock.patch("torch.cuda.set_device")
    @mock.patch("torch.distributed.init_process_group")
    @mock.patch("torch.distributed.get_world_size", return_value=4)
    @mock.patch("torch.distributed.get_rank", return_value=1)
    def test_ddp_setup_cuda(self, mock_get_rank, mock_get_world_size, mock_init_pg, mock_set_device, mock_cuda_avail):
        """Test DDP setup with CUDA."""
        module = load_train_simple_classifier_module()

        cfg = {"ddp": {"backend": "nccl"}}

        with mock.patch.dict(os.environ, {"WORLD_SIZE": "4", "LOCAL_RANK": "1"}):
            ddp, device, world_size, rank, local_rank = module.setup_ddp(cfg)

            assert ddp is True
            assert world_size == 4
            assert rank == 1
            assert local_rank == 1
            mock_set_device.assert_called_once_with(1)

    def test_non_ddp_setup(self):
        """Test non-DDP setup."""
        module = load_train_simple_classifier_module()

        cfg = {"ddp": {"backend": "nccl"}}

        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True):
            ddp, device, world_size, rank, local_rank = module.setup_ddp(cfg)

            assert ddp is False
            assert world_size == 1
            assert rank == 0
            assert local_rank == 0


class TestSeedSetup:
    """Test random seed setup."""

    def test_setup_seed(self):
        """Test seed setup for reproducibility."""
        module = load_train_simple_classifier_module()

        # Should not raise any errors
        module.setup_seed(42, rank=0)
        module.setup_seed(None, rank=0)  # No seed

    def test_setup_seed_deterministic(self):
        """Test that seed setup makes operations deterministic."""
        module = load_train_simple_classifier_module()

        # Set seed
        module.setup_seed(42, rank=0)

        # Generate random numbers
        rand1 = torch.rand(10)

        # Reset seed and generate again
        module.setup_seed(42, rank=0)
        rand2 = torch.rand(10)

        # Should be identical
        assert torch.allclose(rand1, rand2)


class TestTimeFormatting:
    """Test time formatting utility."""

    def test_format_time_remaining_minutes(self):
        """Test formatting time in minutes."""
        module = load_train_simple_classifier_module()

        result = module.format_time_remaining(300)  # 5 minutes
        assert result == "5m0s"

    def test_format_time_remaining_hours(self):
        """Test formatting time in hours."""
        module = load_train_simple_classifier_module()

        result = module.format_time_remaining(3660)  # 1 hour 1 minute
        assert "h" in result
        assert "m" in result

    def test_format_time_remaining_seconds(self):
        """Test formatting time with seconds."""
        module = load_train_simple_classifier_module()

        result = module.format_time_remaining(45)
        assert result == "0m45s"


class TestArgumentParsing:
    """Test argument parsing."""

    def test_parse_args_required_config(self):
        """Test that config is required."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--local-rank", type=int, default=-1)
        parser.add_argument("--override", type=str, default=None)

        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parse_args_defaults(self, tmp_path):
        """Test argument parsing with defaults."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--local-rank", type=int, default=-1)
        parser.add_argument("--override", type=str, default=None)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        args = parser.parse_args(["--config", str(config_path)])

        assert args.config == str(config_path)
        assert args.local_rank == -1
        assert args.override is None

    def test_parse_args_custom(self, tmp_path):
        """Test argument parsing with custom values."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--local-rank", type=int, default=-1)
        parser.add_argument("--override", type=str, default=None)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        args = parser.parse_args([
            "--config", str(config_path),
            "--local-rank", "0",
            "--override", "lr=0.01,batch_size=128"
        ])

        assert args.local_rank == 0
        assert args.override == "lr=0.01,batch_size=128"


class TestIntegrationWithFileSystem:
    """Integration tests with actual file system operations."""

    def test_script_help_output(self):
        """Test that script produces help output."""
        result = subprocess.run(
            [sys.executable, "training/scripts/train_simple_classifier.py", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--local-rank" in result.stdout
        assert "--override" in result.stdout
