"""Tests for WarmupScheduler."""

import pytest
import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from fall_detection.utils.scheduler import WarmupScheduler


class TestWarmupSchedulerInitialization:
    """Tests for WarmupScheduler initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(optimizer, None)

        assert scheduler.optimizer == optimizer
        assert scheduler.scheduler is None
        assert scheduler.warmup_steps == 0
        assert scheduler.warmup_strategy == 'linear'
        assert scheduler.warmup_init_lr == 1e-5
        assert scheduler.current_step == 0
        assert scheduler.base_lrs == [0.1]
        assert scheduler.warmup_finished is False

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        optimizer = Adam([torch.randn(10)], lr=0.01)
        base_scheduler = StepLR(optimizer, step_size=10)
        scheduler = WarmupScheduler(
            optimizer,
            base_scheduler,
            warmup_steps=100,
            warmup_strategy='constant',
            warmup_init_lr=1e-6
        )

        assert scheduler.optimizer == optimizer
        assert scheduler.scheduler == base_scheduler
        assert scheduler.warmup_steps == 100
        assert scheduler.warmup_strategy == 'constant'
        assert scheduler.warmup_init_lr == 1e-6
        assert scheduler.current_step == 0
        assert scheduler.base_lrs == [0.01]
        assert scheduler.warmup_finished is False

    def test_init_captures_base_lrs(self):
        """Test that base_lrs correctly captures optimizer's initial learning rates."""
        optimizer = SGD([
            {'params': [torch.randn(10)], 'lr': 0.1},
            {'params': [torch.randn(10)], 'lr': 0.01},
        ])
        scheduler = WarmupScheduler(optimizer, None)

        assert scheduler.base_lrs == [0.1, 0.01]

    def test_init_with_none_scheduler(self):
        """Test initialization with None scheduler."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(optimizer, None, warmup_steps=10)

        assert scheduler.scheduler is None


class TestLinearWarmupStrategy:
    """Tests for linear warmup strategy."""

    def test_linear_warmup_progression(self):
        """Test that linear warmup correctly progresses from init_lr to base_lr."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=10,
            warmup_strategy='linear',
            warmup_init_lr=0.0
        )

        learning_rates = []
        for _ in range(10):
            scheduler.step_batch()
            learning_rates.append(optimizer.param_groups[0]['lr'])

        # Check first step is at warmup_init_lr (alpha=0/10=0)
        assert learning_rates[0] == 0.0

        # Check progression is linear: alpha = step/10, so LRs are 0.0, 0.01, ..., 0.09
        expected_lrs = [0.0 + i * 0.01 for i in range(10)]
        for actual, expected in zip(learning_rates, expected_lrs):
            assert abs(actual - expected) < 1e-10

        # First step after warmup reaches base_lr
        scheduler.step_batch()
        assert optimizer.param_groups[0]['lr'] == 0.1

    def test_linear_warmup_nonzero_init(self):
        """Test linear warmup with non-zero initial learning rate."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=5,
            warmup_strategy='linear',
            warmup_init_lr=0.05
        )

        learning_rates = []
        for _ in range(5):
            scheduler.step_batch()
            learning_rates.append(optimizer.param_groups[0]['lr'])

        # Check progression: 0.05, 0.06, 0.07, 0.08, 0.09
        expected_lrs = [0.05 + i * 0.01 for i in range(5)]
        for actual, expected in zip(learning_rates, expected_lrs):
            assert abs(actual - expected) < 1e-10


class TestConstantWarmupStrategy:
    """Tests for constant warmup strategy."""

    def test_constant_warmup_stays_constant(self):
        """Test that constant warmup keeps LR at warmup_init_lr."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=10,
            warmup_strategy='constant',
            warmup_init_lr=1e-4
        )

        for _ in range(10):
            scheduler.step_batch()
            assert optimizer.param_groups[0]['lr'] == 1e-4

    def test_constant_warmup_jumps_to_base_lr(self):
        """Test that constant warmup jumps to base_lr after warmup ends."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=5,
            warmup_strategy='constant',
            warmup_init_lr=1e-4
        )

        # Warmup phase
        for _ in range(5):
            scheduler.step_batch()
            assert optimizer.param_groups[0]['lr'] == 1e-4

        # First step after warmup
        scheduler.step_batch()
        assert optimizer.param_groups[0]['lr'] == 0.1


class TestWarmupCompletion:
    """Tests for warmup completion behavior."""

    def test_warmup_finished_flag(self):
        """Test that warmup_finished flag is set correctly."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=5,
            warmup_strategy='linear'
        )

        assert scheduler.warmup_finished is False

        # During warmup (steps 0-4)
        for _ in range(5):
            assert scheduler.warmup_finished is False
            scheduler.step_batch()

        # Step 5: first step after warmup ends, sets warmup_finished=True
        scheduler.step_batch()
        assert scheduler.warmup_finished is True

    def test_transition_to_base_lr(self):
        """Test transition to base learning rate after warmup."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=5,
            warmup_strategy='linear',
            warmup_init_lr=0.0
        )

        # Run through warmup (steps 0-4)
        for _ in range(5):
            scheduler.step_batch()

        # Step 5: transition to base_lr
        scheduler.step_batch()

        # After warmup, LR should be at base_lr
        assert optimizer.param_groups[0]['lr'] == 0.1

        # Continue stepping, LR should stay at base_lr
        for _ in range(10):
            scheduler.step_batch()
            assert optimizer.param_groups[0]['lr'] == 0.1

    def test_zero_warmup_steps(self):
        """Test behavior with zero warmup steps."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=0,
            warmup_strategy='linear'
        )

        # With 0 warmup steps, warmup should immediately be finished
        assert scheduler.warmup_finished is False
        scheduler.step_batch()
        assert scheduler.warmup_finished is True
        assert optimizer.param_groups[0]['lr'] == 0.1


class TestStepMethod:
    """Tests for step() method."""

    def test_step_no_scheduler(self):
        """Test step() with no base scheduler."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(optimizer, None, warmup_steps=0)

        # Should not raise any errors
        scheduler.step()
        scheduler.step(metrics=0.5)

    def test_step_with_scheduler_no_metrics(self):
        """Test step() with scheduler that doesn't need metrics."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        base_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        scheduler = WarmupScheduler(optimizer, base_scheduler, warmup_steps=0)

        # Mark warmup as finished
        scheduler.warmup_finished = True

        initial_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        assert optimizer.param_groups[0]['lr'] == initial_lr * 0.1

    def test_step_with_scheduler_and_metrics(self):
        """Test step() with scheduler that uses metrics (ReduceLROnPlateau)."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        base_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=0)
        scheduler = WarmupScheduler(optimizer, base_scheduler, warmup_steps=0)

        # Mark warmup as finished
        scheduler.warmup_finished = True

        # First call with bad metric
        scheduler.step(metrics=1.0)

        # Second call with worse metric should reduce LR
        scheduler.step(metrics=2.0)
        assert optimizer.param_groups[0]['lr'] < 0.1

    def test_step_during_warmup(self):
        """Test that step() doesn't call base scheduler during warmup."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        base_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            base_scheduler,
            warmup_steps=5,
            warmup_strategy='linear'
        )

        # During warmup, step() should not affect LR
        for _ in range(5):
            initial_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            # LR should be controlled by warmup, not base scheduler
            assert optimizer.param_groups[0]['lr'] == initial_lr

    def test_step_after_warmup(self):
        """Test that step() calls base scheduler after warmup."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        base_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        scheduler = WarmupScheduler(
            optimizer,
            base_scheduler,
            warmup_steps=5,
            warmup_strategy='linear'
        )

        # Complete warmup
        for _ in range(6):
            scheduler.step_batch()

        # Now step() should affect LR
        initial_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        assert optimizer.param_groups[0]['lr'] == initial_lr * 0.5


class TestStepBatchMethod:
    """Tests for step_batch() method."""

    def test_step_batch_increments_step(self):
        """Test that step_batch() increments current_step."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(optimizer, None, warmup_steps=10)

        for i in range(20):
            assert scheduler.current_step == i
            scheduler.step_batch()

    def test_step_batch_after_warmup(self):
        """Test step_batch() behavior after warmup is complete."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=5,
            warmup_strategy='linear',
            warmup_init_lr=0.0
        )

        # Complete warmup (steps 0-4 are warmup, step 5 transitions to base_lr)
        for _ in range(6):
            scheduler.step_batch()

        # After warmup, LR should stay at base_lr
        assert optimizer.param_groups[0]['lr'] == 0.1

        # Continue stepping
        for _ in range(10):
            scheduler.step_batch()
            assert optimizer.param_groups[0]['lr'] == 0.1
            assert scheduler.warmup_finished is True


class TestGetLastLR:
    """Tests for get_last_lr() method."""

    def test_get_last_lr_single_group(self):
        """Test get_last_lr() with single param group."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=5,
            warmup_strategy='linear',
            warmup_init_lr=0.0
        )

        # Initial LR before any step_batch is the optimizer's LR (0.1), not warmup_init_lr
        # because step_batch hasn't been called yet
        assert scheduler.get_last_lr() == [0.1]

        scheduler.step_batch()
        # After first step_batch: alpha = 0/5 = 0, so LR = 0.0 + 0 * (0.1 - 0.0) = 0.0
        assert scheduler.get_last_lr() == [0.0]

        scheduler.step_batch()
        # After second step_batch: alpha = 1/5 = 0.2, so LR = 0.0 + 0.2 * 0.1 = 0.02
        assert abs(scheduler.get_last_lr()[0] - 0.02) < 1e-10

    def test_get_last_lr_returns_optimizer_lr(self):
        """Test that get_last_lr() returns current optimizer LR values."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=5,
            warmup_strategy='constant',
            warmup_init_lr=1e-4
        )

        for _ in range(10):
            lrs = scheduler.get_last_lr()
            assert lrs == [optimizer.param_groups[0]['lr']]
            scheduler.step_batch()


class TestIsWarmupProperty:
    """Tests for is_warmup property."""

    def test_is_warmup_during_warmup(self):
        """Test is_warmup returns True during warmup."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(optimizer, None, warmup_steps=5)

        for _ in range(5):
            assert scheduler.is_warmup is True
            scheduler.step_batch()

    def test_is_warmup_after_warmup(self):
        """Test is_warmup returns False after warmup."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(optimizer, None, warmup_steps=5)

        for _ in range(5):
            scheduler.step_batch()

        assert scheduler.is_warmup is False

        for _ in range(10):
            assert scheduler.is_warmup is False
            scheduler.step_batch()

    def test_is_warmup_zero_steps(self):
        """Test is_warmup with zero warmup steps."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(optimizer, None, warmup_steps=0)

        assert scheduler.is_warmup is False


class TestMultipleParamGroups:
    """Tests for multiple parameter groups."""

    def test_multiple_groups_linear_warmup(self):
        """Test linear warmup with multiple param groups."""
        optimizer = SGD([
            {'params': [torch.randn(10)], 'lr': 0.1},
            {'params': [torch.randn(10)], 'lr': 0.01},
        ])
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=5,
            warmup_strategy='linear',
            warmup_init_lr=0.0
        )

        for step in range(5):
            scheduler.step_batch()
            alpha = step / 5
            expected_lr1 = 0.0 + alpha * 0.1
            expected_lr2 = 0.0 + alpha * 0.01
            lrs = scheduler.get_last_lr()
            assert abs(lrs[0] - expected_lr1) < 1e-10
            assert abs(lrs[1] - expected_lr2) < 1e-10

    def test_multiple_groups_constant_warmup(self):
        """Test constant warmup with multiple param groups."""
        optimizer = SGD([
            {'params': [torch.randn(10)], 'lr': 0.1},
            {'params': [torch.randn(10)], 'lr': 0.01},
        ])
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=5,
            warmup_strategy='constant',
            warmup_init_lr=1e-4
        )

        for _ in range(5):
            scheduler.step_batch()
            lrs = scheduler.get_last_lr()
            assert lrs[0] == 1e-4
            assert lrs[1] == 1e-4

        # After warmup, each group should return to its own base_lr
        scheduler.step_batch()
        lrs = scheduler.get_last_lr()
        assert lrs[0] == 0.1
        assert lrs[1] == 0.01

    def test_multiple_groups_step_scheduler(self):
        """Test step() with multiple param groups and base scheduler."""
        optimizer = SGD([
            {'params': [torch.randn(10)], 'lr': 0.1},
            {'params': [torch.randn(10)], 'lr': 0.01},
        ])
        base_scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        scheduler = WarmupScheduler(
            optimizer,
            base_scheduler,
            warmup_steps=5,
            warmup_strategy='linear'
        )

        # Complete warmup
        for _ in range(6):
            scheduler.step_batch()

        # Apply step
        scheduler.step()
        lrs = scheduler.get_last_lr()
        assert lrs[0] == 0.05  # 0.1 * 0.5
        assert lrs[1] == 0.005  # 0.01 * 0.5


class TestReduceLROnPlateauIntegration:
    """Tests for integration with ReduceLROnPlateau."""

    def test_reduce_lr_on_plateau_basic(self):
        """Test basic ReduceLROnPlateau integration."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        base_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=0, factor=0.5)
        scheduler = WarmupScheduler(
            optimizer,
            base_scheduler,
            warmup_steps=5,
            warmup_strategy='linear',
            warmup_init_lr=0.0
        )

        # Complete warmup
        for _ in range(6):
            scheduler.step_batch()

        assert optimizer.param_groups[0]['lr'] == 0.1

        # First metric
        scheduler.step(metrics=1.0)
        assert optimizer.param_groups[0]['lr'] == 0.1

        # Worse metric should reduce LR
        scheduler.step(metrics=2.0)
        assert optimizer.param_groups[0]['lr'] == 0.05

    def test_reduce_lr_on_plateau_during_warmup(self):
        """Test that ReduceLROnPlateau is not called during warmup."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        base_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=0, factor=0.5)
        scheduler = WarmupScheduler(
            optimizer,
            base_scheduler,
            warmup_steps=5,
            warmup_strategy='constant',
            warmup_init_lr=0.01
        )

        # During warmup, calling step() with metrics should not affect LR
        # Note: step_batch must be called first to set LR to warmup_init_lr
        for _ in range(5):
            scheduler.step_batch()
            assert optimizer.param_groups[0]['lr'] == 0.01
            scheduler.step(metrics=float('inf'))  # Would normally reduce LR
            assert optimizer.param_groups[0]['lr'] == 0.01
        # Need one more step_batch to finish warmup (step 5 triggers warmup_finished)
        scheduler.step_batch()

    def test_reduce_lr_on_plateau_after_warmup(self):
        """Test ReduceLROnPlateau behavior after warmup."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        base_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            base_scheduler,
            warmup_steps=3,
            warmup_strategy='linear'
        )

        # Complete warmup (need 4 steps to finish: steps 0,1,2 are warmup, step 3 finishes)
        for _ in range(4):
            scheduler.step_batch()

        # With patience=1, need 2 consecutive worse metrics
        scheduler.step(metrics=1.0)
        assert optimizer.param_groups[0]['lr'] == 0.1

        scheduler.step(metrics=1.1)
        assert optimizer.param_groups[0]['lr'] == 0.1

        scheduler.step(metrics=1.2)
        assert abs(optimizer.param_groups[0]['lr'] - 0.01) < 1e-10


class TestCosineAnnealingIntegration:
    """Tests for integration with CosineAnnealingLR."""

    def test_cosine_annealing_after_warmup(self):
        """Test CosineAnnealingLR integration after warmup."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        base_scheduler = CosineAnnealingLR(optimizer, T_max=10)
        scheduler = WarmupScheduler(
            optimizer,
            base_scheduler,
            warmup_steps=5,
            warmup_strategy='linear',
            warmup_init_lr=0.0
        )

        # Complete warmup (need 6 steps: steps 0-4 are warmup, step 5 finishes)
        for _ in range(6):
            scheduler.step_batch()

        assert optimizer.param_groups[0]['lr'] == 0.1

        # Cosine annealing should decrease LR
        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(5):
            scheduler.step()
            assert optimizer.param_groups[0]['lr'] <= initial_lr
            initial_lr = optimizer.param_groups[0]['lr']


class TestEdgeCases:
    """Tests for edge cases."""

    def test_warmup_init_lr_equals_base_lr(self):
        """Test when warmup_init_lr equals base_lr."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=5,
            warmup_strategy='linear',
            warmup_init_lr=0.1
        )

        for _ in range(10):
            scheduler.step_batch()
            assert optimizer.param_groups[0]['lr'] == 0.1

    def test_warmup_init_lr_greater_than_base_lr(self):
        """Test when warmup_init_lr is greater than base_lr."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=5,
            warmup_strategy='linear',
            warmup_init_lr=0.2
        )

        # Linear warmup should decrease from 0.2 to 0.1
        learning_rates = []
        for _ in range(5):
            scheduler.step_batch()
            learning_rates.append(optimizer.param_groups[0]['lr'])

        assert learning_rates[0] == 0.2
        assert learning_rates[-1] < 0.2

    def test_large_warmup_steps(self):
        """Test with large number of warmup steps."""
        optimizer = SGD([torch.randn(10)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=1000,
            warmup_strategy='linear',
            warmup_init_lr=0.0
        )

        for _ in range(1000):
            assert scheduler.is_warmup is True
            scheduler.step_batch()

        # Step 1000: transition from warmup to base_lr
        scheduler.step_batch()

        assert scheduler.is_warmup is False
        # After 1000 warmup steps, step 1000 transitions to base_lr
        # Note: step 999 has alpha = 999/1000 = 0.999, so LR = 0.0999
        # Step 1000 transitions to base_lr = 0.1
        assert optimizer.param_groups[0]['lr'] == 0.1

    def test_step_batch_without_optimizer_update(self):
        """Test that step_batch() doesn't require optimizer step."""
        optimizer = SGD([torch.randn(10, requires_grad=True)], lr=0.1)
        scheduler = WarmupScheduler(
            optimizer,
            None,
            warmup_steps=5,
            warmup_strategy='linear'
        )

        # step_batch() should work without calling optimizer.step()
        for _ in range(10):
            scheduler.step_batch()

        assert scheduler.current_step == 10
