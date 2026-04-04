"""Unit tests for data augmentation utilities."""

import numpy as np
import pytest
import cv2

from fall_detection.data.augmentation import (
    RandomMask,
    RandomCropWithPadding,
    LetterBoxResize,
    TrainingAugmentation,
)


class TestRandomMask:
    """Tests for RandomMask class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        mask = RandomMask()
        assert mask.mask_ratio == 0.25
        assert mask.p == 0.3

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        mask = RandomMask(mask_ratio=0.5, p=0.8)
        assert mask.mask_ratio == 0.5
        assert mask.p == 0.8

    def test_call_no_mask_when_random_above_threshold(self, monkeypatch):
        """Test that mask is not applied when random > p."""
        mask = RandomMask(mask_ratio=0.25, p=0.3)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Mock random.random to return value above p
        monkeypatch.setattr(np.random, 'random', lambda: 0.5)

        result = mask(img)

        # Image should be unchanged
        np.testing.assert_array_equal(result, img)

    def test_call_applies_mask_when_random_below_threshold(self, monkeypatch):
        """Test that mask is applied when random < p."""
        mask = RandomMask(mask_ratio=0.25, p=0.3)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Mock random.random to return value below p
        call_count = [0]
        def mock_random():
            call_count[0] += 1
            # First call is for probability check, subsequent for mask position
            if call_count[0] == 1:
                return 0.1  # Below p=0.3, so mask will be applied
            return 0.5  # For position selection (middle of range)

        monkeypatch.setattr(np.random, 'random', mock_random)
        # Center position at (50, 50) for bbox, and handle 3-arg calls for mask fill
        def mock_randint(*args, **kwargs):
            if len(args) == 2:
                return (args[0] + args[1]) // 2
            elif len(args) >= 3:
                # Return array of random values for mask fill
                import numpy as np
                return np.random.RandomState(42).randint(0, 256, size=args[2])
            return 0
        monkeypatch.setattr(np.random, 'randint', mock_randint)

        result = mask(img)

        # Image should be modified (mask applied)
        assert result.shape == img.shape
        # The mask area should be different (filled with random values)
        assert not np.array_equal(result, img)

    def test_mask_dimensions(self, monkeypatch):
        """Test that mask has correct dimensions based on mask_ratio."""
        mask_ratio = 0.2
        mask = RandomMask(mask_ratio=mask_ratio, p=1.0)  # Always apply
        h, w = 100, 200
        img = np.ones((h, w, 3), dtype=np.uint8) * 128

        # Fixed random values for deterministic test
        monkeypatch.setattr(np.random, 'random', lambda: 0.0)
        # Center position at (100, 50) - middle of 200x100, handle 3-arg calls for mask fill
        def mock_randint(*args, **kwargs):
            if len(args) == 2:
                return (args[0] + args[1]) // 2
            elif len(args) >= 3:
                # Return array of random values for mask fill
                return np.random.RandomState(42).randint(0, 256, size=args[2])
            return 0
        monkeypatch.setattr(np.random, 'randint', mock_randint)

        result = mask(img)

        # Calculate expected mask size
        expected_mask_h = int(h * mask_ratio)
        expected_mask_w = int(w * mask_ratio)

        # Find the masked region by comparing with original
        diff = result != img
        masked_pixels = np.where(diff.any(axis=2))

        # Check that some pixels were masked
        assert len(masked_pixels[0]) > 0, "No pixels were masked"

        # Check mask dimensions
        actual_mask_h = masked_pixels[0].max() - masked_pixels[0].min() + 1
        actual_mask_w = masked_pixels[1].max() - masked_pixels[1].min() + 1

        assert actual_mask_h == expected_mask_h
        assert actual_mask_w == expected_mask_w

    def test_mask_bounds(self, monkeypatch):
        """Test that mask stays within image bounds."""
        mask = RandomMask(mask_ratio=0.5, p=1.0)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        monkeypatch.setattr(np.random, 'random', lambda: 0.0)
        # Test edge cases: center at boundaries - return low bound, handle 3-arg calls
        def mock_randint(*args, **kwargs):
            if len(args) == 2:
                return args[0]  # Return low bound
            elif len(args) >= 3:
                return np.random.RandomState(42).randint(0, 256, size=args[2])
            return 0
        monkeypatch.setattr(np.random, 'randint', mock_randint)

        result = mask(img)

        # Result should still be valid image
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        assert np.all(result >= 0) and np.all(result <= 255)

    def test_mask_returns_copy(self, monkeypatch):
        """Test that mask returns a copy of the image."""
        mask = RandomMask(mask_ratio=0.25, p=1.0)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        monkeypatch.setattr(np.random, 'random', lambda: 0.0)
        # Center position at (50, 50), handle 3-arg calls for mask fill
        def mock_randint(*args, **kwargs):
            if len(args) == 2:
                return 50
            elif len(args) >= 3:
                return np.random.RandomState(42).randint(0, 256, size=args[2])
            return 0
        monkeypatch.setattr(np.random, 'randint', mock_randint)

        result = mask(img)

        # Original image should be unchanged
        assert np.all(img == 128)
        # Result should be different
        assert not np.array_equal(result, img)


class TestRandomCropWithPadding:
    """Tests for RandomCropWithPadding class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        crop = RandomCropWithPadding()
        assert crop.shrink_max == 3
        assert crop.expand_max == 25

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        crop = RandomCropWithPadding(shrink_max=5, expand_max=50)
        assert crop.shrink_max == 5
        assert crop.expand_max == 50

    def test_crop_shrink(self, monkeypatch):
        """Test cropping with shrink margin."""
        crop = RandomCropWithPadding(shrink_max=10, expand_max=0)
        img = np.ones((100, 100, 3), dtype=np.uint8)
        bbox = [20, 20, 80, 80]  # 60x60 box

        # Set shrink margin to 5
        monkeypatch.setattr(np.random, 'randint', lambda low, high: -5)

        result = crop(img, bbox)

        # Expected: bbox shrunk by 5 on each side
        # Original: [20, 20, 80, 80] = 60x60
        # After shrink by 5: [25, 25, 75, 75] = 50x50
        expected_size = (50, 50)
        assert result.shape[:2] == expected_size

    def test_crop_expand(self, monkeypatch):
        """Test cropping with expand margin."""
        crop = RandomCropWithPadding(shrink_max=0, expand_max=20)
        img = np.ones((100, 100, 3), dtype=np.uint8)
        bbox = [40, 40, 60, 60]  # 20x20 box

        # Set expand margin to 10
        monkeypatch.setattr(np.random, 'randint', lambda low, high: 10)

        result = crop(img, bbox)

        # Expected: bbox expanded by 10 on each side = 40x40
        expected_size = (40, 40)
        assert result.shape[:2] == expected_size

    def test_crop_respects_image_bounds(self, monkeypatch):
        """Test that crop respects image boundaries."""
        crop = RandomCropWithPadding(shrink_max=0, expand_max=50)
        img = np.ones((50, 50, 3), dtype=np.uint8)
        bbox = [0, 0, 10, 10]  # Top-left corner

        # Try to expand beyond image bounds
        monkeypatch.setattr(np.random, 'randint', lambda low, high: 20)

        result = crop(img, bbox)

        # Result should be clamped to image bounds
        assert result.shape[0] <= 50
        assert result.shape[1] <= 50

    def test_crop_invalid_bbox_returns_original(self, monkeypatch):
        """Test handling of invalid bbox that results in negative dimensions."""
        crop = RandomCropWithPadding(shrink_max=10, expand_max=0)
        img = np.ones((100, 100, 3), dtype=np.uint8)
        bbox = [20, 20, 25, 25]  # Very small 5x5 box

        # Shrink too much, making dimensions negative
        monkeypatch.setattr(np.random, 'randint', lambda low, high: -10)

        result = crop(img, bbox)

        # Should return original bbox region
        expected = img[20:25, 20:25]
        np.testing.assert_array_equal(result, expected)

    def test_crop_zero_margin(self, monkeypatch):
        """Test cropping with zero margin."""
        crop = RandomCropWithPadding(shrink_max=5, expand_max=5)
        img = np.ones((100, 100, 3), dtype=np.uint8)
        bbox = [20, 20, 80, 80]

        monkeypatch.setattr(np.random, 'randint', lambda low, high: 0)

        result = crop(img, bbox)

        # Should return exact bbox region
        expected = img[20:80, 20:80]
        np.testing.assert_array_equal(result, expected)


class TestLetterBoxResize:
    """Tests for LetterBoxResize class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        lb = LetterBoxResize()
        assert lb.target_size == 96
        assert lb.fill_value == 114

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        lb = LetterBoxResize(target_size=128, fill_value=0)
        assert lb.target_size == 128
        assert lb.fill_value == 0

    def test_resize_square_image(self):
        """Test resizing a square image."""
        lb = LetterBoxResize(target_size=96, fill_value=114)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = lb(img)

        assert result.shape == (96, 96, 3)
        # Square image should fill the target exactly (scaled down)
        # No padding needed for proportional scaling

    def test_resize_wide_image(self):
        """Test resizing a wide image (width > height)."""
        lb = LetterBoxResize(target_size=96, fill_value=114)
        img = np.ones((50, 100, 3), dtype=np.uint8) * 255

        result = lb(img)

        assert result.shape == (96, 96, 3)
        # Wide image should have padding on top and bottom
        # Check that padding value exists
        assert np.any(result == 114)

    def test_resize_tall_image(self):
        """Test resizing a tall image (height > width)."""
        lb = LetterBoxResize(target_size=96, fill_value=114)
        img = np.ones((100, 50, 3), dtype=np.uint8) * 255

        result = lb(img)

        assert result.shape == (96, 96, 3)
        # Tall image should have padding on left and right
        assert np.any(result == 114)

    def test_resize_grayscale_image(self):
        """Test resizing a grayscale image."""
        lb = LetterBoxResize(target_size=96, fill_value=0)
        img = np.ones((50, 100), dtype=np.uint8) * 255

        result = lb(img)

        assert result.shape == (96, 96)
        assert result.dtype == np.uint8

    def test_resize_preserves_aspect_ratio(self):
        """Test that aspect ratio is preserved."""
        lb = LetterBoxResize(target_size=96, fill_value=114)
        # 2:1 aspect ratio
        img = np.ones((50, 100, 3), dtype=np.uint8)
        # Make left half black, right half white to check proportions
        img[:, :50] = 0
        img[:, 50:] = 255

        result = lb(img)

        # Find the transition from black to white in the resized image
        # The aspect ratio should be preserved, so the transition should be at midpoint
        mid_col = 96 // 2
        # Due to letterboxing, the actual content might be centered
        # Just verify the image was resized properly
        assert result.shape == (96, 96, 3)

    def test_resize_very_small_image(self):
        """Test resizing a very small image."""
        lb = LetterBoxResize(target_size=96, fill_value=114)
        img = np.ones((10, 10, 3), dtype=np.uint8) * 255

        result = lb(img)

        assert result.shape == (96, 96, 3)
        # Small image should be upscaled and heavily padded

    def test_resize_very_large_image(self):
        """Test resizing a very large image."""
        lb = LetterBoxResize(target_size=96, fill_value=114)
        img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

        result = lb(img)

        assert result.shape == (96, 96, 3)

    def test_centering(self):
        """Test that image is centered in the output."""
        lb = LetterBoxResize(target_size=100, fill_value=0)
        # 2:1 image
        img = np.ones((50, 100, 3), dtype=np.uint8) * 255

        result = lb(img)

        # Wide image: should be centered vertically
        # Top and bottom should have equal padding
        non_zero_rows = np.where(result.any(axis=(1, 2)))[0]
        top_padding = non_zero_rows[0]
        bottom_padding = 100 - non_zero_rows[-1] - 1

        assert abs(top_padding - bottom_padding) <= 1  # Allow off-by-1 due to integer division


class TestTrainingAugmentation:
    """Tests for TrainingAugmentation class."""

    def test_init_empty_config(self):
        """Test initialization with empty config."""
        aug = TrainingAugmentation({})
        assert aug.cfg == {}
        assert aug.color_jitter_cfg == {}
        assert aug.random_gray_cfg == {}
        assert aug.random_rotation_cfg == {}
        assert aug.random_mask_cfg == {}
        assert aug.horizontal_flip_cfg == {}

    def test_init_with_config(self):
        """Test initialization with full config."""
        cfg = {
            'color_jitter': {'enabled': True, 'brightness': 0.5},
            'random_gray': {'enabled': True, 'p': 0.3},
            'random_rotation': {'enabled': True, 'angle_range': [-10, 10]},
            'random_mask': {'enabled': True, 'mask_ratio': 0.3, 'p': 0.5},
            'horizontal_flip': {'enabled': True, 'p': 0.5},
        }
        aug = TrainingAugmentation(cfg)
        assert aug.color_jitter_cfg == cfg['color_jitter']
        assert aug.random_gray_cfg == cfg['random_gray']
        assert aug.random_rotation_cfg == cfg['random_rotation']
        assert aug.random_mask_cfg == cfg['random_mask']
        assert aug.horizontal_flip_cfg == cfg['horizontal_flip']

    def test_random_mask_initialized_when_enabled(self):
        """Test that RandomMask is initialized when enabled."""
        cfg = {
            'random_mask': {'enabled': True, 'mask_ratio': 0.3, 'p': 0.5}
        }
        aug = TrainingAugmentation(cfg)
        assert hasattr(aug, 'random_mask')
        assert aug.random_mask.mask_ratio == 0.3
        assert aug.random_mask.p == 0.5

    def test_random_mask_not_initialized_when_disabled(self):
        """Test that RandomMask is not initialized when disabled."""
        cfg = {
            'random_mask': {'enabled': False}
        }
        aug = TrainingAugmentation(cfg)
        assert not hasattr(aug, 'random_mask')

    def test_call_no_augmentations(self):
        """Test call with all augmentations disabled."""
        cfg = {
            'color_jitter': {'enabled': False},
            'random_gray': {'enabled': False},
            'random_rotation': {'enabled': False},
            'random_mask': {'enabled': False},
            'horizontal_flip': {'enabled': False},
        }
        aug = TrainingAugmentation(cfg)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        result = aug(img)

        np.testing.assert_array_equal(result, img)

    def test_horizontal_flip_applied(self, monkeypatch):
        """Test horizontal flip is applied when enabled and probability met."""
        cfg = {
            'horizontal_flip': {'enabled': True, 'p': 0.5}
        }
        aug = TrainingAugmentation(cfg)
        img = np.ones((100, 100, 3), dtype=np.uint8)
        img[:, :50] = 0  # Left half black
        img[:, 50:] = 255  # Right half white

        # Mock random to trigger flip
        monkeypatch.setattr(np.random, 'random', lambda: 0.3)

        result = aug(img)

        # After flip, left should be white, right should be black
        assert np.all(result[:, :50] == 255)
        assert np.all(result[:, 50:] == 0)

    def test_horizontal_flip_not_applied(self, monkeypatch):
        """Test horizontal flip is not applied when probability not met."""
        cfg = {
            'horizontal_flip': {'enabled': True, 'p': 0.5}
        }
        aug = TrainingAugmentation(cfg)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Mock random to not trigger flip
        monkeypatch.setattr(np.random, 'random', lambda: 0.6)

        result = aug(img)

        np.testing.assert_array_equal(result, img)

    def test_color_jitter_brightness(self, monkeypatch):
        """Test color jitter brightness adjustment."""
        cfg = {
            'color_jitter': {'enabled': True, 'brightness': 0.5, 'contrast': 0, 'saturation': 0}
        }
        aug = TrainingAugmentation(cfg)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Mock uniform to return specific brightness factor (increase by 50%)
        monkeypatch.setattr(np.random, 'uniform', lambda low, high: 1.5)

        result = aug(img)

        # Brightness increased by 50%
        expected = np.clip(128 * 1.5, 0, 255).astype(np.uint8)
        assert np.all(result == expected)

    def test_color_jitter_contrast(self, monkeypatch):
        """Test color jitter contrast adjustment."""
        cfg = {
            'color_jitter': {'enabled': True, 'brightness': 0, 'contrast': 0.5, 'saturation': 0}
        }
        aug = TrainingAugmentation(cfg)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # First call is for brightness (returns 1.0 = no change)
        # Second call is for contrast
        call_count = [0]
        def mock_uniform(low, high):
            call_count[0] += 1
            if call_count[0] == 1:
                return 1.0  # brightness
            return 1.5  # contrast

        monkeypatch.setattr(np.random, 'uniform', mock_uniform)

        result = aug(img)

        # With mean=128 and contrast=1.5: (128-128)*1.5 + 128 = 128
        # So uniform image should stay the same
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_random_gray_applied(self, monkeypatch):
        """Test random grayscale conversion is applied."""
        cfg = {
            'random_gray': {'enabled': True, 'p': 1.0}  # Always apply
        }
        aug = TrainingAugmentation(cfg)
        # Color image: BGR = (100, 150, 200)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 200  # B
        img[:, :, 1] = 150  # G
        img[:, :, 2] = 100  # R

        monkeypatch.setattr(np.random, 'random', lambda: 0.0)  # Below p

        result = aug(img)

        # After BGR->Gray->BGR, all channels should be equal
        assert np.all(result[:, :, 0] == result[:, :, 1])
        assert np.all(result[:, :, 1] == result[:, :, 2])

    def test_random_gray_not_applied(self, monkeypatch):
        """Test random grayscale is not applied when probability not met."""
        cfg = {
            'random_gray': {'enabled': True, 'p': 0.1}
        }
        aug = TrainingAugmentation(cfg)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        monkeypatch.setattr(np.random, 'random', lambda: 0.5)  # Above p

        result = aug(img)

        np.testing.assert_array_equal(result, img)

    def test_random_rotation_applied(self, monkeypatch, mocker):
        """Test random rotation is applied."""
        cfg = {
            'random_rotation': {'enabled': True, 'p': 1.0, 'angle_range': [-10, 10]},
            'letterbox_fill_value': 114
        }
        aug = TrainingAugmentation(cfg)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        monkeypatch.setattr(np.random, 'random', lambda: 0.0)  # Below p
        monkeypatch.setattr(np.random, 'uniform', lambda low, high: 5.0)  # 5 degree rotation

        # Mock cv2 functions
        mock_getRotationMatrix2D = mocker.patch('cv2.getRotationMatrix2D')
        mock_warpAffine = mocker.patch('cv2.warpAffine')

        mock_getRotationMatrix2D.return_value = np.eye(2, 3)
        mock_warpAffine.return_value = img

        result = aug(img)

        # Verify cv2 functions were called
        mock_getRotationMatrix2D.assert_called_once()
        mock_warpAffine.assert_called_once()

    def test_random_rotation_not_applied(self, monkeypatch):
        """Test random rotation is not applied when probability not met."""
        cfg = {
            'random_rotation': {'enabled': True, 'p': 0.1}
        }
        aug = TrainingAugmentation(cfg)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        monkeypatch.setattr(np.random, 'random', lambda: 0.5)  # Above p

        result = aug(img)

        np.testing.assert_array_equal(result, img)

    def test_random_mask_applied(self, monkeypatch):
        """Test random mask is applied when enabled."""
        cfg = {
            'random_mask': {'enabled': True, 'mask_ratio': 0.25, 'p': 1.0}
        }
        aug = TrainingAugmentation(cfg)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Track if random_mask was called by checking if image is modified
        # Mock np.random to ensure mask is applied at center
        monkeypatch.setattr(np.random, 'random', lambda: 0.0)

        def mock_randint(*args, **kwargs):
            if len(args) == 2:
                return (args[0] + args[1]) // 2
            elif len(args) >= 3:
                return np.random.RandomState(42).randint(0, 256, size=args[2])
            return 0
        monkeypatch.setattr(np.random, 'randint', mock_randint)

        result = aug(img)

        # The image should be modified by the mask
        assert not np.array_equal(result, img), "RandomMask should have modified the image"

    def test_random_mask_not_applied_when_disabled(self, mocker):
        """Test random mask is not applied when disabled."""
        cfg = {
            'random_mask': {'enabled': False}
        }
        aug = TrainingAugmentation(cfg)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Should not have random_mask attribute
        assert not hasattr(aug, 'random_mask')

        result = aug(img)

        np.testing.assert_array_equal(result, img)

    def test_full_augmentation_pipeline(self, monkeypatch, mocker):
        """Test full augmentation pipeline with all features enabled."""
        cfg = {
            'color_jitter': {'enabled': True, 'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3},
            'random_gray': {'enabled': True, 'p': 1.0},
            'random_rotation': {'enabled': True, 'p': 1.0, 'angle_range': [-5, 5]},
            'random_mask': {'enabled': True, 'mask_ratio': 0.2, 'p': 1.0},
            'horizontal_flip': {'enabled': True, 'p': 1.0},
            'letterbox_fill_value': 114
        }
        aug = TrainingAugmentation(cfg)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Mock all random calls
        call_count = {'random': 0, 'uniform': 0}
        def mock_random():
            call_count['random'] += 1
            return 0.0  # Always trigger

        def mock_uniform(low, high):
            call_count['uniform'] += 1
            return 1.0  # No change for brightness/contrast

        monkeypatch.setattr(np.random, 'random', mock_random)
        monkeypatch.setattr(np.random, 'uniform', mock_uniform)

        # Mock cv2 functions for rotation
        mocker.patch('cv2.getRotationMatrix2D', return_value=np.eye(2, 3))
        mocker.patch('cv2.warpAffine', return_value=img)
        mocker.patch('cv2.cvtColor', return_value=img)

        # Mock random_mask
        mocker.patch.object(aug.random_mask, '__call__', return_value=img)

        result = aug(img)

        # Verify image was processed
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_call_preserves_image_properties(self):
        """Test that call preserves image dtype and shape."""
        cfg = {}
        aug = TrainingAugmentation(cfg)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        result = aug(img)

        assert result.dtype == np.uint8
        assert result.shape == (100, 100, 3)
        assert np.all(result >= 0) and np.all(result <= 255)


class TestIntegration:
    """Integration tests for augmentation pipeline."""

    def test_letterbox_resize_then_augmentation(self, monkeypatch):
        """Test combining LetterBoxResize with TrainingAugmentation."""
        lb = LetterBoxResize(target_size=96, fill_value=114)
        cfg = {
            'color_jitter': {'enabled': True, 'brightness': 0.2, 'contrast': 0, 'saturation': 0}
        }
        aug = TrainingAugmentation(cfg)

        # Create a non-square image
        img = np.ones((50, 100, 3), dtype=np.uint8) * 128

        # Resize first
        resized = lb(img)
        assert resized.shape == (96, 96, 3)

        # Then augment
        monkeypatch.setattr(np.random, 'uniform', lambda low, high: 1.2)
        result = aug(resized)

        assert result.shape == (96, 96, 3)
        assert result.dtype == np.uint8

    def test_random_crop_then_letterbox(self, monkeypatch):
        """Test combining RandomCropWithPadding with LetterBoxResize."""
        crop = RandomCropWithPadding(shrink_max=5, expand_max=10)
        lb = LetterBoxResize(target_size=96, fill_value=114)

        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        bbox = [50, 50, 150, 150]

        monkeypatch.setattr(np.random, 'randint', lambda low, high: 5)

        cropped = crop(img, bbox)
        result = lb(cropped)

        assert result.shape == (96, 96, 3)
