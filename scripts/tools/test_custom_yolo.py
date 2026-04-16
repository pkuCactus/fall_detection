"""Test script for CustomYOLO with anchor-based detection."""

import sys
import torch

sys.path.insert(0, 'src')

from fall_detection.models.yolo import CustomYOLO, AnchorDetect


def test_anchor_detect():
    """Test AnchorDetect module."""
    print("Testing AnchorDetect module...")

    # Create anchor detect head
    nc = 1  # number of classes
    anchors = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    ch = [192, 96, 160]  # input channels for P5, P4, P3

    head = AnchorDetect(nc=nc, anchors=anchors, ch=ch)

    # Set stride for anchor scaling
    head._stride = [32, 16, 8]

    # Training mode - use fresh inputs
    head.train()
    x_train = [
        torch.randn(1, 192, 20, 20),  # P5
        torch.randn(1, 96, 40, 40),   # P4
        torch.randn(1, 160, 80, 80),  # P3
    ]
    output = head(x_train)
    print(f"  Training output shapes:")
    for i, o in enumerate(output):
        print(f"    P{5-i}: {o.shape}")  # Should be (bs, na, ny, nx, nc+5)

    # Eval mode - use fresh inputs
    head.eval()
    x_eval = [
        torch.randn(1, 192, 20, 20),  # P5
        torch.randn(1, 96, 40, 40),   # P4
        torch.randn(1, 160, 80, 80),  # P3
    ]
    with torch.no_grad():
        pred, features = head(x_eval)
    print(f"  Inference output shape: {pred.shape}")
    expected_anchors = (20*20 + 40*40 + 80*80) * 3
    print(f"  Expected: (1, {expected_anchors}, {nc+5})")

    print("  AnchorDetect test passed!")
    return True


def test_custom_yolo_load():
    """Test loading CustomYOLO with anchor-based config."""
    print("\nTesting CustomYOLO loading...")

    try:
        model = CustomYOLO("configs/model/ori_detector_anchor.yaml")
        print(f"  Model loaded successfully!")
        print(f"  Model type: {type(model.model).__name__}")

        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 640)
        model.model.eval()
        with torch.no_grad():
            output = model.model(dummy_input)
        print(f"  Forward pass successful!")
        print(f"  Output type: {type(output)}")

        return True
    except Exception as e:
        print(f"  Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_yolo_original():
    """Test loading CustomYOLO with original config (no anchor)."""
    print("\nTesting CustomYOLO with original config...")

    try:
        # Note: Original config needs to be updated to use standard Detect
        # or we can test with a standard YOLOv8 model
        model = CustomYOLO("yolov8n.pt")
        print(f"  Model loaded successfully!")
        print(f"  Model type: {type(model.model).__name__}")
        return True
    except Exception as e:
        print(f"  Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("CustomYOLO and AnchorDetect Test Suite")
    print("=" * 60)

    tests = [
        ("AnchorDetect Module", test_anchor_detect),
        ("CustomYOLO (Anchor)", test_custom_yolo_load),
        ("CustomYOLO (Standard)", test_custom_yolo_original),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n  Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
