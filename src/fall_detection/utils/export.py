import torch

from fall_detection.models.classifier import FallClassifier
from fall_detection.models.simple_classifier import SimpleFallClassifier


def export_classifier_onnx(out_path: str = "fall_classifier.onnx"):
    """导出 FallClassifier 为 ONNX 格式."""
    model = FallClassifier()
    model.eval()

    roi = torch.zeros(1, 3, 96, 96, dtype=torch.float32)
    kpts = torch.zeros(1, 17, 3, dtype=torch.float32)
    motion = torch.zeros(1, 8, dtype=torch.float32)

    torch.onnx.export(
        model,
        (roi, kpts, motion),
        out_path,
        input_names=["roi", "kpts", "motion"],
        output_names=["prob"],
        opset_version=11,
    )
    print(f"ONNX exported to {out_path}")


def export_simple_classifier_onnx(out_path: str = "simple_fall_classifier.onnx"):
    """导出 SimpleFallClassifier 为 ONNX 格式."""
    model = SimpleFallClassifier()
    model.eval()

    x = torch.zeros(1, 3, 96, 96, dtype=torch.float32)

    torch.onnx.export(
        model,
        x,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=11,
    )
    print(f"ONNX exported to {out_path}")
