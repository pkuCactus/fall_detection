import torch

from fall_detection.classifier import FallClassifier


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
        dynamic_axes={
            "roi": {0: "batch_size"},
            "kpts": {0: "batch_size"},
            "motion": {0: "batch_size"},
            "prob": {0: "batch_size"},
        },
        opset_version=11,
    )
    print(f"ONNX exported to {out_path}")
