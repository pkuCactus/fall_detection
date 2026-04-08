#!/usr/bin/env python
"""Export YOLO-World model with support for WxH image sizes."""

import argparse
import os

import yaml
from ultralytics import YOLOWorld


def parse_imgsz(value):
    """Parse image size: int or WxH string."""
    try:
        return int(value)
    except ValueError:
        w, h = value.lower().split("x")
        return (int(w), int(h))


def str_to_bool(value):
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "y")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO-World model")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    parser.add_argument(
        "--imgsz",
        type=parse_imgsz,
        default=(832, 448),
        help="Image size: int (square) or WxH (e.g., 832x448)",
    )
    parser.add_argument("--format", default="onnx", help="Export format (onnx, engine, tflite, etc.)")
    parser.add_argument("--device", default="0", help="Device (0, cpu)")
    parser.add_argument("--half", type=str_to_bool, default=False, help="FP16 half precision")
    parser.add_argument("--int8", type=str_to_bool, default=False, help="INT8 quantization")
    parser.add_argument("--dynamic", type=str_to_bool, default=False, help="Dynamic batch size")
    args = parser.parse_args()

    # Load model
    model = YOLOWorld(args.weights)

    # Export
    print(f"Exporting to {args.format.upper()} format...")
    print(f"Input size: {args.imgsz}")

    export_path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
    )

    print("\n" + "=" * 50)
    print("Export Complete")
    print("=" * 50)
    print(f"Output: {export_path}")
    print(f"Format: {args.format}")
    print(f"Input size: {args.imgsz}")
    print("=" * 50)


if __name__ == "__main__":
    main()
