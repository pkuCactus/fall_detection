#!/usr/bin/env python
"""Validate YOLO-World model with support for WxH image sizes."""

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


def main():
    parser = argparse.ArgumentParser(description="Validate YOLO-World model")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    parser.add_argument("--data", default="data/fall_detection.yaml", help="Data config YAML")
    parser.add_argument(
        "--imgsz",
        type=parse_imgsz,
        default=(832, 448),
        help="Image size: int (square) or WxH (e.g., 832x448)",
    )
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="0", help="Device (0, 0,1, cpu)")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Validation split")
    args = parser.parse_args()

    # Load data config to get class names
    with open(args.data, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg.get("names", {})
    if isinstance(names, dict):
        classes = [names[i] for i in sorted(names)]
    elif isinstance(names, list):
        classes = list(names)
    else:
        classes = ["person"]

    # Load model and set classes
    model = YOLOWorld(args.weights)
    model.set_classes(classes)

    # Validate
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        split=args.split,
    )

    # Print results
    print("\n" + "=" * 50)
    print("Validation Results")
    print("=" * 50)
    print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
    print("=" * 50)


if __name__ == "__main__":
    main()
