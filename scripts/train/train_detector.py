"""Train person detector with YOLOv8 (config-based).

Refactored to use config file like train_simple_classifier.
Supports DDP, random seed, modular config sections, and warmup.
"""

import sys
import traceback

import torch
from ultralytics import YOLO

from fall_detection.utils import parse_args, load_config


def main():
    """Main entry point."""
    args = parse_args("Train person detector with YOLO")
    cfg = load_config(args)

    # Load model
    model = YOLO(cfg.get("model", "data/models/pretrained/yolov8n.pt"))  # 支持直接传路径或使用默认预训练权重

    # Train
    try:
        model.train(**cfg)
    except Exception as e:
        print(f"\nError during training: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
