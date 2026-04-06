"""Train person detector with YOLOv8 (config-based).

Refactored to use config file like train_simple_classifier.
"""

import argparse
import ast
import os
import sys
from typing import Any, Dict

import yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train person detector with YOLOv8"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Override config values, e.g., 'epochs=100,batch=8'",
    )
    return parser.parse_args()


def load_config(args) -> Dict[str, Any]:
    """Load and parse configuration."""
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Process command line overrides
    if args.override:
        for override in args.override.split(","):
            key, value = override.split("=")
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            d[keys[-1]] = value

    return cfg


def main():
    """Main entry point."""
    args = parse_args()
    cfg = load_config(args)

    # Import ultralytics after config loading for faster error handling
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics>=8.0")
        sys.exit(1)

    # Get configuration values
    data_yaml = cfg.get("data", "data/fall_detection.yaml")
    epochs = cfg.get("epochs", 50)
    imgsz = cfg.get("imgsz", 1280)
    batch = cfg.get("batch", 16)
    model_name = cfg.get("model", "yolov8n.pt")
    project = cfg.get("project", "yolov8")
    name = cfg.get("name", "exp")
    workers = cfg.get("workers", 8)

    # Optimizer settings
    lr0 = cfg.get("lr0", 0.01)
    lrf = cfg.get("lrf", 0.01)
    momentum = cfg.get("momentum", 0.937)
    weight_decay = cfg.get("weight_decay", 0.0005)

    # Augmentation settings
    hsv_h = cfg.get("hsv_h", 0.015)
    hsv_s = cfg.get("hsv_s", 0.7)
    hsv_v = cfg.get("hsv_v", 0.4)
    degrees = cfg.get("degrees", 0.0)
    translate = cfg.get("translate", 0.1)
    scale = cfg.get("scale", 0.5)
    shear = cfg.get("shear", 0.0)
    perspective = cfg.get("perspective", 0.0)
    flipud = cfg.get("flipud", 0.0)
    fliplr = cfg.get("fliplr", 0.5)
    mosaic = cfg.get("mosaic", 1.0)
    mixup = cfg.get("mixup", 0.0)
    copy_paste = cfg.get("copy_paste", 0.0)

    # Training control
    patience = cfg.get("patience", 10)
    save = cfg.get("save", True)
    save_period = cfg.get("save_period", 10)
    exist_ok = cfg.get("exist_ok", False)
    verbose = cfg.get("verbose", True)
    device = cfg.get("device", None)

    # Resolve device
    if device is None:
        device = 0 if os.path.exists("/proc/driver/nvidia/version") else "cpu"

    print(f"Starting YOLOv8 detector training...")
    print(f"  Config: {args.config}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch: {batch}")
    print(f"  Model: {model_name}")
    print(f"  Output: runs/detect/{project}/{name}/")

    # Load model
    model = YOLO(model_name)

    # Train
    try:
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            workers=workers,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            mosaic=mosaic,
            mixup=mixup,
            copy_paste=copy_paste,
            patience=patience,
            save=save,
            save_period=save_period,
            exist_ok=exist_ok,
            verbose=verbose,
            device=device,
        )
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create symlink/copy for best weights
    best_path = os.path.join("runs/detect", project, name, "weights", "best.pt")
    out_path = os.path.join("runs/detect", project, name, "best.pt")
    if os.path.exists(best_path):
        try:
            os.link(best_path, out_path)
            print(f"Best weights linked to {out_path}")
        except OSError:
            import shutil
            shutil.copy2(best_path, out_path)
            print(f"Best weights copied to {out_path}")

    print(f"\nTraining complete. Results saved to: runs/detect/{project}/{name}/")


if __name__ == "__main__":
    main()
