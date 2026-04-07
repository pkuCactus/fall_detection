"""Train person detector with YOLOWorld (config-based).

Refactored to use config file like train_simple_classifier.
"""

import argparse
import ast
import os
import random
import sys
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml


def parse_imgsz(value):
    """Parse image size: int or WxH string."""
    try:
        return int(value)
    except ValueError:
        w, h = value.lower().split("x")
        return (int(w), int(h))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train person detector with YOLOWorld"
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


def setup_ddp() -> Tuple[bool, int, int]:
    """Detect DDP environment set by torchrun.

    Returns:
        (ddp_enabled, world_size, rank)
    """
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return ddp, world_size, rank


def setup_seed(seed, rank: int) -> None:
    """Set random seed for reproducibility."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if rank == 0:
        print(f"Random seed set to {seed}")


def check_and_download_model(model_name, rank: int = 0):
    """Check if model file exists, auto-download if needed."""
    if os.path.exists(model_name):
        return model_name

    if rank == 0:
        print(f"Model '{model_name}' not found locally.")
        print(f"Ultralytics will attempt to download it automatically...")
    return model_name


def main():
    """Main entry point."""
    args = parse_args()
    cfg = load_config(args)

    # Detect DDP environment (set by torchrun)
    ddp, world_size, rank = setup_ddp()

    # Setup seed
    setup_seed(cfg.get("seed"), rank)

    # Import ultralytics after config loading for faster error handling
    try:
        from ultralytics import YOLOWorld
    except ImportError:
        if rank == 0:
            print("Error: ultralytics not installed. Run: pip install ultralytics>=8.3.0")
        sys.exit(1)

    # Get configuration values
    data_yaml = cfg.get("data", "data/fall_detection_yolo_world.yaml")
    epochs = cfg.get("epochs", 50)
    imgsz = cfg.get("imgsz", 1280)
    batch = cfg.get("batch", 16)
    model_name = cfg.get("model", "yolov8l-worldv2.pt")
    project = cfg.get("project", "yolo_world")
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

    # Device: DDP mode uses auto-selection; single GPU mode respects config
    device_cfg = cfg.get("device", None)
    if ddp:
        device = None  # Force auto-select in DDP mode (ultralytics handles GPU assignment)
    elif device_cfg is None:
        device = 0 if os.path.exists("/proc/driver/nvidia/version") else "cpu"
    else:
        device = device_cfg

    if rank == 0:
        print(f"Starting YOLOWorld training...")
        print(f"  Config: {args.config}")
        print(f"  Data: {data_yaml}")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {imgsz}")
        print(f"  Batch: {batch}")
        print(f"  Model: {model_name}")
        print(f"  Output: runs/detect/{project}/{name}/")
        print(f"  DDP: {ddp}, World size: {world_size}")

    # Load data config to get class names
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
        names = data_cfg.get("names", {})
        if isinstance(names, dict):
            classes = [names[i] for i in sorted(names)]
        elif isinstance(names, list):
            classes = list(names)
        else:
            classes = ["person"]
    except FileNotFoundError:
        if rank == 0:
            print(f"Error: Data config file not found: {data_yaml}")
        sys.exit(1)

    # Check and load model
    model_path = check_and_download_model(model_name, rank=rank)

    try:
        model = YOLOWorld(model_path)
    except Exception as e:
        if rank == 0:
            print(f"\nError loading model '{model_name}': {e}")
            print("\nTroubleshooting tips:")
            print("1. Ensure ultralytics>=8.3.0: pip install -U ultralytics")
            print("2. For v2.1 models, you may need to download manually:")
            print(f"   wget https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}")
            print("3. Or use v2 models: yolov8l-worldv2.pt")
        sys.exit(1)

    # Set class names
    model.set_classes(classes)
    if rank == 0:
        print(f"  Classes: {classes}")

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
        if rank == 0:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Create symlink/copy for best weights (rank 0 only)
    if rank == 0:
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
