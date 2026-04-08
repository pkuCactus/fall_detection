"""Train person detector with YOLOv8 (config-based).

Refactored to use config file like train_simple_classifier.
Supports DDP, random seed, modular config sections, and warmup.
"""

import argparse
import ast
import os
import random
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train person detector with YOLOv8"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--local-rank", type=int, default=-1)
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


def setup_ddp(cfg: Dict[str, Any]) -> Tuple[bool, int, int]:
    """Setup Distributed Data Parallel.

    Returns:
        (ddp_enabled, world_size, rank)
    """
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if ddp:
        ddp_cfg = cfg.get("ddp", {})
        backend = ddp_cfg.get("backend", "nccl")

        dist.init_process_group(backend=backend if torch.cuda.is_available() else "gloo")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    return ddp, world_size, rank


def setup_seed(seed: Optional[int], rank: int) -> None:
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


def main():
    """Main entry point."""
    args = parse_args()
    cfg = load_config(args)

    # Setup DDP
    ddp, world_size, rank = setup_ddp(cfg)

    # Setup seed
    setup_seed(cfg.get("seed"), rank)

    # Import ultralytics after config loading for faster error handling
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics>=8.0")
        sys.exit(1)

    # Get configuration values with modular sections
    # Data config
    data_cfg = cfg.get("data", {})
    data_yaml = data_cfg.get("yaml", "data/yaml/fall_detection.yaml") if isinstance(data_cfg, dict) else data_cfg

    # Model config
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "yolov8n.pt") if isinstance(model_cfg, dict) else cfg.get("model", "yolov8n.pt")
    pretrained = model_cfg.get("pretrained", None) if isinstance(model_cfg, dict) else None

    # Training config
    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", cfg.get("epochs", 50))
    imgsz = train_cfg.get("imgsz", cfg.get("imgsz", 1280))
    batch = train_cfg.get("batch", cfg.get("batch", 16))
    workers = train_cfg.get("workers", cfg.get("workers", 8))

    # Optimizer config
    optim_cfg = cfg.get("optimizer", {})
    lr0 = optim_cfg.get("lr0", cfg.get("lr0", 0.01))
    lrf = optim_cfg.get("lrf", cfg.get("lrf", 0.01))
    momentum = optim_cfg.get("momentum", cfg.get("momentum", 0.937))
    weight_decay = optim_cfg.get("weight_decay", cfg.get("weight_decay", 0.0005))

    # Warmup config
    warmup_cfg = cfg.get("warmup", {})
    warmup_epochs = warmup_cfg.get("epochs", 3.0)
    warmup_momentum = warmup_cfg.get("momentum", 0.8)
    warmup_bias_lr = warmup_cfg.get("bias_lr", 0.1)

    # Augmentation config
    aug_cfg = cfg.get("augmentation", {})
    hsv_h = aug_cfg.get("hsv_h", cfg.get("hsv_h", 0.015))
    hsv_s = aug_cfg.get("hsv_s", cfg.get("hsv_s", 0.7))
    hsv_v = aug_cfg.get("hsv_v", cfg.get("hsv_v", 0.4))
    degrees = aug_cfg.get("degrees", cfg.get("degrees", 0.0))
    translate = aug_cfg.get("translate", cfg.get("translate", 0.1))
    scale = aug_cfg.get("scale", cfg.get("scale", 0.5))
    shear = aug_cfg.get("shear", cfg.get("shear", 0.0))
    perspective = aug_cfg.get("perspective", cfg.get("perspective", 0.0))
    flipud = aug_cfg.get("flipud", cfg.get("flipud", 0.0))
    fliplr = aug_cfg.get("fliplr", cfg.get("fliplr", 0.5))
    mosaic = aug_cfg.get("mosaic", cfg.get("mosaic", 1.0))
    mixup = aug_cfg.get("mixup", cfg.get("mixup", 0.0))
    copy_paste = aug_cfg.get("copy_paste", cfg.get("copy_paste", 0.0))

    # Output config
    out_cfg = cfg.get("output", {})
    project = out_cfg.get("project", cfg.get("project", "yolov8"))
    name = out_cfg.get("name", cfg.get("name", "exp"))
    exist_ok = out_cfg.get("exist_ok", cfg.get("exist_ok", False))

    # Training control
    patience = cfg.get("patience", 10)
    save = cfg.get("save", True)
    save_period = cfg.get("save_period", 10)
    verbose = cfg.get("verbose", True)

    # Device
    device_cfg = cfg.get("device", None)
    if device_cfg is None:
        device = 0 if os.path.exists("/proc/driver/nvidia/version") else "cpu"
    else:
        device = device_cfg

    if rank == 0:
        print(f"Starting YOLOv8 detector training...")
        print(f"  Config: {args.config}")
        print(f"  Data: {data_yaml}")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {imgsz}")
        print(f"  Batch: {batch}")
        print(f"  Model: {model_name}")
        print(f"  Output: runs/detect/{project}/{name}/")
        print(f"  DDP: {ddp}, World size: {world_size}")

    # Load model
    if pretrained and os.path.exists(pretrained):
        model = YOLO(pretrained)
    else:
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
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
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
