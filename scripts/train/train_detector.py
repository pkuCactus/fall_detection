"""Train person detector with YOLOv8 or YOLO-World (config-based).

Refactored to use config file like train_simple_classifier.
Supports DDP, random seed, modular config sections, and warmup.
Supports both standard YOLO and YOLO-World (open-vocabulary) training.
"""

import os
import sys
import traceback

import yaml
from ultralytics import YOLO, YOLOWorld

from fall_detection.utils import parse_args, load_config


def is_ddp() -> bool:
    """Check if running in DDP mode."""
    return int(os.environ.get("RANK", -1)) != -1 or int(os.environ.get("LOCAL_RANK", -1)) != -1


def load_data_config(data_path: str) -> dict:
    """Load YOLO data config file."""
    if not os.path.exists(data_path):
        return {}
    with open(data_path, 'r') as f:
        return yaml.safe_load(f)


def is_yolo_world(model_path: str) -> bool:
    """Check if model is YOLO-World based on path/name."""
    if not model_path:
        return False
    model_name = os.path.basename(model_path).lower()
    return 'world' in model_name


def setup_yolo_world(model: YOLO, data_cfg: dict, cfg: dict):
    """Setup YOLO-World model with classes for open-vocabulary training.

    Args:
        model: YOLO model instance
        data_cfg: Data configuration dict with 'names' field
        cfg: Training configuration
    """
    # Get class names from config or data config
    names = data_cfg.get('names', {})

    # Handle different formats: list or dict
    if isinstance(names, dict):
        # Dict format: {0: 'person', 1: 'car'}
        class_names = [names[i] for i in sorted(names.keys())]
    elif isinstance(names, list):
        # List format: ['person', 'car']
        class_names = names
    else:
        class_names = []

    if class_names:
        print(f"Setting YOLO-World classes: {class_names}")
        model.set_classes(class_names)
    else:
        print("Warning: No class names found for YOLO-World, using default")


def main():
    """Main entry point."""
    args = parse_args("Train person detector with YOLO or YOLO-World")
    cfg = load_config(args)

    # Determine model type
    model_path = cfg.get("model", "data/models/pretrained/yolov8n.pt")
    model_type = cfg.get("model_type", "auto").lower()

    # Auto-detect YOLO-World
    if model_type == "auto":
        model_type = "yoloworld" if is_yolo_world(model_path) else "yolo"

    print(f"Model type: {model_type}")
    print(f"Model path: {model_path}")

    # Load model
    try:
        model = YOLO(model_path) if model_type == "yolo" else YOLOWorld(model_path)
    except Exception as e:
        print(f"\nError loading model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Setup YOLO-World specific settings
    if model_type == "yoloworld":
        # Load data config to get class names and path
        data_path = cfg.get("data", "")
        if data_path and os.path.exists(data_path):
            data_cfg = load_data_config(data_path)

            # Delete cached text embeddings before training
            # YOLO and YOLO-World have different embeddings, need to regenerate
            dataset_path = data_cfg.get("path", "")
            if dataset_path:
                embed_cache = os.path.join(dataset_path, "images", "text_embeddings_clip_ViT-B_32.pt")
                if os.path.exists(embed_cache):
                    print(f"Removing cached text embeddings: {embed_cache}")
                    os.remove(embed_cache)

            setup_yolo_world(model, data_cfg, cfg)
        else:
            print(f"Warning: Data config not found: {data_path}")
            print("YOLO-World requires class names in data config")

    # Train
    try:
        # For YOLO-World DDP training, use custom trainer to fix validation AP issue
        trainer_cfg = cfg.pop("trainer", None)
        if model_type == "yoloworld" and trainer_cfg != "default":
            from fall_detection.trainers import WorldTrainerDDP
            model.train(trainer=WorldTrainerDDP, **cfg)
        else:
            model.train(**cfg)
    except Exception as e:
        print(f"\nError during training: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
