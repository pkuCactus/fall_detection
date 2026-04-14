#!/usr/bin/env python3
"""Generate text embeddings for YOLO-World training.

This script uses CLIP to generate text embeddings for class names,
which are required for YOLO-World open-vocabulary training.

Usage:
    python generate_yoloworld_embeddings.py data/configs/fall_detection_yoloworld.yaml
    python generate_yoloworld_embeddings.py data/configs/fall_detection_yoloworld.yaml --force
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text embeddings for YOLO-World",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings for a data config
  python generate_yoloworld_embeddings.py data/configs/fall_detection_yoloworld.yaml

  # Force regenerate embeddings (overwrite existing)
  python generate_yoloworld_embeddings.py data/configs/fall_detection_yoloworld.yaml --force

  # Specify custom CLIP model
  python generate_yoloworld_embeddings.py data/configs/fall_detection_yoloworld.yaml --clip-model ViT-B/16
        """,
    )
    parser.add_argument(
        "data_yaml",
        type=str,
        help="Path to YOLO data config YAML file",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force regenerate embeddings (overwrite existing)",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B/32",
        choices=["RN50", "RN101", "ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="CLIP model variant to use (default: ViT-B/32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda (default: auto)",
    )
    return parser.parse_args()


def load_data_config(yaml_path: str) -> dict:
    """Load YOLO data config."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def get_class_names(data_cfg: dict) -> list:
    """Extract class names from data config."""
    names = data_cfg.get('names', {})

    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    elif isinstance(names, list):
        return names
    else:
        return []


def generate_text_embeddings(class_names: list, clip_model_name: str, device: str) -> torch.Tensor:
    """Generate text embeddings using CLIP.

    Args:
        class_names: List of class names
        clip_model_name: CLIP model variant (e.g., 'ViT-B/32')
        device: Device to use

    Returns:
        Text embeddings tensor
    """
    if not CLIP_AVAILABLE:
        raise RuntimeError("CLIP is not installed. Install with: pip install git+https://github.com/openai/CLIP.git")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading CLIP model: {clip_model_name}")
    model, preprocess = clip.load(clip_model_name, device=device)

    # Format text prompts for YOLO-World style
    # YOLO-World typically uses "a photo of a {class}" template
    text_prompts = [f"a photo of a {name}" for name in class_names]

    print(f"Generating embeddings for {len(class_names)} classes:")
    for i, name in enumerate(class_names, 1):
        print(f"  {i}. {name}")

    # Tokenize and encode
    tokens = clip.tokenize(text_prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        # Normalize embeddings
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu()


def main():
    """Main entry point."""
    args = parse_args()

    if not CLIP_AVAILABLE:
        print("Error: CLIP is required but not installed.")
        print("Install with: pip install git+https://github.com/openai/CLIP.git")
        sys.exit(1)

    # Check data config exists
    if not os.path.exists(args.data_yaml):
        print(f"Error: Data config not found: {args.data_yaml}")
        sys.exit(1)

    # Load config
    data_cfg = load_data_config(args.data_yaml)
    dataset_path = data_cfg.get('path', '')

    if not dataset_path:
        print("Error: 'path' not specified in data config")
        sys.exit(1)

    # Get class names
    class_names = get_class_names(data_cfg)
    if not class_names:
        print("Error: No class names found in data config")
        sys.exit(1)

    print(f"Dataset path: {dataset_path}")
    print(f"Number of classes: {len(class_names)}")

    # Determine output path
    # YOLO-World expects embeddings in images/text_embeddings_clip_ViT-B_32.pt
    # Format: {class_name: embedding_tensor} dict
    clip_model_safe = args.clip_model.replace('/', '_')
    output_dir = os.path.join(dataset_path, 'images')
    output_path = os.path.join(output_dir, f'text_embeddings_clip_{clip_model_safe}.pt')

    # Check if already exists
    if os.path.exists(output_path) and not args.force:
        print(f"\nEmbeddings already exist: {output_path}")
        print("Use --force to regenerate")
        return

    # Create output directory (dataset root)
    os.makedirs(output_dir, exist_ok=True)

    # Generate embeddings
    print(f"\nGenerating embeddings with CLIP {args.clip_model}...")
    try:
        embeddings = generate_text_embeddings(
            class_names,
            args.clip_model,
            args.device
        )
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        sys.exit(1)

    # Save embeddings as {class_name: embedding_tensor} dict (ultralytics format)
    txt_map = {name: embeddings[i] for i, name in enumerate(class_names)}

    torch.save(txt_map, output_path)
    print(f"\nSaved embeddings to: {output_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Classes: {len(class_names)}")
    print(f"  CLIP model: {args.clip_model}")


if __name__ == "__main__":
    main()
