#!/usr/bin/env python3
"""使用 Cleanlab 查找噪声标签（标签错误的数据）.

根据 simple_classifier_voc.yaml 配置，对 train 和 val 图像进行推理，
获取预测概率和标签，使用 cleanlab 识别可能的标签错误.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "src")
from fall_detection.data import VOCFallDataset
from fall_detection.models import SimpleFallClassifier


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        description="使用 Cleanlab 查找噪声标签",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（使用配置文件中的最佳模型）
  python find_noisy_labels.py --config configs/training/simple_classifier_voc.yaml

  # 指定模型路径
  python find_noisy_labels.py --config configs/training/simple_classifier_voc.yaml \
      --model runs/simple_classifier/best.pt

  # 只分析训练集
  python find_noisy_labels.py --config configs/training/simple_classifier_voc.yaml --split train

  # 保存结果到文件
  python find_noisy_labels.py --config configs/training/simple_classifier_voc.yaml \
      --output noisy_labels.json
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="配置文件路径 (如 configs/training/simple_classifier_voc.yaml)"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="模型权重路径 (默认使用配置文件中的 output.dir/best.pt)"
    )

    parser.add_argument(
        "--split", "-s",
        type=str,
        default="all",
        choices=["train", "val", "all"],
        help="要分析的数据集划分 (默认: all)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出结果文件路径 (JSON 格式)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="推理 batch size (默认: 128)"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="数据加载 worker 数量 (默认: 4)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="显示前 K 个最可能的噪声标签 (默认: 50)"
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """加载配置文件."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_model_path(config: Dict, model_override: Optional[str]) -> str:
    """获取模型路径."""
    if model_override:
        return model_override

    output_dir = config.get("output", {}).get("dir", "outputs/simple_classifier")
    default_path = os.path.join(output_dir, "best.pt")

    if os.path.exists(default_path):
        return default_path

    raise FileNotFoundError(
        f"Model not found: {default_path}\n"
        "Please specify model path with --model or train a model first."
    )


def create_dataset(config: Dict, split: str) -> Optional[VOCFallDataset]:
    """根据配置创建数据集."""
    voc_cfg = config.get("voc", {})
    input_cfg = config.get("input", {})
    crop_cfg = config.get("crop", {})

    # 获取目录列表
    if split == "train":
        data_dirs = voc_cfg.get("train_dirs", [])
    elif split == "val":
        data_dirs = voc_cfg.get("val_dirs", [])
    else:
        return None

    if not data_dirs:
        return None

    fall_classes = voc_cfg.get("fall_classes", ["fall"])
    normal_classes = voc_cfg.get("normal_classes")

    dataset = VOCFallDataset(
        data_dirs=data_dirs,
        split=split,
        transform=None,  # 推理时不做数据增强
        target_size=input_cfg.get("target_size", 96),
        use_letterbox=input_cfg.get("use_letterbox", True),
        fill_value=input_cfg.get("fill_value", 114),
        fall_classes=fall_classes,
        normal_classes=normal_classes,
        shrink_max=crop_cfg.get("shrink_max", 3),
        expand_max=crop_cfg.get("expand_max", 25),
        cache_size=0,  # 禁用缓存
    )

    return dataset


def get_predictions(
    model: SimpleFallClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """获取模型预测概率和标签.

    Args:
        model: 分类器模型
        dataloader: 数据加载器
        device: 计算设备

    Returns:
        (probs, labels, image_paths)
        - probs: (N, num_classes) 预测概率
        - labels: (N,) 真实标签
        - image_paths: (N,) 图像路径列表
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images, labels = batch
            images = images.to(device)

            # 前向传播
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

            # 获取图像路径（从 dataset.samples）
            # 注意：这里我们稍后通过索引映射

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return probs, labels, all_paths


def get_image_paths(dataset: VOCFallDataset) -> List[str]:
    """获取数据集中所有图像的路径."""
    paths = []
    for sample in dataset.samples:
        image_path, _, _ = sample
        paths.append(image_path)
    return paths


def analyze_with_cleanlab(
    probs: np.ndarray,
    labels: np.ndarray,
    image_paths: List[str],
    top_k: int = 50,
) -> List[Dict]:
    """使用 Cleanlab 分析噪声标签.

    Args:
        probs: (N, num_classes) 预测概率
        labels: (N,) 真实标签
        image_paths: (N,) 图像路径
        top_k: 返回前 K 个最可能的噪声标签

    Returns:
        噪声标签列表，每个元素包含:
        - idx: 样本索引
        - image_path: 图像路径
        - given_label: 给定标签
        - predicted_label: 预测标签
        - confidence: 预测置信度
        - label_quality_score: 标签质量分数（越低越可能是噪声）
    """
    try:
        import cleanlab
    except ImportError:
        print("Error: cleanlab is not installed.")
        print("Please install it with: pip install cleanlab")
        sys.exit(1)

    from cleanlab.filter import find_label_issues

    print("\nAnalyzing with Cleanlab...")

    # 使用 cleanlab 查找标签问题
    # 返回布尔数组，True 表示可能是标签错误
    label_issues = find_label_issues(
        labels=labels,
        pred_probs=probs,
        return_indices_ranked_by="self_confidence",
    )

    # 获取预测标签和置信度
    pred_labels = np.argmax(probs, axis=1)
    pred_confidence = np.max(probs, axis=1)

    # 计算标签质量分数（使用 normalized margin）
    # 分数越低，越可能是标签错误
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]  # 最高 - 次高
    label_quality_score = 1 - margin  # 转换为分数，越低越差

    # 构建结果列表
    results = []
    for idx in label_issues[:top_k]:
        results.append({
            "idx": int(idx),
            "image_path": image_paths[idx],
            "given_label": int(labels[idx]),
            "predicted_label": int(pred_labels[idx]),
            "confidence": float(pred_confidence[idx]),
            "label_quality_score": float(label_quality_score[idx]),
            "probabilities": {
                0: float(probs[idx, 0]),
                1: float(probs[idx, 1]),
            } if probs.shape[1] == 2 else probs[idx].tolist(),
        })

    return results


def print_results(results: List[Dict], split: str):
    """打印分析结果."""
    print(f"\n{'=' * 80}")
    print(f"Cleanlab Analysis Results - {split.upper()}")
    print(f"{'=' * 80}")

    if not results:
        print("No label issues found.")
        return

    print(f"\nTop {len(results)} potential label issues (ranked by confidence):\n")

    print(f"{'Rank':<6} {'Idx':<8} {'Given':<8} {'Pred':<8} {'Conf':<10} {'Score':<10} {'Image Path'}")
    print("-" * 120)

    for rank, item in enumerate(results, 1):
        path_display = item['image_path']
        if len(path_display) > 50:
            path_display = "..." + path_display[-47:]

        label_map = {0: "normal", 1: "fall"}
        given_str = label_map.get(item['given_label'], str(item['given_label']))
        pred_str = label_map.get(item['predicted_label'], str(item['predicted_label']))

        print(
            f"{rank:<6} "
            f"{item['idx']:<8} "
            f"{given_str:<8} "
            f"{pred_str:<8} "
            f"{item['confidence']:<10.4f} "
            f"{item['label_quality_score']:<10.4f} "
            f"{path_display}"
        )

    print(f"\n{'=' * 80}")
    print("Label Mapping: 0=normal, 1=fall")
    print("Score: Lower is more likely to be mislabeled")
    print(f"{'=' * 80}")


def main():
    args = parse_args()

    # 加载配置
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model_path = get_model_path(config, args.model)
    print(f"Loading model from: {model_path}")

    model_cfg = config.get("model", {})
    model = SimpleFallClassifier(
        model_path=model_path,
        dropout=model_cfg.get("dropout", 0.1),
        num_classes=model_cfg.get("num_classes", 2),
        fall_class_idx=model_cfg.get("fall_class_idx", 1),
    )
    model = model.to(device)
    model.eval()

    # 分析指定的 split
    splits_to_analyze = []
    if args.split == "all":
        splits_to_analyze = ["train", "val"]
    else:
        splits_to_analyze = [args.split]

    all_results = {}

    for split in splits_to_analyze:
        print(f"\n{'=' * 60}")
        print(f"Processing {split.upper()} set")
        print(f"{'=' * 60}")

        # 创建数据集
        dataset = create_dataset(config, split)
        if dataset is None or len(dataset) == 0:
            print(f"Skipping {split}: No data found")
            continue

        print(f"Dataset size: {len(dataset)} samples")

        # 获取图像路径
        image_paths = get_image_paths(dataset)

        # 创建 DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # 获取预测
        print("Running inference...")
        probs, labels, _ = get_predictions(model, dataloader, device)

        print(f"Collected {len(probs)} predictions")
        print(f"Label distribution: {np.bincount(labels, minlength=2)}")

        # 使用 cleanlab 分析
        results = analyze_with_cleanlab(probs, labels, image_paths, top_k=args.top_k)

        # 打印结果
        print_results(results, split)

        all_results[split] = results

    # 保存结果
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\nDone!")


if __name__ == "__main__":
    main()
