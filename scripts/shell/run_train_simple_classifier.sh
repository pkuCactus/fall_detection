#!/bin/bash
set -e

# 简单图像分类器训练脚本（COCO格式标注，支持DDP、数据增强、LetterBox）
# 使用方式:
#   bash scripts/shell/run_train_simple_classifier.sh --config configs/training/simple_classifier.yaml
#   bash scripts/shell/run_train_simple_classifier.sh --config configs/training/simple_classifier.yaml --ngpus 2

CONFIG=""
NGPUS=1
OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG="$2"; shift 2 ;;
    --ngpus) NGPUS="$2"; shift 2 ;;
    --override) OVERRIDE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# 检查必需参数
if [[ -z "$CONFIG" ]]; then
  echo "Error: --config is required"
  echo "Usage: bash scripts/shell/run_train_simple_classifier.sh --config configs/training/simple_classifier.yaml [--ngpus 2]"
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: Config file not found: $CONFIG"
  exit 1
fi

echo "Starting simple classifier training..."
echo "  Config: ${CONFIG}"
echo "  GPUs: ${NGPUS}"
if [[ -n "$OVERRIDE" ]]; then
  echo "  Override: ${OVERRIDE}"
fi

OVERRIDE_ARG=""
if [[ -n "$OVERRIDE" ]]; then
  OVERRIDE_ARG="--override ${OVERRIDE}"
fi

if [ "${NGPUS}" -gt 1 ]; then
  echo "Starting DDP training on ${NGPUS} GPUs..."
  torchrun --nproc_per_node="${NGPUS}" \
    training/scripts/train_simple_classifier.py \
    --config "${CONFIG}" \
    ${OVERRIDE_ARG}
else
  echo "Starting single-GPU training..."
  python training/scripts/train_simple_classifier.py \
    --config "${CONFIG}" \
    ${OVERRIDE_ARG}
fi

echo "Training completed."
