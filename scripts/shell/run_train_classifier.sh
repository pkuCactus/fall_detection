#!/bin/bash
set -e

# 分类器训练脚本（支持DDP）
# 使用方式:
#   bash scripts/shell/run_train_classifier.sh
#   bash scripts/shell/run_train_classifier.sh --ngpus 2 --batch-size 16

CACHE_DIR="outputs/cache"
EPOCHS=100
BATCH_SIZE=32
LR=0.001
VAL_RATIO=0.2
OUTPUT_DIR="outputs/classifier"
NGPUS=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --ngpus) NGPUS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

if [ "${NGPUS}" -gt 1 ]; then
  echo "Starting DDP classifier training on ${NGPUS} GPUs..."
  torchrun --nproc_per_node="${NGPUS}" \
    training/scripts/train_classifier.py \
    --cache-dir "${CACHE_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --val-ratio "${VAL_RATIO}" \
    --output-dir "${OUTPUT_DIR}"
else
  echo "Starting single-GPU classifier training..."
  python training/scripts/train_classifier.py \
    --cache-dir "${CACHE_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --val-ratio "${VAL_RATIO}" \
    --output-dir "${OUTPUT_DIR}"
fi
