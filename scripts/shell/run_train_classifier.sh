#!/bin/bash
set -e

# 融合分类器训练脚本（config-based）
# 使用方式:
#   bash scripts/shell/run_train_classifier.sh --config configs/training/classifier.yaml
#   bash scripts/shell/run_train_classifier.sh --config configs/training/classifier.yaml --ngpus 2
#   bash scripts/shell/run_train_classifier.sh --config configs/training/classifier.yaml --override "epochs=100,batch_size=16"

CONFIG="configs/training/classifier.yaml"
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

if [ "${NGPUS}" -gt 1 ]; then
  echo "Starting DDP classifier training on ${NGPUS} GPUs..."
  echo "Config: ${CONFIG}"
  if [ -n "${OVERRIDE}" ]; then
    echo "Override: ${OVERRIDE}"
  fi
  torchrun --nproc_per_node="${NGPUS}" \
    scripts/train/train_classifier.py \
    --config "${CONFIG}" \
    ${OVERRIDE:+--override "${OVERRIDE}"}
else
  echo "Starting single-GPU classifier training..."
  echo "Config: ${CONFIG}"
  if [ -n "${OVERRIDE}" ]; then
    echo "Override: ${OVERRIDE}"
  fi
  python scripts/train/train_classifier.py \
    --config "${CONFIG}" \
    ${OVERRIDE:+--override "${OVERRIDE}"}
fi
