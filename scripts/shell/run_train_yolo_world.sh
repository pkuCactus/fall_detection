#!/bin/bash
set -e

# YOLOWorld训练脚本（config-based，类似simple_classifier）
# 使用方式:
#   bash scripts/shell/run_train_yolo_world.sh --config configs/training/yolo_world.yaml
#   bash scripts/shell/run_train_yolo_world.sh --config configs/training/yolo_world.yaml --ngpus 2
#   bash scripts/shell/run_train_yolo_world.sh --config configs/training/yolo_world.yaml --override "epochs=100,batch=8"

CONFIG="configs/training/yolo_world.yaml"
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
  echo "Starting DDP YOLOWorld training on ${NGPUS} GPUs..."
  echo "Config: ${CONFIG}"
  if [ -n "${OVERRIDE}" ]; then
    echo "Override: ${OVERRIDE}"
  fi
  torchrun --nproc_per_node="${NGPUS}" \
    scripts/train/train_yolo_world.py \
    --config "${CONFIG}" \
    ${OVERRIDE:+--override "${OVERRIDE}"}
else
  echo "Starting single-GPU YOLOWorld training..."
  echo "Config: ${CONFIG}"
  if [ -n "${OVERRIDE}" ]; then
    echo "Override: ${OVERRIDE}"
  fi
  python scripts/train/train_yolo_world.py \
    --config "${CONFIG}" \
    ${OVERRIDE:+--override "${OVERRIDE}"}
fi
