#!/bin/bash
set -e

# 检测器训练脚本（支持DDP）
# 使用方式:
#   bash scripts/shell/run_train_detector.sh
#   bash scripts/shell/run_train_detector.sh --ngpus 2 --batch 8

DATA="data/fall_detection.yaml"
EPOCHS=50
IMGSZ=832
BATCH=16
MODEL="yolov8n.pt"
PROJECT="train/detector"
NAME="exp"
NGPUS=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --data) DATA="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --ngpus) NGPUS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ "${NGPUS}" -gt 1 ]; then
  echo "Starting DDP detector training on ${NGPUS} GPUs..."
  torchrun --nproc_per_node="${NGPUS}" \
    scripts/train_detector.py \
    --data "${DATA}" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --model "${MODEL}" \
    --project "${PROJECT}" \
    --name "${NAME}"
else
  echo "Starting single-GPU detector training..."
  python scripts/train_detector.py \
    --data "${DATA}" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --model "${MODEL}" \
    --project "${PROJECT}" \
    --name "${NAME}"
fi
