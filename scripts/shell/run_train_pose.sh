#!/bin/bash
set -e

# 姿态估计器训练/微调脚本（支持DDP）
# 使用方式:
#   bash scripts/shell/run_train_pose.sh
#   bash scripts/shell/run_train_pose.sh --ngpus 2 --batch 32

DATA="data/fall_pose.yaml"
EPOCHS=100
IMGSZ=128
BATCH=64
MODEL="yolov8n-pose.pt"
PROJECT="train/pose"
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
  echo "Starting DDP pose training on ${NGPUS} GPUs..."
  torchrun --nproc_per_node="${NGPUS}" \
    scripts/train_pose.py \
    --data "${DATA}" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --model "${MODEL}" \
    --project "${PROJECT}" \
    --name "${NAME}"
else
  echo "Starting single-GPU pose training..."
  python scripts/train_pose.py \
    --data "${DATA}" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMGSZ}" \
    --batch "${BATCH}" \
    --model "${MODEL}" \
    --project "${PROJECT}" \
    --name "${NAME}"
fi
