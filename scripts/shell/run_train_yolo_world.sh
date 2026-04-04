#!/bin/bash
set -e

# YOLOWorld训练脚本（支持DDP）
# 使用方式:
#   bash scripts/shell/run_train_yolo_world.sh
#   bash scripts/shell/run_train_yolo_world.sh --ngpus 2 --batch 8

DATA="data/fall_detection.yaml"
EPOCHS=50
IMGSZ_W=832
IMGSZ_H=448
BATCH=16
MODEL="yolov8l-worldv2.pt"
PROJECT="outputs/yolo_world"
NAME="exp"
NGPUS=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --data) DATA="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --imgsz) IMGSZ_W="$2"; IMGSZ_H="$2"; shift 2 ;;
    --imgsz-w) IMGSZ_W="$2"; shift 2 ;;
    --imgsz-h) IMGSZ_H="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --ngpus) NGPUS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ "${NGPUS}" -gt 1 ]; then
  echo "Starting DDP YOLOWorld training on ${NGPUS} GPUs..."
  echo "Image size: ${IMGSZ_W}x${IMGSZ_H}"
  torchrun --nproc_per_node="${NGPUS}" \
    training/scripts/train_yolo_world.py \
    --data "${DATA}" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMGSZ_W}" "${IMGSZ_H}" \
    --batch "${BATCH}" \
    --model "${MODEL}" \
    --project "${PROJECT}" \
    --name "${NAME}"
else
  echo "Starting single-GPU YOLOWorld training..."
  echo "Image size: ${IMGSZ_W}x${IMGSZ_H}"
  python training/scripts/train_yolo_world.py \
    --data "${DATA}" \
    --epochs "${EPOCHS}" \
    --imgsz "${IMGSZ_W}" "${IMGSZ_H}" \
    --batch "${BATCH}" \
    --model "${MODEL}" \
    --project "${PROJECT}" \
    --name "${NAME}"
fi
