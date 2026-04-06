#!/bin/bash
set -e

# YOLOWorld验证/测试脚本
# 支持非正方形分辨率 (WxH 格式)
# 使用方式:
#   bash scripts/shell/run_validate_yolo_world.sh --weights outputs/yolo_world/exp/weights/best.pt
#   bash scripts/shell/run_validate_yolo_world.sh --weights best.pt --imgsz 832x448 --data data/fall_detection.yaml

WEIGHTS="outputs/yolo_world/exp/weights/best.pt"
DATA="data/fall_detection.yaml"
IMGSZ="832x448"  # 支持 WxH 格式: "832x448" 或整数 "640"
BATCH=16
DEVICE="0"
CONF=0.001
IOU=0.6
SPLIT="val"  # val 或 test

while [[ $# -gt 0 ]]; do
  case $1 in
    --weights) WEIGHTS="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --conf) CONF="$2"; shift 2 ;;
    --iou) IOU="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "Starting YOLOWorld validation..."
echo "Weights: ${WEIGHTS}"
echo "Image size: ${IMGSZ}"
echo "Data: ${DATA}"
echo "Split: ${SPLIT}"

python scripts/train/validate_yolo_world.py \
  --weights "${WEIGHTS}" \
  --data "${DATA}" \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --device "${DEVICE}" \
  --conf "${CONF}" \
  --iou "${IOU}" \
  --split "${SPLIT}"
