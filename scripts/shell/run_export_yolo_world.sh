#!/bin/bash
set -e

# YOLOWorld导出脚本
# 支持非正方形分辨率 (WxH 格式)
# 使用方式:
#   bash scripts/shell/run_export_yolo_world.sh --weights outputs/yolo_world/exp/weights/best.pt
#   bash scripts/shell/run_export_yolo_world.sh --weights best.pt --imgsz 832x448 --format onnx

WEIGHTS="outputs/yolo_world/exp/weights/best.pt"
IMGSZ="832x448"  # 支持 WxH 格式: "832x448" 或整数 "640"
FORMAT="onnx"    # 导出格式: onnx, engine, tflite, etc.
DEVICE="0"
HALF=false       # FP16半精度
INT8=false       # INT8量化
DYNAMIC=false    # 动态批次

while [[ $# -gt 0 ]]; do
  case $1 in
    --weights) WEIGHTS="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --format) FORMAT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --half) HALF="$2"; shift 2 ;;
    --int8) INT8="$2"; shift 2 ;;
    --dynamic) DYNAMIC="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# 解析IMGSZ为WIDTH和HEIGHT
if [[ "$IMGSZ" == *"x"* ]]; then
  WIDTH=$(echo "$IMGSZ" | cut -d'x' -f1)
  HEIGHT=$(echo "$IMGSZ" | cut -d'x' -f2)
else
  WIDTH=$IMGSZ
  HEIGHT=$IMGSZ
fi

echo "Starting YOLOWorld export..."
echo "Weights: ${WEIGHTS}"
echo "Image size: ${WIDTH}x${HEIGHT}"
echo "Format: ${FORMAT}"

python scripts/train/export_yolo_world.py \
  --weights "${WEIGHTS}" \
  --imgsz "${IMGSZ}" \
  --format "${FORMAT}" \
  --device "${DEVICE}" \
  --half "${HALF}" \
  --int8 "${INT8}" \
  --dynamic "${DYNAMIC}"
