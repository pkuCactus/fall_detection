#!/bin/bash
set -e

# 特征提取脚本（单进程，无需DDP）
# 使用方式:
#   bash scripts/shell/run_extract_features.sh
#   bash scripts/shell/run_extract_features.sh --video-dir /path/to/videos --out-dir outputs/cache

VIDEO_DIR="data/videos"
LABEL_FILE="data/labels.json"
CONFIG="configs/default.yaml"
OUT_DIR="outputs/cache"
SAMPLE_FPS=5

while [[ $# -gt 0 ]]; do
  case $1 in
    --video-dir) VIDEO_DIR="$2"; shift 2 ;;
    --label-file) LABEL_FILE="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --sample-fps) SAMPLE_FPS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

python scripts/train/extract_features.py \
  --video-dir "${VIDEO_DIR}" \
  --label-file "${LABEL_FILE}" \
  --config "${CONFIG}" \
  --out-dir "${OUT_DIR}" \
  --sample-fps "${SAMPLE_FPS}"
