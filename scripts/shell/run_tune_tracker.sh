#!/bin/bash
set -e

# 跟踪器参数调优脚本（单进程，无需DDP）
# 使用方式:
#   bash scripts/shell/run_tune_tracker.sh
#   bash scripts/shell/run_tune_tracker.sh --video-dir /path/to/videos

VIDEO_DIR="data/videos"
OUTPUT="train/tracker/tune_result.json"

while [[ $# -gt 0 ]]; do
  case $1 in
    --video-dir) VIDEO_DIR="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

python scripts/tune_tracker.py \
  --video-dir "${VIDEO_DIR}" \
  --output "${OUTPUT}"
