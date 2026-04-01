#!/bin/bash
set -e

# 端到端评估与阈值搜索脚本（单进程，无需DDP）
# 使用方式:
#   bash scripts/shell/run_evaluate_pipeline.sh
#   bash scripts/shell/run_evaluate_pipeline.sh --mock-detector

VIDEO_DIR="data/videos"
GT_FILE="data/event_gt.json"
CONFIG="configs/default.yaml"
OUTPUT="train/eval/eval_result.json"
MOCK_DETECTOR=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --video-dir) VIDEO_DIR="$2"; shift 2 ;;
    --gt-file) GT_FILE="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --mock-detector) MOCK_DETECTOR="--mock-detector"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

python scripts/evaluate_pipeline.py \
  --video-dir "${VIDEO_DIR}" \
  --gt-file "${GT_FILE}" \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  ${MOCK_DETECTOR}
