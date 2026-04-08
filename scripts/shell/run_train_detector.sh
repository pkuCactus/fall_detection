#!/bin/bash
set -e

# 检测器训练脚本（config-based）
# 使用方式:
#   bash scripts/shell/run_train_detector.sh --config configs/training/detector.yaml
#   bash scripts/shell/run_train_detector.sh --config configs/training/detector.yaml --override "epochs=100,batch=8"

CONFIG="configs/training/detector.yaml"
OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG="$2"; shift 2 ;;
    --override) OVERRIDE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "Config: ${CONFIG}"
if [ -n "${OVERRIDE}" ]; then
  echo "Override: ${OVERRIDE}"
fi

PYTHONPATH=src:$PYTHONPATH \
python scripts/train/train_detector.py \
  --config "${CONFIG}" \
  ${OVERRIDE:+--override "${OVERRIDE}"}
