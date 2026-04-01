#!/bin/bash

# Pipeline演示脚本
# 展示完整的跌倒检测Pipeline

VIDEO="${1:-data/sample.mp4}"
CONFIG="${3:-configs/default.yaml}"

echo "Running Fall Detection Pipeline Demo..."
echo "Video: $VIDEO"
echo "Config: $CONFIG"
echo ""
echo "Controls:"
echo "  ESC - quit"
echo ""

python scripts/run_pipeline_demo.py
