#!/bin/bash

# Pipeline演示脚本
# 展示完整的跌倒检测Pipeline

VIDEO="${1:-data/sample.mp4}"
OUTPUT="${2:-outputs/demo_output.mp4}"

echo "Running Fall Detection Pipeline Demo..."
echo "Video: $VIDEO"
echo "Output: $OUTPUT"
echo ""
echo "Controls:"
echo "  ESC - quit"
echo ""

python deployment/run_pipeline_demo.py --video $VIDEO --output $OUTPUT $@
