#!/bin/bash

# Pipeline演示脚本
# 展示完整的跌倒检测Pipeline

VIDEO="${1:-data/sample.mp4}"
OUTPUT="${2:-outputs/demo_output.mp4}"
shift 2

echo "Running Fall Detection Pipeline Demo..."
echo "Video: $VIDEO"
echo "Output: $OUTPUT"
echo ""
echo "Controls:"
echo "  ESC - quit"
echo ""

python scripts/demo/run_pipeline_demo.py --video $VIDEO --output $OUTPUT $@
