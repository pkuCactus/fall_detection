#!/bin/bash

# Tracker演示脚本
# 展示ByteTrack-lite跟踪效果

VIDEO="${1:-data/sample.mp4}"
OUTPUT="${2:-}"

echo "Running Tracker Demo..."
echo "Video: $VIDEO"

if [ -n "$OUTPUT" ]; then
    python deployment/demo_tracker.py --video "$VIDEO" --output "$OUTPUT"
else
    python deployment/demo_tracker.py --video "$VIDEO"
fi
