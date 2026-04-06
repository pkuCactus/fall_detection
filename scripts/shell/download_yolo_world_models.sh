#!/bin/bash
set -e

# 下载 YOLO-World 预训练模型
# 使用方式:
#   bash scripts/shell/download_yolo_world_models.sh [model_size] [version]
#   bash scripts/shell/download_yolo_world_models.sh l v2.1    # 下载 yolov8l-worldv2.1.pt
#   bash scripts/shell/download_yolo_world_models.sh l v2      # 下载 yolov8l-worldv2.pt

SIZE="${1:-l}"      # 默认 l 模型
VERSION="${2:-v2}"  # 默认 v2 版本 (v2.1 可能不存在)

# 模型文件名
MODEL="yolov8${SIZE}-world${VERSION}.pt"

# 尝试多个可能的下载源
URLS=(
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/${MODEL}"
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/${MODEL}"
    "https://github.com/ultralytics/assets/releases/download/v8.1.0/${MODEL}"
    "https://github.com/ultralytics/assets/releases/latest/download/${MODEL}"
)

echo "Attempting to download ${MODEL}..."

DOWNLOADED=false
for URL in "${URLS[@]}"; do
    echo "Trying: ${URL}"

    if command -v wget > /dev/null 2>&1; then
        if wget -q --spider "${URL}" 2>/dev/null; then
            wget -c "${URL}" -O "${MODEL}"
            DOWNLOADED=true
            break
        fi
    elif command -v curl > /dev/null 2>&1; then
        HTTP_CODE=$(curl -L -s -o /dev/null -w "%{http_code}" "${URL}")
        if [ "$HTTP_CODE" = "200" ]; then
            curl -L -o "${MODEL}" "${URL}"
            DOWNLOADED=true
            break
        fi
    fi
done

if [ "$DOWNLOADED" = true ] && [ -f "${MODEL}" ]; then
    echo ""
    echo "Download complete: ${MODEL}"
    ls -lh "${MODEL}"
else
    echo ""
    echo "Error: Failed to download ${MODEL}"
    echo ""
    echo "Possible reasons:"
    echo "1. Model version '${VERSION}' does not exist"
    echo "2. Model size '${SIZE}' is invalid (valid: s, m, l, x)"
    echo ""
    echo "Try using v2 instead of v2.1:"
    echo "  bash scripts/shell/download_yolo_world_models.sh ${SIZE} v2"
    echo ""
    echo "Or use Python to auto-download:"
    echo "  python -c \"from ultralytics import YOLOWorld; YOLOWorld('${MODEL}')\""
    exit 1
fi
