#!/bin/bash

# 启动HTTP服务器预览 .superpowers/ 目录下的HTML文件
# 使用方式:
#   bash scripts/shell/run_superpowers_server.sh
#   bash scripts/shell/run_superpowers_server.sh 8080

PORT="${1:-8080}"
SUPERPOWERS_DIR=".superpowers"

if [ ! -d "${SUPERPOWERS_DIR}" ]; then
    echo "Error: ${SUPERPOWERS_DIR} directory not found!"
    exit 1
fi

echo "Starting HTTP server for ${SUPERPOWERS_DIR}/"
echo "URL: http://localhost:${PORT}"
echo "Press Ctrl+C to stop"
echo ""

# 查找可用的Python版本
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Error: Python not found!"
    exit 1
fi

cd "${SUPERPOWERS_DIR}" && "${PYTHON}" -m http.server "${PORT}"
