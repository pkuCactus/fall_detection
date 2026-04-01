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

# 查找可用的Python版本
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Error: Python not found!"
    exit 1
fi

# 创建临时目录，只包含三个HTML文件
echo "Creating temporary directory with HTML files only..."
TMP_DIR=$(mktemp -d)
trap "rm -rf ${TMP_DIR}" EXIT

# 查找并复制HTML文件
find "${SUPERPOWERS_DIR}" -name "*.html" -type f | while read -r file; do
    cp "${file}" "${TMP_DIR}/"
done

# 生成只包含三个文件的简洁索引页
cat > "${TMP_DIR}/index.html" << 'HTMLEOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Superpowers HTML Files</title>
    <style>
        body { font-family: sans-serif; padding: 40px; max-width: 800px; margin: 0 auto; }
        h1 { border-bottom: 1px solid #ccc; padding-bottom: 10px; }
        ul { list-style: none; padding: 0; }
        li { margin: 15px 0; }
        a { color: #0366d6; text-decoration: none; font-size: 18px; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>HTML Files</h1>
    <ul>
HTMLEOF

# 动态添加HTML文件列表
for file in "${TMP_DIR}"/*.html; do
    if [ "$(basename "${file}")" != "index.html" ]; then
        filename=$(basename "${file}")
        echo "        <li><a href=\"${filename}\">${filename}</a></li>" >> "${TMP_DIR}/index.html"
    fi
done

cat >> "${TMP_DIR}/index.html" << 'HTMLEOF'
    </ul>
</body>
</html>
HTMLEOF

echo ""
echo "Starting HTTP server..."
echo "URL: http://localhost:${PORT}"
echo "Press Ctrl+C to stop"
echo ""

cd "${TMP_DIR}" && "${PYTHON}" -m http.server "${PORT}"
