#!/bin/bash

# 启动HTTP服务器预览 .superpowers/ 目录下的HTML文件
# 使用方式:
#   bash scripts/shell/run_superpowers_server.sh
#   bash scripts/shell/run_superpowers_server.sh 8080

PORT="${1:-8081}"
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

# 使用Python处理HTML文件，添加UTF-8 meta标签
"${PYTHON}" << PYTHON_SCRIPT
import os
import re

src_dir = "${SUPERPOWERS_DIR}"
dst_dir = "${TMP_DIR}"

html_template = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "微软雅黑", sans-serif; padding: 20px; max-width: 1000px; margin: 0 auto; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        .subtitle {{ color: #666; font-size: 14px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
        pre {{ background: #f4f4f4; padding: 16px; border-radius: 6px; overflow-x: auto; }}
        .highlight {{ background: #fff3cd; padding: 2px 4px; border-radius: 3px; }}
        .decision {{ background: #d4edda; padding: 12px; border-radius: 6px; margin: 10px 0; }}
        .module {{ background: #e7f3ff; padding: 12px; border-radius: 6px; margin: 10px 0; border-left: 4px solid #0366d6; }}
    </style>
</head>
<body>
{content}
</body>
</html>'''

for root, dirs, files in os.walk(src_dir):
    for f in files:
        if f.endswith('.html'):
            src_path = os.path.join(root, f)
            dst_path = os.path.join(dst_dir, f)

            with open(src_path, 'r', encoding='utf-8') as fp:
                content = fp.read()

            # 如果文件已经有完整的HTML结构，只添加/确保meta charset
            if '<html' in content.lower() and '<head>' in content.lower():
                # 添加meta charset如果还没有
                if 'charset' not in content.lower():
                    content = content.replace('<head>', '<head>\n    <meta charset="UTF-8">', 1)
                with open(dst_path, 'w', encoding='utf-8') as fp:
                    fp.write(content)
            else:
                # 包裹完整HTML结构
                title = os.path.splitext(f)[0].replace('-', ' ').replace('_', ' ').title()
                wrapped = html_template.format(title=title, content=content)
                with open(dst_path, 'w', encoding='utf-8') as fp:
                    fp.write(wrapped)

            print(f"Processed: {f}")

print("Done processing HTML files.")
PYTHON_SCRIPT

# 生成只包含三个文件的简洁索引页
cat > "${TMP_DIR}/index.html" << 'HTMLEOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Superpowers HTML Files</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", "微软雅黑", sans-serif; padding: 40px; max-width: 800px; margin: 0 auto; }
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
