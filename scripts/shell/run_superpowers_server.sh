#!/bin/bash

# 启动HTTP服务器预览 .superpowers/ 和 docs/ 目录下的HTML文件
# 使用方式:
#   bash scripts/shell/run_superpowers_server.sh
#   bash scripts/shell/run_superpowers_server.sh 8080

PORT="${1:-8081}"

if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Error: Python not found!"
    exit 1
fi

TMP_DIR=$(mktemp -d)
trap "rm -rf ${TMP_DIR}" EXIT

"${PYTHON}" << PYTHON_SCRIPT
import os

src_dirs = [".superpowers", "docs"]
dst_dir = "${TMP_DIR}"
html_files = []

for src_dir in src_dirs:
    full_src = os.path.join(".", src_dir)
    if not os.path.isdir(full_src):
        continue
    for root, dirs, files in os.walk(full_src):
        for f in files:
            if not f.endswith('.html'):
                continue
            src_path = os.path.join(root, f)
            rel_path = os.path.relpath(src_path, ".")
            dst_path = os.path.join(dst_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            with open(src_path, 'r', encoding='utf-8') as fp:
                content = fp.read()
            if '<html' in content.lower() and '<head>' in content.lower():
                if 'charset' not in content.lower():
                    content = content.replace('<head>', '<head>\n    <meta charset="UTF-8">', 1)
                with open(dst_path, 'w', encoding='utf-8') as fp:
                    fp.write(content)
            else:
                title = os.path.splitext(f)[0].replace('-', ' ').replace('_', ' ').title()
                wrapped = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; padding: 20px; max-width: 1000px; margin: 0 auto; line-height: 1.6; }}
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
                with open(dst_path, 'w', encoding='utf-8') as fp:
                    fp.write(wrapped)
            html_files.append(rel_path)
            print(f"Processed: {rel_path}")

groups = {}
for f in html_files:
    d = os.path.dirname(f) or 'root'
    groups.setdefault(d, []).append(f)

lines = ['<!DOCTYPE html>', '<html>', '<head>', '    <meta charset="UTF-8">', '    <title>HTML Files</title>', '    <style>', '        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; padding: 40px; max-width: 900px; margin: 0 auto; }', '        h1 { border-bottom: 1px solid #ccc; padding-bottom: 10px; }', '        h2 { color: #555; font-size: 18px; margin-top: 24px; }', '        ul { list-style: none; padding: 0; }', '        li { margin: 8px 0; }', '        a { color: #0366d6; text-decoration: none; font-size: 16px; }', '        a:hover { text-decoration: underline; }', '        .dir { color: #888; font-size: 13px; }', '    </style>', '</head>', '<body>', '    <h1>HTML Files</h1>']
for d in sorted(groups.keys()):
    lines.append(f'    <h2>{d}/</h2>')
    lines.append('    <ul>')
    for f in sorted(groups[d]):
        lines.append(f'        <li><a href="{f}">{os.path.basename(f)}</a> <span class="dir">{f}</span></li>')
    lines.append('    </ul>')
lines += ['</body>', '</html>']

with open(os.path.join(dst_dir, 'index.html'), 'w', encoding='utf-8') as fp:
    fp.write('\n'.join(lines))

print("Done.")
PYTHON_SCRIPT

echo ""
echo "Starting HTTP server..."
echo "URL: http://localhost:${PORT}"
echo "Press Ctrl+C to stop"
echo ""

cd "${TMP_DIR}" && "${PYTHON}" -m http.server "${PORT}"
