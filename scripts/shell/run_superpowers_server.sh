#!/bin/bash

# 启动 HTTP 服务器预览 .superpowers/ 和 docs/ 目录下的 HTML 文件
# 使用方式:
#   bash scripts/shell/run_superpowers_server.sh
#   bash scripts/shell/run_superpowers_server.sh 8080

PORT="${1:-8081}"

if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "\033[31mError: Python not found!\033[0m"
    exit 1
fi

TMP_DIR=$(mktemp -d)
trap "rm -rf ${TMP_DIR}" EXIT

"${PYTHON}" - "${TMP_DIR}" << 'PYTHON_SCRIPT'
import os, sys, datetime

dst_dir = sys.argv[1]
src_dirs = [".superpowers", "docs"]
html_files = []
found_any = False

# ---------- helpers ----------
def wrap_fragment(content, title):
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{ --bg:#0f172a; --card:#1e293b; --text:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; --accent2:#818cf8; --border:#334155; --hover:#253449; }}
        * {{ box-sizing:border-box; margin:0; padding:0; }}
        body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text); padding:24px; line-height:1.7; }}
        .container {{ max-width:960px; margin:0 auto; }}
        h1 {{ font-size:26px; margin-bottom:8px; font-weight:600; }}
        .subtitle {{ color:var(--muted); font-size:14px; margin-bottom:24px; }}
        .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:16px; }}
        .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; transition:all .2s; cursor:pointer; text-decoration:none; color:inherit; display:block; }}
        .card:hover {{ background:var(--hover); border-color:var(--accent); transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,0,0,.3); }}
        .card-title {{ font-size:16px; font-weight:600; margin-bottom:6px; display:flex; align-items:center; gap:8px; }}
        .card-title::before {{ content:"📄"; font-size:18px; }}
        .card-path {{ font-size:12px; color:var(--muted); font-family:monospace; word-break:break-all; }}
        .section-title {{ font-size:13px; text-transform:uppercase; letter-spacing:1px; color:var(--accent); margin:32px 0 16px; font-weight:600; display:flex; align-items:center; gap:8px; }}
        .section-title::before {{ content:""; display:inline-block; width:4px; height:16px; background:var(--accent); border-radius:2px; }}
        .badge {{ display:inline-block; background:rgba(56,189,248,.15); color:var(--accent); padding:2px 8px; border-radius:12px; font-size:11px; margin-left:auto; }}
        .empty {{ color:var(--muted); text-align:center; padding:60px 20px; }}
        .empty-icon {{ font-size:48px; margin-bottom:12px; opacity:.5; }}
        @media (max-width:600px) {{ .grid {{ grid-template-columns:1fr; }} body {{ padding:16px; }} }}
        table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:14px; }}
        th,td {{ border:1px solid var(--border); padding:10px 12px; text-align:left; }}
        th {{ background:var(--hover); color:var(--accent); font-weight:500; }}
        code {{ background:rgba(129,140,248,.15); color:var(--accent2); padding:2px 6px; border-radius:4px; font-family:monospace; font-size:13px; }}
        pre {{ background:var(--bg); padding:16px; border-radius:8px; overflow-x:auto; border:1px solid var(--border); }}
        .highlight {{ background:rgba(234,179,8,.15); color:#facc15; padding:2px 6px; border-radius:4px; }}
        .decision {{ background:rgba(34,197,94,.1); border-left:4px solid #22c55e; padding:12px; border-radius:6px; margin:10px 0; }}
        .module {{ background:rgba(56,189,248,.1); border-left:4px solid var(--accent); padding:12px; border-radius:6px; margin:10px 0; }}
    </style>
</head>
<body>
<div class="container">
{content}
</div>
</body>
</html>'''

def build_index(groups, total):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation Index</title>
    <style>
        :root {{ --bg:#0f172a; --card:#1e293b; --text:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; --accent2:#818cf8; --border:#334155; --hover:#253449; }}
        * {{ box-sizing:border-box; margin:0; padding:0; }}
        body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text); padding:24px; line-height:1.7; min-height:100vh; }}
        .container {{ max-width:960px; margin:0 auto; }}
        header {{ margin-bottom:32px; padding-bottom:20px; border-bottom:1px solid var(--border); }}
        h1 {{ font-size:28px; font-weight:700; background:linear-gradient(90deg,var(--accent),var(--accent2)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; display:inline-block; }}
        .meta {{ color:var(--muted); font-size:14px; margin-top:6px; display:flex; align-items:center; gap:16px; flex-wrap:wrap; }}
        .meta span {{ display:flex; align-items:center; gap:4px; }}
        .search-box {{ width:100%; padding:12px 16px; border-radius:10px; border:1px solid var(--border); background:var(--card); color:var(--text); font-size:15px; margin-bottom:24px; outline:none; transition:border-color .2s; }}
        .search-box:focus {{ border-color:var(--accent); }}
        .search-box::placeholder {{ color:var(--muted); }}
        .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:16px; }}
        .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px; transition:all .2s; cursor:pointer; text-decoration:none; color:inherit; display:block; }}
        .card:hover {{ background:var(--hover); border-color:var(--accent); transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,0,0,.3); }}
        .card.hidden {{ display:none; }}
        .card-title {{ font-size:15px; font-weight:600; margin-bottom:6px; display:flex; align-items:center; gap:8px; }}
        .card-title::before {{ content:"📄"; font-size:18px; }}
        .card-path {{ font-size:12px; color:var(--muted); font-family:monospace; word-break:break-all; }}
        .section-title {{ font-size:13px; text-transform:uppercase; letter-spacing:1px; color:var(--accent); margin:28px 0 14px; font-weight:600; display:flex; align-items:center; gap:8px; }}
        .section-title::before {{ content:""; display:inline-block; width:4px; height:16px; background:var(--accent); border-radius:2px; }}
        .badge {{ display:inline-block; background:rgba(56,189,248,.15); color:var(--accent); padding:2px 8px; border-radius:12px; font-size:11px; margin-left:auto; }}
        .empty {{ color:var(--muted); text-align:center; padding:60px 20px; }}
        .empty-icon {{ font-size:48px; margin-bottom:12px; opacity:.5; }}
        footer {{ margin-top:48px; padding-top:20px; border-top:1px solid var(--border); color:var(--muted); font-size:12px; text-align:center; }}
        @media (max-width:600px) {{ .grid {{ grid-template-columns:1fr; }} body {{ padding:16px; }} }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Documentation</h1>
        <div class="meta">
            <span>📁 {total} file(s)</span>
            <span>🕐 {now}</span>
        </div>
    </header>
    <input type="text" class="search-box" id="search" placeholder="Search files...">''']
    if not html_files:
        lines.append('''    <div class="empty">
        <div class="empty-icon">📂</div>
        <p>No HTML files found in .superpowers/ or docs/</p>
    </div>''')
    else:
        for d in sorted(groups.keys()):
            count = len(groups[d])
            lines.append(f'    <div class="section-title">{d}/ <span class="badge">{count}</span></div>')
            lines.append('    <div class="grid">')
            for f in sorted(groups[d]):
                name = os.path.splitext(os.path.basename(f))[0]
                lines.append(f'        <a class="card" href="{f}">\n            <div class="card-title">{name}</div>\n            <div class="card-path">{f}</div>\n        </a>')
            lines.append('    </div>')
    lines += ['    <footer>Powered by run_superpowers_server.sh</footer>',
              '</div>',
              '<script>',
              '    const search = document.getElementById(\'search\');',
              '    const cards = document.querySelectorAll(\'.card\');',
              '    const sections = document.querySelectorAll(\'.section-title\');',
              '    search.addEventListener(\'input\', e => {',
              '        const q = e.target.value.toLowerCase();',
              '        cards.forEach(c => {',
              '            const text = c.textContent.toLowerCase();',
              '            c.classList.toggle(\'hidden\', !text.includes(q));',
              '        });',
              '        sections.forEach(s => {',
              '            const next = s.nextElementSibling;',
              '            if (next && next.classList.contains(\'grid\')) {',
              '                const visible = next.querySelectorAll(\'.card:not(.hidden)\');',
              '                s.style.display = visible.length ? \'\' : \'none\';',
              '            }',
              '        });',
              '    });',
              '</script>',
              '</body>',
              '</html>']
    return '\n'.join(lines)

# ---------- process source files ----------
for src_dir in src_dirs:
    full_src = os.path.join(".", src_dir)
    if not os.path.isdir(full_src):
        continue
    found_any = True
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
            low = content.lower()
            if '<html' in low and '<head>' in low:
                if 'charset' not in low:
                    content = content.replace('<head>', '<head>\n    <meta charset="UTF-8">', 1)
                with open(dst_path, 'w', encoding='utf-8') as fp:
                    fp.write(content)
            else:
                title = os.path.splitext(f)[0].replace('-', ' ').replace('_', ' ').title()
                with open(dst_path, 'w', encoding='utf-8') as fp:
                    fp.write(wrap_fragment(content, title))
            html_files.append(rel_path)
            print(f"Processed: {rel_path}")

if not found_any:
    print("Warning: .superpowers/ or docs/ directories not found.")

# ---------- build index ----------
groups = {}
for f in html_files:
    d = os.path.dirname(f) or 'root'
    groups.setdefault(d, []).append(f)

with open(os.path.join(dst_dir, 'index.html'), 'w', encoding='utf-8') as fp:
    fp.write(build_index(groups, len(html_files)))

print(f"Done. {len(html_files)} file(s) indexed.")
PYTHON_SCRIPT

echo ""
echo -e "\033[36mStarting HTTP server...\033[0m"
echo -e "\033[32mURL: http://localhost:${PORT}\033[0m"
echo -e "\033[33mPress Ctrl+C to stop\033[0m"
echo ""

cd "${TMP_DIR}" && "${PYTHON}" -m http.server "${PORT}"
