#!/usr/bin/env python3
"""Write kineworld-site/exp002.html from scripts/compare_arms.py --out JSON."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

LABEL = {"A": "A 锚", "B": "B 负对照", "C": "C 正对照"}

TMPL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KINE-EXP-002 因果消融 — 勘境</title>
<style>
  :root {{ --bg:#0a0e14; --bg-soft:#101722; --fg:#e8edf4; --muted:#8b98a9; --accent:#38d9a9; --border:#1e2a3a; }}
  body {{ background:var(--bg); color:var(--fg); font-family:"Inter","PingFang SC",system-ui,sans-serif; line-height:1.7; margin:0; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:40px 24px; }}
  a {{ color:var(--accent); text-decoration:none; }}
  table {{ width:100%; border-collapse:collapse; margin-top:20px; }}
  th,td {{ border:1px solid var(--border); padding:10px 12px; text-align:left; }}
  th {{ color:var(--accent); }}
  .wait {{ color:#f0c36d; }}
  .ok {{ color:var(--accent); font-weight:700; }}
  .bad {{ color:#ff6b6b; font-weight:700; }}
</style>
</head>
<body>
<div class="wrap">
  <p><a href="/">勘境</a> · <a href="/report.html">EXP-001</a> · <a href="/vs-baize.html">对照场</a></p>
  <h1>KINE-EXP-002 因果消融</h1>
  <p>冻结 EXP-001 编码器。A 无干预 / B 随机标签 / C 对齐标签。本页由 compare_arms JSON 生成，不手填。</p>
  <table>
    <tr><th>臂</th><th>EVT-1</th><th>CAU-1</th><th>FUT-1</th></tr>
{rows}
  </table>
  <p class="{klass}">裁决：{verdict} — {notes}</p>
  <p>通过线：C EVT-1 ≥ 0.58 且 &gt; A；CAU 比 A 高 ≥ 0.08；FUT 回撤 ≤ 0.03；B 不胜 C。</p>
  <p>更新于 {stamp} UTC</p>
</div>
</body>
</html>
"""

def cell(v):
    if v is None:
        return '<td class="wait">pending</td>'
    return f"<td>{v:.3f}</td>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    arms = data["arms"]
    lines = []
    for arm in "ABC":
        r = arms[arm]
        lines.append(
            f"    <tr><td>{LABEL[arm]}</td>{cell(r.get('KINE-EVT-1'))}{cell(r.get('KINE-CAU-1'))}{cell(r.get('KINE-FUT-1'))}</tr>"
        )
    ok = bool(data.get("pass"))
    html = TMPL.format(
        rows="\n".join(lines),
        klass="ok" if ok else "bad",
        verdict="PASS" if ok else "FAIL",
        notes="; ".join(data.get("notes") or []),
        stamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
    )
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out} verdict={'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    main()
