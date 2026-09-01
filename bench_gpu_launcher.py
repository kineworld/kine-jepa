#!/usr/bin/env python
# One-click KINE-Bench launcher for KineOne-WM / V-JEPA 2 on GPU.
#
# Wraps `python -m kinebench run` with the 98-clip real-trajectory eval defaults,
# writes JSON, then renders a self-contained HTML report (no server needed).
#
# Usage (on the user's GPU machine):
#   python bench_gpu_launcher.py --data-dir ../kine-datapipe/clips --device cuda
#   python bench_gpu_launcher.py --smoke            # CPU validation, no data needed
#
# All flags forward to kinebench; this script only sets sane defaults + reports.
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "kine-bench"))

from kinebench.__main__ import main as kinebench_main


def run_bench(argv):
    return kinebench_main(argv)


def _rows(d):
    out = []
    for k, v in d.items():
        if k in ("status",):
            continue
        if isinstance(v, float):
            v = f"{v:.4f}"
        out.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    return "\n".join(out)


def render_html(results, out_html):
    model = results.get("model", {})
    mname = model.get("name") if isinstance(model, dict) else model
    tasks = results.get("tasks", {})
    na = results.get("not_applicable", {})
    task_cards = []
    for name, payload in tasks.items():
        if not isinstance(payload, dict):
            continue
        body = _rows(payload)
        task_cards.append(f"""<div class="card"><h3>{name}</h3>
          <table><tbody>{body}</tbody></table></div>""")
    na_cards = []
    for name, payload in na.items():
        reason = payload.get("reason", "n/a") if isinstance(payload, dict) else str(payload)
        na_cards.append(f"""<div class="card na"><h3>{name} · n/a</h3>
          <p class="note">{reason}</p></div>""")
    wall = results.get("wall_s", "—")
    nclips = results.get("num_clips", "—")
    dev = results.get("device", "—")
    src = results.get("data", "—")
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KINE-Bench 评测报告 · {mname}</title>
<style>
  :root {{ --ink:#0f172a; --mut:#64748b; --bd:#e2e8f0; --bg:#fff; --ok:#16a34a; --acc:#2563eb; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
         margin:0; background:#f1f5f9; color:var(--ink); padding:28px; }}
  .wrap {{ max-width:960px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .lead {{ color:var(--mut); font-size:14px; margin:0 0 16px; }}
  .meta {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-bottom:18px; }}
  .meta div {{ background:var(--bg); border:1px solid var(--bd); border-radius:10px; padding:10px 12px; }}
  .meta b {{ display:block; font-size:12px; color:var(--mut); font-weight:600; }}
  .meta span {{ font-size:15px; font-weight:700; }}
  .card {{ background:var(--bg); border:1px solid var(--bd); border-radius:12px; padding:14px 16px; margin-bottom:12px; }}
  .card h3 {{ margin:0 0 8px; font-size:15px; }}
  .card.na {{ border-color:#fed7aa; background:#fff7ed; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--bd); }}
  .note {{ font-size:13px; color:var(--mut); line-height:1.6; margin:0; }}
  footer {{ color:var(--mut); font-size:12px; text-align:center; padding:18px; }}
</style></head><body><div class="wrap">
  <h1>KINE-Bench 评测报告 · {mname}</h1>
  <p class="lead">真实特征上的世界模型基准（KINE-Bench v{results.get('version','?')}）。缺失能力按协议报 n/a，不伪造。</p>
  <div class="meta">
    <div><b>模型</b><span>{mname}</span></div>
    <div><b>数据</b><span>{src}</span></div>
    <div><b>片段数</b><span>{nclips}</span></div>
    <div><b>设备</b><span>{dev}</span></div>
    <div><b>耗时</b><span>{wall}s</span></div>
  </div>
  <h2 style="font-size:16px;margin:6px 0 10px;">可运行任务</h2>
  {''.join(task_cards) if task_cards else '<p class="note">无（全部 n/a）</p>'}
  <h2 style="font-size:16px;margin:14px 0 10px;">不适用任务（诚实 n/a）</h2>
  {''.join(na_cards) if na_cards else '<p class="note">无</p>'}
  <footer>由 bench_gpu_launcher.py 生成 · 全部指标来自 KINE-Bench 可复现运行</footer>
</div></body></html>"""
    out_html.write_text(html, encoding="utf-8")
    return out_html


def main():
    ap = argparse.ArgumentParser(prog="bench_gpu_launcher",
        description="一键 KINE-Bench 评测（封装 CUDA + 98 条 + HTML 报告）")
    ap.add_argument("--data-dir", type=str, default="../kine-datapipe/clips")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    ap.add_argument("--max-clips", type=int, default=98)
    ap.add_argument("--model", type=str, default="vjepa2-vitl-256")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true", help="CPU 验证模式，无需数据")
    args = ap.parse_args()

    out_json = Path(args.out) if args.out else (ROOT / "bench_report.json")
    out_html = out_json.with_suffix(".html")

    argv = ["run", "--model", args.model, "--device", args.device,
            "--max-clips", str(args.max_clips), "--num-frames", str(args.num_frames),
            "--batch-size", str(args.batch_size), "--out", str(out_json)]
    if args.img_size:
        argv += ["--img-size", str(args.img_size)]
    if args.smoke:
        argv.append("--smoke")
    else:
        argv += ["--data-dir", str(args.data_dir)]

    print(f"[launcher] invoking kinebench: {' '.join(argv)}", flush=True)
    rc = run_bench(argv)
    if rc != 0:
        print(f"[launcher] kinebench exited {rc}; 见上方错误")
        return rc

    results = json.loads(out_json.read_text(encoding="utf-8"))
    render_html(results, out_html)
    print(f"[launcher] report -> {out_html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
