#!/usr/bin/env python
# KINE-Bench 分辨率-吞吐扫描（GPU）
#
# 目的：回答评审最可能追问的一句——"你的 64px 数字是不是降采样省算力换来的？"
# 做法：在同一块 GPU、同一份合成片段上跑 64 / 128 / 256 三档分辨率，
#       记录每档墙钟与每片段耗时，输出 gpu_sweep.json + HTML。
#
# 用法（Git Bash）：
#   export KINE_VJEPA2_LOCAL="C:/Users/zoah/AppData/Local/Temp/vjepa2" \
#          HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
#   python gpu_resolution_sweep.py --device cuda --clips "64:32,128:32,256:16"
#   python gpu_resolution_sweep.py --device cuda --clips "64:4,128:4,256:2"   # 快速自检
#
# 说明：合成片段无真实运动结构，TEMP/MOT 分数无意义（见诚实边界）；
#       本脚本只产出**吞吐与可运行性**证据，不产出竞争力分数。
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
LAUNCHER = ROOT / "bench_gpu_launcher.py"


def parse_clips(spec):
    """'64:32,128:32,256:16' -> [(64,32),(128,32),(256,16)]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            out.append((int(a), int(b)))
        else:
            out.append((int(part), 32))
    return out


def run_one(img_size, n_clips, device, num_frames, batch_size, tag):
    out_json = ROOT / f"sweep_{img_size}.json"
    cmd = [PY, "-B", "-u", str(LAUNCHER),
           "--synthetic", "--device", device,
           "--img-size", str(img_size),
           "--max-clips", str(n_clips),
           "--num-frames", str(num_frames),
           "--batch-size", str(batch_size),
           "--out", str(out_json)]
    print(f"\n[sweep] ==== img_size={img_size} clips={n_clips} device={device} ====", flush=True)
    print(f"[sweep] $ {' '.join(cmd[len(cmd) - 12:])}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"[sweep] !! img_size={img_size} 失败 (rc={proc.returncode})", flush=True)
        return {"img_size": img_size, "num_clips": n_clips, "device": device,
                "status": "failed", "rc": proc.returncode, "outer_wall_s": round(wall, 1),
                "batch_size": batch_size, "num_frames": num_frames}
    data = json.loads(out_json.read_text(encoding="utf-8"))
    inner_wall = float(data.get("wall_s", wall))
    rec = {
        "img_size": img_size,
        "num_clips": n_clips,
        "device": device,
        "num_frames": num_frames,
        "batch_size": batch_size,
        "status": "ok",
        "inner_wall_s": round(inner_wall, 1),          # 评测内部计时（含模型加载）
        "outer_wall_s": round(wall, 1),                # 进程外总计
        "sec_per_clip": round(inner_wall / max(1, n_clips), 2),
        "clips_per_min": round(60.0 * n_clips / max(1e-9, inner_wall), 2),
        "tasks": data.get("tasks", {}),
        "not_applicable": data.get("not_applicable", {}),
        "skipped": data.get("skipped", {}),
        "grid": data.get("grid"),
        "model": data.get("model", {}),
    }
    print(f"[sweep] img_size={img_size}: {rec['inner_wall_s']}s / {n_clips} 条 "
          f"= {rec['sec_per_clip']}s 每条 ({rec['clips_per_min']} 条/分)", flush=True)
    return rec


def render_html(rows, out_path):
    base = next((r for r in rows if r.get("img_size") == 64 and r.get("status") == "ok"), None)
    trs = []
    for r in rows:
        if r.get("status") != "ok":
            trs.append(f"<tr class='bad'><td>{r['img_size']}px</td><td colspan='5'>失败 rc={r.get('rc')}</td></tr>")
            continue
        sp = r["sec_per_clip"]
        ratio = ""
        if base and base["sec_per_clip"] > 0:
            ratio = f"{sp / base['sec_per_clip']:.1f}×"
        trs.append(
            f"<tr><td><b>{r['img_size']}px</b></td><td>{r['num_clips']}</td>"
            f"<td>{r['inner_wall_s']}s</td><td>{sp}s</td><td>{r['clips_per_min']}</td><td>{ratio}</td></tr>")
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KINE-Bench 分辨率-吞吐扫描</title>
<style>
  :root{{--ink:#0f172a;--mut:#64748b;--bd:#e2e8f0;--acc:#2563eb;--warn:#b45309;}}
  *{{box-sizing:border-box}}
  body{{font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
       margin:0;background:#f1f5f9;color:var(--ink);padding:28px}}
  .wrap{{max-width:900px;margin:0 auto}}
  h1{{font-size:21px;margin:0 0 4px}} .lead{{color:var(--mut);font-size:14px;margin:0 0 18px}}
  table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--bd);
        border-radius:12px;overflow:hidden;font-size:14px}}
  th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid var(--bd)}}
  th{{background:#f8fafc;font-size:12px;color:var(--mut)}}
  tr:last-child td{{border-bottom:none}} tr.bad td{{color:#b91c1c}}
  .note{{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:12px 14px;
        font-size:13px;color:var(--warn);margin-top:16px;line-height:1.7}}
</style></head><body><div class="wrap">
<h1>KINE-Bench 分辨率-吞吐扫描</h1>
<p class="lead">V-JEPA 2 (ViT-L/16, num_frames=16) · CUDA · 合成片段 ·
回答评审质疑「64px 是否降采样省算力」</p>
<table><thead><tr><th>分辨率</th><th>片段数</th><th>墙钟</th><th>每片段</th><th>条/分</th><th>相对 64px</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>
<div class="note"><b>诚实边界：</b>本扫描只测量<b>吞吐与可运行性</b>，不测量模型竞争力。
合成片段不含真实运动结构，表内 TEMP/MOT 分数是噪声（例如 2 条样本下 TEMP 可在 0.0 与 1.0 之间跳变），
CAU-1 的 AUC=1.0 属 degraded 口径（<code>auc_do=null</code>，do-branch 未暴露）。
真实能力数字必须以 <code>--data-dir</code> 喂入真实视频后重跑。</div>
</div></body></html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"[sweep] report -> {out_path} ({out_path.stat().st_size}B)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    ap.add_argument("--clips", default="64:32,128:32,256:16",
                    help="分辨率:片段数 列表，如 64:32,128:32,256:16")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out-json", default="gpu_sweep.json")
    ap.add_argument("--out-html", default="gpu_sweep.html")
    args = ap.parse_args()

    plan = parse_clips(args.clips)
    print(f"[sweep] device={args.device} plan={plan} num_frames={args.num_frames} "
          f"batch_size={args.batch_size}", flush=True)
    rows = []
    for img, n in plan:
        rows.append(run_one(img, n, args.device, args.num_frames, args.batch_size, f"s{img}"))

    payload = {
        "device": args.device,
        "num_frames": args.num_frames,
        "batch_size": args.batch_size,
        "model": next((r.get("model") for r in rows if r.get("model")), None),
        "plan": [{"img_size": i, "num_clips": n} for i, n in plan],
        "rows": rows,
    }
    out_json = ROOT / args.out_json
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[sweep] json -> {out_json}", flush=True)
    render_html(rows, ROOT / args.out_html)

    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\n[sweep] 完成 {len(ok)}/{len(rows)} 档", flush=True)
    for r in ok:
        print(f"[sweep]   {r['img_size']:>3}px  {r['sec_per_clip']:>7}s/条  "
              f"{r['clips_per_min']:>6} 条/分", flush=True)


if __name__ == "__main__":
    main()
