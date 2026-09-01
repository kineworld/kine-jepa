#!/usr/bin/env python
# Implementation inspired by V-JEPA 2-AC (arXiv:2506.09985, MIT); all code original.
"""Interactive counterfactual demo (flagship, self-contained HTML).

Trains nothing -- it precomputes a grid of futures from the REAL
CounterfactualRollout on a small latent space, PCA-projects them to 2D, and
emits a single HTML file with pure-JS interactivity: pick do(x), scrub arm
sliders + horizon, and watch the rolled-out future path + its divergence from
the "no-intervention" baseline redraw live. No server, no GPU, no build step.

This is the artifact you show a judge: "here, imagine the same scene three
different ways by toggling what-if levers."
"""
from __future__ import annotations

import json
import os
import sys
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from kineworld_jepa.counterfactual import CounterfactualRollout

torch.manual_seed(0)

DIM, A, H, LAT_CLIP = 16, 2, 20, 3.0
DO_LABELS = ["无介入（基线）", "撤支撑", "断接触", "随机"]
ARM_GRID = [-1.0, 0.0, 1.0]

# single-token latent so each step is a clean 16-d point for PCA
cf = CounterfactualRollout(dim=DIM, action_dim=A, depth=2, heads=4, latent_clip=LAT_CLIP).eval()
z0 = torch.randn(1, 1, DIM)

all_latents = []

def roll_scenario(do_id, a1, a2):
    arm = torch.tensor([[[a1, a2]] * H], dtype=torch.float32)  # (1,H,2)
    do_ids = torch.full((1, H), do_id, dtype=torch.long)
    futures = cf(z0, arm, do_ids)                              # list[(1,1,DIM)]
    seq = torch.stack([f[0, 0] for f in futures], dim=0)       # (H, DIM)
    return seq

scenarios = {}
for do_id in range(4):
    for a1 in ARM_GRID:
        for a2 in ARM_GRID:
            key = f"d{do_id}_a{a1:+.0f}_{a2:+.0f}"
            seq = roll_scenario(do_id, a1, a2)
            all_latents.append(seq)
            # divergence vs do=0 same arm
            base = scenarios.get(f"d0_a{a1:+.0f}_{a2:+.0f}")
            if base is not None:
                div = float((seq[-1] - base["seq"][-1]).pow(2).mean().sqrt().item())
            else:
                div = 0.0
            scenarios[key] = {"seq": seq, "div": div}

# PCA -> 2D over all timesteps of all scenarios
X = torch.stack(all_latents, dim=0).reshape(-1, DIM)          # (N, DIM)
Xc = X - X.mean(dim=0)
U, S, V = torch.svd(Xc)
comp = V[:, :2]                                                 # (DIM, 2)
proj = (Xc @ comp)                                             # (N, 2)
explained = (S[:2] / S.sum()).tolist()

# reattach 2D coords per scenario
n_per = H
idx = 0
scn_out = {}
base_keys = {}
for do_id in range(4):
    for a1 in ARM_GRID:
        for a2 in ARM_GRID:
            key = f"d{do_id}_a{a1:+.0f}_{a2:+.0f}"
            pts = proj[idx:idx + n_per].tolist()
            idx += n_per
            div = scenarios[key]["div"]
            scn_out[key] = {"path": pts, "div": round(div, 4)}
            if do_id == 0:
                base_keys[(a1, a2)] = key

payload = {
    "do_labels": DO_LABELS,
    "H": H,
    "arm_grid": ARM_GRID,
    "pca_explained": [round(float(e), 3) for e in explained],
    "scenarios": scn_out,
}

# ---- HTML (inline JSON + pure JS) ------------------------------------------
html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KineOne-WM · 可交互反事实推演</title>
<style>
  :root {{ --ink:#0f172a; --mut:#64748b; --bd:#e2e8f0; --bg:#fff; --acc:#2563eb; --ok:#16a34a; --bad:#94a3b8; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
         margin:0; background:#f1f5f9; color:var(--ink); padding:24px; }}
  .wrap {{ max-width:920px; margin:0 auto; }}
  h1 {{ font-size:21px; margin:0 0 4px; }}
  .lead {{ color:var(--mut); font-size:13.5px; margin:0 0 16px; line-height:1.6; }}
  .grid {{ display:grid; grid-template-columns:1fr 260px; gap:16px; }}
  .plot {{ background:var(--bg); border:1px solid var(--bd); border-radius:14px; padding:10px; }}
  .panel {{ background:var(--bg); border:1px solid var(--bd); border-radius:14px; padding:16px; }}
  .do-btns {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:14px; }}
  .do-btns button {{ padding:9px 8px; border:1px solid var(--bd); background:#f8fafc; border-radius:9px;
                    font-size:13px; cursor:pointer; color:var(--ink); }}
  .do-btns button.on {{ background:var(--acc); color:#fff; border-color:var(--acc); font-weight:700; }}
  .ctl {{ margin:12px 0; }}
  .ctl label {{ font-size:12.5px; color:var(--mut); display:flex; justify-content:space-between; }}
  input[type=range] {{ width:100%; }}
  .stat {{ font-size:13px; color:var(--mut); margin-top:10px; line-height:1.7; }}
  .stat b {{ color:var(--ink); font-size:15px; }}
  .legend {{ font-size:11.5px; color:var(--mut); margin-top:8px; }}
  footer {{ color:var(--mut); font-size:12px; text-align:center; padding:16px; }}
</style></head><body><div class="wrap">
  <h1>KineOne-WM · 可交互反事实推演</h1>
  <p class="lead">同一初始场景、同一手臂指令，只切换 do(x) 反事实杠杆，未来轨迹实时重绘。
  这是白泽类纯编码器做不到的——<b>想象替代未来</b>。以下轨迹由真实 CounterfactualRollout 在 16-d latent 空间预计算、PCA 投影到 2D。</p>
  <div class="grid">
    <div class="plot"><svg id="svg" viewBox="0 0 600 460" width="100%" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="legend">横/纵轴 = latent 主成分（解释率 {payload['pca_explained'][0]*100:.0f}% / {payload['pca_explained'][1]*100:.0f}%）。实线=所选 do(x)，虚线=同臂下其余 do(x) 作参照。</div>
    </div>
    <div class="panel">
      <div class="do-btns" id="dobtns"></div>
      <div class="ctl"><label>手臂指令 · 维度1 <span id="a1v">0.0</span></label>
        <input type="range" id="a1" min="-1" max="1" step="0.5" value="0"></div>
      <div class="ctl"><label>手臂指令 · 维度2 <span id="a2v">0.0</span></label>
        <input type="range" id="a2" min="-1" max="1" step="0.5" value="0"></div>
      <div class="ctl"><label>推演步数（horizon） <span id="hv">{H}</span></label>
        <input type="range" id="h" min="1" max="{H}" step="1" value="{H}"></div>
      <div class="stat">所选 do(x)：<b id="doname">{DO_LABELS[0]}</b><br>
        与「无介入」终态分歧：<b id="div">0.0000</b><br>
        <span id="hint" style="color:var(--ok)">切换 do(x) 看轨迹如何分叉</span></div>
    </div>
  </div>
  <footer>由 counterfactual_interactive.py 生成 · 真实 CounterfactualRollout 预计算 · 全离线可交互</footer>
</div>
<script>
const DATA = {json.dumps(payload, ensure_ascii=False)};
let doSel = 0, a1 = 0, a2 = 0, H = DATA.H;
const svg = document.getElementById('svg');
const W=600, Hh=460, pad=30;
function toXY(p){{ return [(p[0]*70+W/2), (Hh/2 - p[1]*70)]; }}
function line(pts, upto, stroke, w, dash){{ if(pts.length===0) return '';
  const seg = pts.slice(0, upto); if(seg.length===0) return '';
  let d = seg.map((p,i)=>(i?'L':'M')+toXY(p).join(',')).join(' ');
  return `<path d="${{d}}" fill="none" stroke="${{stroke}}" stroke-width="${{w}}" ${{dash?`stroke-dasharray="5,4"`:''}}/>`; }}
function dot(p,c){{ const [x,y]=toXY(p); return `<circle cx="${{x}}" cy="${{y}}" r="4.5" fill="${{c}}"/>`; }}
function snap(v){{ return Math.abs(v-1)<0.25?1:Math.abs(v+1)<0.25?-1:0; }}
function akey(a1,a2){{ return `a${{a1>=0?'+':''}}${{a1}}_${{a2>=0?'+':''}}${{a2}}`; }}
function render(){{ svg.innerHTML='';
  // reference do(x) at same arm (dashed)
  for(let d=0; d<4; d++){{ if(d===doSel) continue;
    const k=`d${{d}}_${{akey(a1,a2)}}`; const sc=DATA.scenarios[k];
    svg.innerHTML += line(sc.path, H, '#cbd5e1', 1.5, true); }}
  // selected do(x) (solid)
  const k=`d${{doSel}}_${{akey(a1,a2)}}`; const sc=DATA.scenarios[k];
  svg.innerHTML += line(sc.path, H, '#2563eb', 3, false);
  svg.innerHTML += dot(sc.path[H-1], '#2563eb');
  svg.innerHTML += dot(sc.path[0], '#0f172a');
  document.getElementById('doname').textContent = DATA.do_labels[doSel];
  document.getElementById('div').textContent = sc.div.toFixed(4);
}}
function buildBtns(){{ const box=document.getElementById('dobtns');
  DATA.do_labels.forEach((l,i)=>{{ const b=document.createElement('button');
    b.textContent=l; if(i===0)b.className='on'; b.onclick=()=>{{ doSel=i;
      [...box.children].forEach(x=>x.className=''); b.className='on'; render(); }};
    box.appendChild(b); }}); }}
document.getElementById('a1').oninput=e=>{{ a1=snap(+e.target.value); e.target.value=a1;
  document.getElementById('a1v').textContent=a1.toFixed(1); render(); }};
document.getElementById('a2').oninput=e=>{{ a2=snap(+e.target.value); e.target.value=a2;
  document.getElementById('a2v').textContent=a2.toFixed(1); render(); }};
document.getElementById('h').oninput=e=>{{ H=+e.target.value;
  document.getElementById('hv').textContent=H; render(); }};
buildBtns(); render();
</script></body></html>"""

out = os.path.join(ROOT, "counterfactual_interactive.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"[demo] wrote {out} ({os.path.getsize(out)} bytes) | scenarios={len(scn_out)} "
      f"pca_explained={payload['pca_explained']}")
