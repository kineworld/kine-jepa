#!/usr/bin/env python
# Implementation inspired by V-JEPA 2-AC (arXiv:2506.09985, MIT); all code original.
"""Counterfactual inference demo -> single-file HTML dashboard.

Runs CounterfactualRollout on *synthetic* latents (no GPU, no real trajectory
data, no weights) and renders a viewable proof that KineOne-WM imagines
alternatives, not just the default future:

  1) 2D PCA scatter of the default future vs three counterfactual futures
     (different discrete do(x) interventions) -- the trajectories visibly
     diverge.
  2) Per-step divergence curve (L2 gap between default and each counterfactual
     latent, mean over tokens).

This is a *visualisation of the latent what-if engine*; swap in real V-JEPA 2
features + trained weights (private repo) to make it physical.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kineworld_jepa.counterfactual import CounterfactualRollout

torch.manual_seed(0)

DIM, V, H, A = 64, 16, 24, 6
model = CounterfactualRollout(dim=DIM, action_dim=A, depth=3, heads=4, latent_clip=8.0)
model.eval()  # deterministic (dropout off) so the demo is reproducible

# --- synthetic scene + arm command -------------------------------------------
z0 = torch.randn(1, V, DIM)
arm = torch.randn(1, H, A).cumsum(1) * 0.15  # smooth-ish motor command

INTERVENTIONS = {
    0: ("无介入 (no-op)", "#2563eb"),
    1: ("撤支撑 (remove_support)", "#dc2626"),
    2: ("断接触 (break_contact)", "#ea580c"),
    3: ("随机 (neg-control)", "#16a34a"),
}


def run(do_id: int) -> list:
    do = torch.full((1, H), do_id, dtype=torch.long)
    return model(z0, arm, do)  # list[(1, V, D)]


def to_points(futures: list) -> torch.Tensor:
    # token-mean -> one (D,) point per timestep -> (H, D)
    return torch.stack([f.mean(dim=1).squeeze(0) for f in futures], dim=0)


base = run(0)
base_pts = to_points(base)
alt_pts = {i: to_points(run(i)) for i in (1, 2, 3)}

final_div = {i: (base_pts - p).pow(2).mean().sqrt().item() for i, p in alt_pts.items()}
per_step = {i: [(base_pts[t] - p[t]).pow(2).mean().sqrt().item() for t in range(H)]
            for i, p in alt_pts.items()}

# --- PCA to 2D over all trajectory points ------------------------------------
all_pts = torch.cat([base_pts] + [alt_pts[i] for i in (1, 2, 3)], dim=0)  # (4H, D)
mean = all_pts.mean(0, keepdim=True)
centered = all_pts - mean
cov = centered.t() @ centered / (centered.shape[0] - 1)
eigval, eigvec = torch.linalg.eigh(cov)
top2 = eigval.argsort(descending=True)[:2]
proj = centered @ eigvec[:, top2]  # (4H, 2)

base_proj = proj[:H]
alt_proj = {i: proj[H * k: H * (k + 1)] for k, i in enumerate((1, 2, 3))}
explained = eigval[top2].sum().item() / eigval.sum().item()


# --- SVG helpers --------------------------------------------------------------
def _scale(vals, lo, hi, size, pad):
    vmin, vmax = vals.min().item(), vals.max().item()
    span = (vmax - vmin) or 1.0
    return [lo + pad + (v - vmin) / span * (size - 2 * pad) for v in vals]


def trajectory_svg() -> str:
    W, Hh, pad = 720, 420, 46
    xs = _scale(proj[:, 0], 0, W, W, pad)
    ys = _scale(proj[:, 1], 0, Hh, Hh, pad)
    # invert y for screen coords
    ys = [Hh - y for y in ys]
    cuts = {"base": (0, H)}
    cuts.update({f"a{k}": (H * idx, H * (idx + 1)) for idx, k in enumerate((1, 2, 3))})

    def poly(slc, color):
        pts = " ".join(f"{xs[j]:.1f},{ys[j]:.1f}" for j in range(*slc))
        return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')

    lines = []
    # faint base first, then alts on top
    lines.append(poly(cuts["base"], INTERVENTIONS[0][1]))
    for k in (1, 2, 3):
        lines.append(poly(cuts[f"a{k}"], INTERVENTIONS[k][1]))
    # start markers
    markers = []
    for key, (_, color) in [("base", INTERVENTIONS[0])] + [(f"a{k}", INTERVENTIONS[k]) for k in (1, 2, 3)]:
        j = cuts[key][0]
        markers.append(f'<circle cx="{xs[j]:.1f}" cy="{ys[j]:.1f}" r="5" fill="{color}"/>')
    legend = "".join(
        f'<div class="lg"><span class="sw" style="background:{c}"></span>{name}</div>'
        for name, c in [INTERVENTIONS[0]] + [INTERVENTIONS[k] for k in (1, 2, 3)])
    return f"""
    <div class="card">
      <h3>轨迹 PCA 投影（默认未来 vs 反事实未来）</h3>
      <p class="sub">每条折线 = 24 步 latent 轨迹（token 均值），按前 2 主成分投影；
         前 2 主成分解释方差 {explained*100:.1f}%。起点 ● 重合，随后因 do(x) 不同而分叉。</p>
      <svg viewBox="0 0 {W} {Hh}" width="100%" preserveAspectRatio="xMidYMid meet">
        <rect x="0" y="0" width="{W}" height="{Hh}" fill="#f8fafc" rx="10"/>
        <line x1="{pad}" y1="{Hh//2}" x2="{W-pad}" y2="{Hh//2}" stroke="#e2e8f0"/>
        <line x1="{W//2}" y1="{pad}" x2="{W//2}" y2="{Hh-pad}" stroke="#e2e8f0"/>
        {''.join(lines)}
        {''.join(markers)}
      </svg>
      <div class="legend">{legend}</div>
    </div>"""


def divergence_svg() -> str:
    W, Hh, pad = 720, 320, 46
    maxv = max(max(v) for v in per_step.values()) * 1.1 or 1.0
    xs = [pad + t / (H - 1) * (W - 2 * pad) for t in range(H)]

    def poly(vals, color):
        pts = " ".join(f"{xs[t]:.1f},{Hh-pad - vals[t]/maxv*(Hh-2*pad):.1f}" for t in range(H))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>'

    grid = "".join(
        f'<line x1="{pad}" y1="{Hh-pad - i/4*(Hh-2*pad)}" x2="{W-pad}" y2="{Hh-pad - i/4*(Hh-2*pad)}" stroke="#eef2f7"/>'
        for i in range(5))
    lines = [poly(per_step[k], INTERVENTIONS[k][1]) for k in (1, 2, 3)]
    legend = "".join(
        f'<div class="lg"><span class="sw" style="background:{c}"></span>{name}</div>'
        for name, c in [INTERVENTIONS[k] for k in (1, 2, 3)])
    return f"""
    <div class="card">
      <h3>逐步分歧曲线（默认 vs 反事实，token 均值 L2）</h3>
      <p class="sub">横轴=步数，纵轴=默认未来与对应反事实未来 latent 的 L2 距离。
         分歧随时间累积，证明世界模型在"想象"不同结局。</p>
      <svg viewBox="0 0 {W} {Hh}" width="100%" preserveAspectRatio="xMidYMid meet">
        <rect x="0" y="0" width="{W}" height="{Hh}" fill="#f8fafc" rx="10"/>
        {grid}
        {''.join(lines)}
      </svg>
      <div class="legend">{legend}</div>
    </div>"""


def metric_cards() -> str:
    cards = [
        ("latent 维度", f"{DIM}"),
        ("推演步数 H", f"{H}"),
        ("撤支撑 终态分歧", f"{final_div[1]:.3f}"),
        ("断接触 终态分歧", f"{final_div[2]:.3f}"),
        ("随机 终态分歧", f"{final_div[3]:.3f}"),
        ("前2主成分解释率", f"{explained*100:.1f}%"),
    ]
    return '<div class="metrics">' + "".join(
        f'<div class="metric"><div class="m-v">{v}</div><div class="m-k">{k}</div></div>'
        for k, v in cards) + '</div>'


html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KineOne-WM · 反事实推演 Demo</title>
<style>
  :root {{ --ink:#0f172a; --mut:#64748b; --bd:#e2e8f0; --bg:#ffffff; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
         margin:0; background:#f1f5f9; color:var(--ink); padding:28px; }}
  .wrap {{ max-width:960px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .lead {{ color:var(--mut); margin:0 0 20px; font-size:14px; }}
  .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin-bottom:20px; }}
  .metric {{ background:var(--bg); border:1px solid var(--bd); border-radius:12px; padding:14px 16px; }}
  .m-v {{ font-size:22px; font-weight:700; }}
  .m-k {{ font-size:12px; color:var(--mut); margin-top:4px; }}
  .card {{ background:var(--bg); border:1px solid var(--bd); border-radius:14px; padding:18px 20px; margin-bottom:18px; }}
  .card h3 {{ margin:0 0 6px; font-size:16px; }}
  .sub {{ color:var(--mut); font-size:13px; margin:0 0 12px; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:10px; font-size:13px; color:var(--mut); }}
  .lg {{ display:flex; align-items:center; gap:6px; }}
  .sw {{ width:14px; height:14px; border-radius:4px; display:inline-block; }}
  .note {{ font-size:13px; color:var(--mut); line-height:1.6; }}
  code {{ background:#f1f5f9; padding:1px 6px; border-radius:5px; font-size:12px; }}
</style></head>
<body><div class="wrap">
  <h1>KineOne-WM · 反事实推演 Demo</h1>
  <p class="lead">动作条件化世界模型（CounterfactualRollout）在合成 latent 上的 what-if 可视化 ·
      同一初始场景 + 同一连续臂指令，仅切换离散 do(x) 介入，未来即分叉。</p>
  {metric_cards()}
  {trajectory_svg()}
  {divergence_svg()}
  <div class="card">
    <h3>这证明了什么</h3>
    <p class="note">传统编码器/评测（如白泽式基准）只能<strong>描述</strong>已发生的画面；
        KineOne-WM 在此之上叠加了<strong>反事实推演</strong>：给定相同前提，它能想象"如果当时撤掉支撑 /
        断开接触，场景会怎样演化"，并量化两种结局的差距。<br>
        生产形态：把合成 latent 换成 V-JEPA 2 真实特征 + 私有训练权重（权重/动作标注/后训练配方闭源），
        即可对真实视频做物理反事实问答。本页仅用合成 latent 演示<strong>引擎本身</strong>。</p>
  </div>
</div></body></html>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counterfactual_demo.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)

print(f"wrote {out}")
print(f"final_div: " + ", ".join(f"do{i}={final_div[i]:.4f}" for i in (1, 2, 3)))
print(f"pca explained(var)={explained*100:.1f}%")
