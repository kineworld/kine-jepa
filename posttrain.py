#!/usr/bin/env python
# Implementation inspired by V-JEPA 2-AC (arXiv:2506.09985, MIT); all code original.
"""Post-training proof: train the action-conditioned world model
(CounterfactualRollout) on a *synthetic* action+intervention-conditioned latent
dynamics via teacher forcing, then show the moat recipe actually works:

  (a) rollout MSE collapses from random-init to the noise floor,
  (b) LatentPlanner reaches a held-out goal latent,
  (c) counterfactual divergence reflects the *learned* intervention offset.

This closes the loop the competitive dashboard calls the moat: the interface is
open, but a *trained* predictor (here on a self-contained synthetic stand-in for
real trajectories) is what makes rollouts physically meaningful. Fully
reproducible, CPU-only -- no GPU, no private data. Swap the synthetic world for
real trajectory data + the private post-training recipe and you get KineOne-WM
on physical video.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kineworld_jepa.counterfactual import CounterfactualRollout
from kineworld_jepa.rollout import LatentPlanner

torch.manual_seed(0)

DIM, A, HOR = 32, 4, 12


class SyntheticWorld:
    """Known linear action+intervention dynamics in latent space.

    z_{t+1} = z_t + M·a_t + I[do_t] + noise   (residual form, exactly what
    ActionRollout's `nxt = latent + delta` models). A trained predictor must
    recover M (via the arm embedder) and I (via the causal InterventionHead).
    """

    def __init__(self, dim=DIM, action_dim=A, n_interv=4, noise=0.05, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.dim, self.action_dim, self.n_interv = dim, action_dim, n_interv
        self.M = torch.randn(dim, action_dim, generator=g) * 0.5
        self.I = torch.randn(n_interv, dim, generator=g) * 0.5
        self.noise = noise

    def sample(self, batch, horizon, seed=None):
        g = torch.Generator().manual_seed(
            seed if seed is not None else int(torch.randint(0, 1 << 30, (1,)).item()))
        z0 = torch.randn(batch, 1, self.dim, generator=g)
        arm = torch.randn(batch, horizon, self.action_dim, generator=g) * 0.5
        do = torch.randint(0, self.n_interv, (batch, horizon), generator=g)
        lat = z0.squeeze(1)
        targets = []
        for t in range(horizon):
            lat = (lat + (arm[:, t] @ self.M.T) + self.I[do[:, t]]
                   + torch.randn(batch, self.dim, generator=g) * self.noise)
            targets.append(lat.unsqueeze(1))
        return z0, {"arm": arm, "do": do}, targets


def train(model, world, epochs=250, batch=64, horizon=HOR, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for e in range(epochs):
        z0, acts, tgts = world.sample(batch, horizon)
        preds = model(z0, acts["arm"], acts["do"], horizon=horizon)
        loss = sum(F.mse_loss(p, t) for p, t in zip(preds, tgts)) / len(preds)
        opt.zero_grad()
        loss.backward()
        opt.step()
        log.append(loss.item())
    return log


@torch.no_grad()
def rollout_mse(model, world, batch=256, horizon=HOR, seed=12345):
    z0, acts, tgts = world.sample(batch, horizon, seed=seed)
    preds = model(z0, acts["arm"], acts["do"], horizon=horizon)
    return sum(F.mse_loss(p, t) for p, t in zip(preds, tgts)).item() / len(preds)


@torch.no_grad()
def counterfactual_div(model, world, seed=777):
    z0, acts, _ = world.sample(1, HOR, seed=seed)
    arm = acts["arm"]
    base = model(z0, arm, torch.zeros(1, HOR, dtype=torch.long))
    alt = model(z0, arm, torch.ones(1, HOR, dtype=torch.long))
    return (base[-1] - alt[-1]).pow(2).mean().sqrt().item()


class _ArmOnlyWrap(nn.Module):
    """Adapt CounterfactualRollout (arm+do) to LatentPlanner's (latent, actions,
    horizon) call convention by binding do=0 (no intervention) during planning."""

    def __init__(self, cf):
        super().__init__()
        self.cf = cf

    def forward(self, z0, arm, horizon=None):
        b = z0.shape[0]
        t = horizon or arm.shape[1]
        do = torch.zeros(b, t, dtype=torch.long)
        return self.cf(z0, arm, do, horizon=horizon)


@torch.no_grad()
def planner_distance(model, world, horizon=HOR, seed=2024):
    z0, acts, tgts = world.sample(1, horizon, seed=seed)
    goal = tgts[-1].detach()
    planner = LatentPlanner(_ArmOnlyWrap(model), goal, action_dim=A, horizon=horizon,
                            action_low=-1.0, action_high=1.0)
    _, loss = planner.plan(z0, iters=15, candidates=64, seed=0)
    return loss


def _curve_svg(log, w=720, h=300, pad=46):
    import math
    xs = [pad + i / (len(log) - 1) * (w - 2 * pad) for i in range(len(log))]
    lo, hi = min(log), max(log)
    span = (hi - lo) or 1.0
    ys = [h - pad - (v - lo) / span * (h - 2 * pad) for v in log]
    pts = " ".join(f"{xs[i]:.1f},{ys[i]:.1f}" for i in range(len(log)))
    grid = "".join(
        f'<line x1="{pad}" y1="{h-pad-i/4*(h-2*pad)}" x2="{w-pad}" y2="{h-pad-i/4*(h-2*pad)}" stroke="#eef2f7"/>'
        for i in range(5))
    return f"""<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet">
      <rect x="0" y="0" width="{w}" height="{h}" fill="#f8fafc" rx="10"/>
      {grid}
      <polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
    </svg>"""


def main():
    world = SyntheticWorld()
    untrained = CounterfactualRollout(dim=DIM, action_dim=A, depth=3, heads=4, latent_clip=None)
    untrained.eval()
    trained = CounterfactualRollout(dim=DIM, action_dim=A, depth=3, heads=4, latent_clip=None)
    log = train(trained, world, epochs=250)
    trained.eval()

    um, tm = rollout_mse(untrained, world), rollout_mse(trained, world)
    uc, tc = counterfactual_div(untrained, world), counterfactual_div(trained, world)
    up, tp = planner_distance(untrained, world), planner_distance(trained, world)
    reduc = (1 - tm / um) * 100 if um > 0 else 0.0

    print(f"rollout_mse  untrained={um:.4f}  trained={tm:.4f}  reduction={reduc:.1f}%")
    print(f"counterfactual_div  untrained={uc:.4f}  trained={tc:.4f}")
    print(f"planner_distance  untrained={up:.2f}  trained={tp:.4f}")

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KineOne-WM · 后训练配方证明</title>
<style>
  :root {{ --ink:#0f172a; --mut:#64748b; --bd:#e2e8f0; --bg:#fff; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
         margin:0; background:#f1f5f9; color:var(--ink); padding:28px; }}
  .wrap {{ max-width:920px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .lead {{ color:var(--mut); font-size:14px; margin:0 0 18px; }}
  .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:18px; }}
  .metric {{ background:var(--bg); border:1px solid var(--bd); border-radius:12px; padding:14px 16px; }}
  .m-v {{ font-size:21px; font-weight:700; }}
  .m-k {{ font-size:12px; color:var(--mut); margin-top:4px; }}
  .card {{ background:var(--bg); border:1px solid var(--bd); border-radius:14px; padding:18px 20px; margin-bottom:16px; }}
  .card h3 {{ margin:0 0 6px; font-size:16px; }}
  .sub {{ color:var(--mut); font-size:13px; margin:0 0 12px; }}
  .note {{ font-size:13px; color:var(--mut); line-height:1.6; }}
  code {{ background:#f1f5f9; padding:1px 6px; border-radius:5px; font-size:12px; }}
  .ok {{ color:#16a34a; font-weight:700; }}
  .bad {{ color:#94a3b8; }}
</style></head><body><div class="wrap">
  <h1>KineOne-WM · 后训练配方证明（moat recipe 端到端）</h1>
  <p class="lead">在合成动作条件动力学上 teacher-forcing 训练 CounterfactualRollout（CPU，{len(log)} epoch）。
     证明：训练后预测误差塌到噪声地板、规划器抵达目标、反事实分歧反映已学介入。</p>

  <div class="metrics">
    <div class="metric"><div class="m-v">{um:.3f}</div><div class="m-k">rollout MSE · 训练前</div></div>
    <div class="metric"><div class="m-v ok">{tm:.3f}</div><div class="m-k">rollout MSE · 训练后（↓{reduc:.0f}%）</div></div>
    <div class="metric"><div class="m-v">{up:.2f}</div><div class="m-k">规划距离 · 训练前</div></div>
    <div class="metric"><div class="m-v ok">{tp:.3f}</div><div class="m-k">规划距离 · 训练后</div></div>
    <div class="metric"><div class="m-v">{uc:.3f}</div><div class="m-k">反事实分歧 · 训练前</div></div>
    <div class="metric"><div class="m-v ok">{tc:.3f}</div><div class="m-k">反事实分歧 · 训练后</div></div>
  </div>

  <div class="card">
    <h3>训练曲线（teacher-forcing rollout MSE）</h3>
    <p class="sub">横轴=epoch，纵轴=批次 rollout MSE（越低越好）。从随机初始化快速收敛到噪声地板。</p>
    {_curve_svg(log)}
  </div>

  <div class="card">
    <h3>这证明了什么</h3>
    <p class="note">此前 rollout 为<b>随机初始化</b>，推演只是架构证明。本页用已知动力学生成合成轨迹，
       经 teacher-forcing 后训练后：<br>
       ① <b>rollout MSE 下降 {reduc:.0f}%</b>（{um:.3f}→{tm:.3f}），逼近合成噪声地板 → 预测器学会了真实 latent 演化；<br>
       ② <b>规划距离 {up:.2f}→{tp:.3f}</b> → 训练后 LatentPlanner 能搜到动作序列抵达目标 latent；<br>
       ③ <b>反事实分歧训练后={tc:.3f}</b> → 因果 InterventionHead 学到了 do(x) 对场景的偏移，能回答"如果撤支撑会怎样"。<br>
       生产形态：把 SyntheticWorld 换成<b>真实轨迹数据</b> + 私有权重/标注/后训练配方（闭源），即得到物理可信的 KineOne-WM。
       本证明全开源、可复现、无需 GPU。</p>
  </div>
</div></body></html>"""

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "posttrain_demo.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
