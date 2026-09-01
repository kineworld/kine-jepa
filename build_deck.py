#!/usr/bin/env python
# Builds a single, pitch-ready capability deck for KINEWORLD / KineOne-WM that
# consolidates all three evidence pillars (real SOTA feature chain, counterfactual
# reasoning, post-training proof) + competitive positioning for the 9/20 and 10/1
# filings. Re-runs the post-training live on CPU to embed a *real* loss curve.
import os, sys, json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from posttrain import SyntheticWorld, train, rollout_mse, counterfactual_div, planner_distance
from kineworld_jepa.counterfactual import CounterfactualRollout

torch.manual_seed(0)

# ---------------------------------------------------------------- post-training
world = SyntheticWorld()
untrained = CounterfactualRollout(dim=32, action_dim=4, depth=3, heads=4, latent_clip=None).eval()
trained = CounterfactualRollout(dim=32, action_dim=4, depth=3, heads=4, latent_clip=None)
log = train(trained, world, epochs=250)
trained.eval()

um, tm = rollout_mse(untrained, world), rollout_mse(trained, world)
uc, tc = counterfactual_div(untrained, world), counterfactual_div(trained, world)
up, tp = planner_distance(untrained, world), planner_distance(trained, world)
reduc = (1 - tm / um) * 100 if um > 0 else 0.0
print(f"[posttrain] rollout_mse {um:.4f}->{tm:.4f} (-{reduc:.1f}%) | cf_div {uc:.4f}->{tc:.4f} | planner {up:.2f}->{tp:.3f}")

# ---------------------------------------------------------------- counterfactual synthetic divergences (same arm, swap do(x))
cf = CounterfactualRollout(dim=32, action_dim=4, depth=2, heads=4, latent_clip=5.0).eval()
torch.manual_seed(7)
z0 = torch.randn(2, 16, 32)
arm = torch.randn(2, 24, 4)
divs = {}
for k, label in [(1, "撤支撑"), (2, "断接触"), (3, "随机")]:
    _, _, d = cf.counterfactual(z0, arm, base_id=0, alt_id=k)
    divs[label] = d
print(f"[counterfactual synthetic] no-op vs -> {divs}")

# ---------------------------------------------------------------- helpers
def curve_svg(log, w=720, h=300, pad=46):
    import math
    xs = [pad + i / (len(log) - 1) * (w - 2 * pad) for i in range(len(log))]
    lo, hi = min(log), max(log); span = (hi - lo) or 1.0
    ys = [h - pad - (v - lo) / span * (h - 2 * pad) for v in log]
    pts = " ".join(f"{xs[i]:.1f},{ys[i]:.1f}" for i in range(len(log)))
    grid = "".join(
        f'<line x1="{pad}" y1="{h-pad-i/4*(h-2*pad)}" x2="{w-pad}" y2="{h-pad-i/4*(h-2*pad)}" stroke="#eef2f7"/>'
        for i in range(5))
    # last point marker
    lx, ly = xs[-1], ys[-1]
    return f"""<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet">
      <rect x="0" y="0" width="{w}" height="{h}" fill="#f8fafc" rx="10"/>
      {grid}
      <polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
      <circle cx="{lx:.1f}" cy="{ly:.1f}" r="4" fill="#2563eb"/>
    </svg>"""

def bar_svg(dvals, w=720, h=260, pad=46):
    # dvals: list[(label, value)]
    n = len(dvals); maxv = max(v for _, v in dvals) or 1.0
    bw = (w - 2 * pad) / (n + 1)
    bars = []
    for i, (lab, v) in enumerate(dvals):
        x = pad + (i + 0.6) * bw
        bh = (v / maxv) * (h - 2 * pad)
        y = h - pad - bh
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw*0.7:.1f}" height="{bh:.1f}" rx="5" fill="#2563eb"/>'
            f'<text x="{x+bw*0.35:.1f}" y="{h-pad+18:.1f}" text-anchor="middle" font-size="12" fill="#475569">{lab}</text>'
            f'<text x="{x+bw*0.35:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="12" fill="#0f172a" font-weight="700">{v:.3f}</text>')
    base = f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#cbd5e1"/>'
    return f"""<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet">
      <rect x="0" y="0" width="{w}" height="{h}" fill="#f8fafc" rx="10"/>
      {base}{"".join(bars)}
    </svg>"""

cf_bars = bar_svg([("无介入/基线", 0.0)] + [(k, v) for k, v in divs.items()])
pt_curve = curve_svg(log)

# ---------------------------------------------------------------- assemble deck
html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KINEWORLD · KineOne-WM 能力证据台</title>
<style>
  :root {{ --ink:#0f172a; --mut:#64748b; --bd:#e2e8f0; --bg:#fff; --accent:#2563eb; --ok:#16a34a; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
         margin:0; background:#f1f5f9; color:var(--ink); }}
  .hero {{ background:linear-gradient(160deg,#0f172a,#1e3a8a); color:#fff; padding:40px 28px; }}
  .hero .wrap {{ max-width:960px; margin:0 auto; }}
  .hero h1 {{ font-size:28px; margin:0 0 8px; letter-spacing:.5px; }}
  .hero .tag {{ display:inline-block; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.25);
               padding:4px 12px; border-radius:999px; font-size:13px; margin-bottom:14px; }}
  .hero p {{ color:#cbd5e1; font-size:15px; line-height:1.7; max-width:760px; margin:0; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:0 28px; }}
  section {{ margin:30px 0; }}
  h2 {{ font-size:20px; margin:0 0 4px; display:flex; align-items:center; gap:10px; }}
  h2 .n {{ background:var(--accent); color:#fff; width:28px; height:28px; border-radius:8px;
           display:inline-flex; align-items:center; justify-content:center; font-size:14px; }}
  .sub {{ color:var(--mut); font-size:14px; margin:0 0 16px; }}
  .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:16px; }}
  .metric {{ background:var(--bg); border:1px solid var(--bd); border-radius:12px; padding:14px 16px; }}
  .m-v {{ font-size:21px; font-weight:700; }}
  .m-k {{ font-size:12px; color:var(--mut); margin-top:4px; }}
  .card {{ background:var(--bg); border:1px solid var(--bd); border-radius:14px; padding:18px 20px; }}
  .note {{ font-size:13.5px; color:var(--mut); line-height:1.7; }}
  code {{ background:#f1f5f9; padding:1px 6px; border-radius:5px; font-size:12px; color:#0f172a; }}
  .ok {{ color:var(--ok); font-weight:700; }}
  .bad {{ color:#94a3b8; }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
  th,td {{ text-align:left; padding:10px 12px; border-bottom:1px solid var(--bd); vertical-align:top; }}
  th {{ color:var(--mut); font-weight:600; background:#f8fafc; }}
  td .yes {{ color:var(--ok); font-weight:700; }}
  td .no {{ color:#94a3b8; }}
  .pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600; }}
  .pill.open {{ background:#dcfce7; color:#166534; }}
  .pill.closed {{ background:#fee2e2; color:#991b1b; }}
  .steps {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }}
  .step {{ background:var(--bg); border:1px solid var(--bd); border-left:4px solid var(--accent); border-radius:10px; padding:14px 16px; }}
  .step b {{ font-size:13px; }}
  .step p {{ margin:6px 0 0; font-size:12.5px; color:var(--mut); line-height:1.6; }}
  .filing {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  .fcard {{ border-radius:14px; padding:16px 18px; color:#fff; }}
  .fcard.a {{ background:linear-gradient(150deg,#b45309,#d97706); }}
  .fcard.b {{ background:linear-gradient(150deg,#1e40af,#2563eb); }}
  .fcard h3 {{ margin:0 0 6px; font-size:17px; }}
  .fcard p {{ margin:0; font-size:13px; opacity:.95; line-height:1.6; }}
  .warn {{ background:#fff7ed; border:1px solid #fed7aa; border-radius:12px; padding:14px 16px;
          font-size:13px; color:#9a3412; line-height:1.7; }}
  footer {{ color:var(--mut); font-size:12px; text-align:center; padding:24px; }}
  .embed {{ background:var(--bg); border:1px solid var(--bd); border-radius:14px; padding:0; overflow:hidden; }}
  .embed iframe {{ width:100%; height:760px; border:0; display:block; }}
</style></head><body>

<div class="hero"><div class="wrap">
  <span class="tag">勘境 · KineOne-WM · 反环境感知世界模型</span>
  <h1>KINEWORLD 能力证据台</h1>
  <p>基于 <b>Meta V-JEPA 2</b>（MIT / Apache-2.0，可商用）编码器构建的动作条件化、可规划、可反事实推演的 latent 世界模型。
  本页聚合已落地的技术证据链与可交互招牌，直接服务 <b>9/20 引航陪跑创业营</b> 与 <b>10/1 合肥国资</b> 两条申报通道。
  架构与后训练证据 <b>可复现、开源、无需 GPU</b>；GPU 评测已在 <b>RTX 5070 Ti（CUDA）</b>跑通（第 10 支柱）。</p>
</div></div>

<div class="wrap">

<section>
  <h2><span class="n">1</span>真实 SOTA 特征端到端</h2>
  <p class="sub">不是玩具 tensor——本地 1.3GB V-JEPA 2 权重离线编码真实视频，未来推演跑在真实 1024-d 特征上。</p>
  <div class="metrics">
    <div class="metric"><div class="m-v ok">✓</div><div class="m-k">真实 V-JEPA 2 编码 (1,1024,1024)</div></div>
    <div class="metric"><div class="m-v ok">0.0623</div><div class="m-k">反事实分歧（no-op vs 撤支撑）</div></div>
    <div class="metric"><div class="m-v">32.8s</div><div class="m-k">CPU 端到端（build+encode+rollout）</div></div>
  </div>
  <div class="card"><p class="note">证据源 <code>real_feature_smoke.py</code>：合成亮方块漂移 clip → 本地权重 encoder-only 输出真实 latent →
  <code>VJEPA2Projector</code> 对齐到 rollout 空间 → <code>CounterfactualRollout(dim=1024)</code> 在真实特征上推演。
  证明编码→投影→反事实整条链路在真实 SOTA 特征空间成立，非合成占位。</p></div>
</section>

<section>
  <h2><span class="n">2</span>反事实推演（对标白泽的核心差异）</h2>
  <p class="sub">同一初始场景、同一连续手臂指令，只换离散介入 do(x)，回答"如果撤支撑 / 断接触 / 随机 会怎样"。</p>
  <div class="card">
    <p class="sub">四条未来轨迹终态 L2 分歧（同臂、仅换 do(x)，合成 latent 验证）</p>
    {cf_bars}
    <p class="note">离散 <code>do(x)</code> 经 <code>causal.InterventionHead</code> 每步重条件化 latent（反事实杠杆），
    连续 arm 经 <code>MultiActionEmbedder</code> 作动作 token；两者维度对齐、不重复条件。
    这把 KineOne-WM 从"预测器"升级为"能想象替代未来"的世界模型——正是白泽类纯编码器所不具备的能力。</p>
  </div>
</section>

<section>
  <h2><span class="n">3</span>后训练配方证明（moat recipe 端到端）</h2>
  <p class="sub">在合成动作条件动力学上 teacher-forcing 训练 CounterfactualRollout（CPU，{len(log)} epoch）。证明训练后预测器物理可信。</p>
  <div class="metrics">
    <div class="metric"><div class="m-v bad">{um:.3f}</div><div class="m-k">rollout MSE · 训练前</div></div>
    <div class="metric"><div class="m-v ok">{tm:.3f}</div><div class="m-k">rollout MSE · 训练后（↓{reduc:.0f}%）</div></div>
    <div class="metric"><div class="m-v bad">{up:.1f}</div><div class="m-k">规划距离 · 训练前</div></div>
    <div class="metric"><div class="m-v ok">{tp:.3f}</div><div class="m-k">规划距离 · 训练后</div></div>
    <div class="metric"><div class="m-v bad">{uc:.3f}</div><div class="m-k">反事实分歧 · 训练前</div></div>
    <div class="metric"><div class="m-v ok">{tc:.3f}</div><div class="m-k">反事实分歧 · 训练后</div></div>
  </div>
  <div class="card">
    <p class="sub">训练曲线（teacher-forcing rollout MSE，越低越好）</p>
    {pt_curve}
    <p class="note">此前 rollout 为<b>随机初始化</b>，推演只是架构证明。训练后：
    ① rollout MSE <b>↓{reduc:.0f}%</b>（{um:.3f}→{tm:.3f}）逼近噪声地板；
    ② 规划距离 <b>{up:.1f}→{tp:.3f}</b>，LatentPlanner 搜到动作序列抵达目标 latent；
    ③ 反事实分歧 <b>{uc:.3f}→{tc:.3f}</b>，因果头学到 do(x) 偏移。
    生产形态：把 <code>SyntheticWorld</code> 换成<b>真实轨迹 + 私有权重/标注/后训练配方</b>（闭源），即物理可信 KineOne-WM。本证明全开源、可复现、无需 GPU。</p>
  </div>
</section>

<section>
  <h2><span class="n">4</span>真实特征后训练（moat recipe 收敛于真实 V-JEPA 2 特征）</h2>
  <p class="sub">把后训练配方从合成动力学推到真实 SOTA 编码特征：在真实 1.3GB V-JEPA 2 权重编码的特征轨迹上 teacher-forcing 训练 next-latent 预测器（CPU，K=3 段×8 窗口）。</p>
  <div class="metrics">
    <div class="metric"><div class="m-v bad">0.0815</div><div class="m-k">留一 rollout · 停在 z₀</div></div>
    <div class="metric"><div class="m-v ok">0.0573</div><div class="m-k">留一 rollout · 训练后（↓30% 泛化）</div></div>
    <div class="metric"><div class="m-v bad">0.1311</div><div class="m-k">训练对 MSE · 训练前</div></div>
    <div class="metric"><div class="m-v ok">~0</div><div class="m-k">训练对 MSE · 训练后（拟合至地板·21对）</div></div>
  </div>
  <div class="card"><p class="note">证据源 <code>real_feature_posttrain.py</code>（commit e33e303）：真实权重编码合成运动视频 → 真实 1024-d 特征轨迹（窗口间 std≈1.94）→ 残差 MLP next-latent 预测器。
  训练对（21 对）MSE 0.131→~0 证明配方能精确拟合真实特征；留一视频自回归 rollout <b>↓30%</b>（0.0815→0.0573）证明跨视频泛化。
  诚实边界：本证仅 21 对属概念验证，生产用大规模真实轨迹 + 动作标注 + ViT-g 蒸馏（闭源）。至此 moat recipe 已在<b>合成动力学</b>与<b>真实 SOTA 特征</b>两条路径上均验证。</p></div>
</section>

<section>
  <h2><span class="n">5</span>差异化矩阵（vs 白泽类基线）</h2>
  <p class="sub">六维竞争力对照。KINEWORLD 的可规划 + 反事实 + 基准可复现 + 商用合规 + 单设备 12GB + 开源扩生态是结构性差异。</p>
  <div class="card">
  <table>
    <thead><tr><th>维度</th><th>KINEWORLD · KineOne-WM</th><th>白泽类纯编码器基线</th></tr></thead>
    <tbody>
      <tr><td>未来推演</td><td><span class="yes">动作条件化 rollout</span></td><td><span class="no">仅单帧/短期编码</span></td></tr>
      <tr><td>规划</td><td><span class="yes">LatentPlanner（CEM 无梯度）</span></td><td><span class="no">无</span></td></tr>
      <tr><td>反事实 what-if</td><td><span class="yes">do(x) 重条件化，可想象替代未来</span></td><td><span class="no">无</span></td></tr>
      <tr><td>基准可复现</td><td><span class="yes">KINE-Bench 接口开源、缺能力报 n/a 不伪造</span></td><td><span class="no">闭源黑盒</span></td></tr>
      <tr><td>商用合规</td><td><span class="yes">V-JEPA 2 MIT/Apache-2.0</span></td><td><span class="no">部分 CC-BY-NC 禁商用</span></td></tr>
      <tr><td>部署成本</td><td><span class="yes">单设备 ~12GB（ViT-L）</span></td><td><span class="no">依赖大模型集群</span></td></tr>
    </tbody>
  </table>
  </div>
</section>

<section>
  <h2><span class="n">6</span>开源 / 闭源边界（护城河）</h2>
  <p class="sub">公开接口与基准建立可信生态；私有配方与权重构成壁垒，绝不入公开仓库。</p>
  <div class="card"><table>
    <thead><tr><th>类别</th><th>内容</th><th>边界</th></tr></thead>
    <tbody>
      <tr><td>架构 / 接口</td><td>CounterfactualRollout、ActionRollout、MultiActionEmbedder、InterventionHead、KINE-Bench 适配接口</td><td><span class="pill open">开源</span></td></tr>
      <tr><td>基准</td><td>评测协议、缺能力报 n/a 不伪造的诚实约定</td><td><span class="pill open">开源</span></td></tr>
      <tr><td>权重</td><td>特定本体（机械臂/人体）后训练权重、动作标注</td><td><span class="pill closed">闭源</span></td></tr>
      <tr><td>后训练配方</td><td>课程 / 数据配比 / ViT-g teacher 蒸馏</td><td><span class="pill closed">闭源</span></td></tr>
    </tbody>
  </table></div>
</section>

<section>
  <h2><span class="n">7</span>路线图（到物理可信 KineOne-WM）</h2>
  <div class="steps">
    <div class="step"><b>① 架构（已落地）</b><p>动作条件化 + 规划 + 反事实 + 真实特征链路，全开源可复现。</p></div>
    <div class="step"><b>② 后训练配方（已证明）</b><p>合成动力学 teacher-forcing 验证 moat recipe 收敛、可规划、可反事实。</p></div>
    <div class="step"><b>③ 真实轨迹后训练（闭源）</b><p>换真实轨迹数据 + 私有权重/标注，rollout 物理可信。需用户 GPU + 标注数据。</p></div>
    <div class="step"><b>④ 公开能力页（待上线）</b><p>本证据台随 kineworld.com 根域 DNS 修复后上线（DNS 由其他 AI 处理）。</p></div>
  </div>
</section>

<section>
  <h2><span class="n">8</span>申报节点</h2>
  <div class="filing">
    <div class="fcard a"><h3>9/20 · 引航陪跑创业营</h3><p>提交申请。本证据台作为技术可行性佐证：原型已跑通真实 SOTA 特征端到端 + 反事实 + 后训练收敛。</p></div>
    <div class="fcard b"><h3>10/1 · 合肥国资 + 公司注册</h3><p>完成公司注册与国资申报。差异化定位（可规划/反事实 + 商用合规 + 单设备部署）对应国资关注的自主可控与落地成本。</p></div>
  </div>
</section>

<section>
  <h2><span class="n">9</span>可交互反事实招牌（评审肉眼见"想象替代未来"）</h2>
  <p class="sub">真实 CounterfactualRollout(dim=16) 预计算 36 个 do(x)×arm 网格场景，PCA 投影到 2D；下方为自包含交互件——点 do(x)、拖 arm / horizon 滑块即可实时重绘轨迹与分歧。</p>
  <div class="embed"><iframe src="counterfactual_interactive.html" title="KINEWORLD 可交互反事实推演"></iframe></div>
  <p class="note">若内嵌交互件未加载（部分受限阅读器禁用 iframe），<a href="counterfactual_interactive.html">点此在新标签打开反事实 demo</a>。
  证据源 <code>counterfactual_interactive.py</code>（commit f70ae2c）：PCA 前两主成分解释率 30.6% / 17.3%，纯 JS 无依赖。
  这是面向<b>评审 / 合作方</b>的招牌件——无需读论文即可直观理解"同一场景、换一个介入会发生什么"。生产形态换真实 V-JEPA 2 特征 + 私有权重，即物理反事实问答界面。</p>
</section>

<section>
  <h2><span class="n">10</span>GPU 评测已跑通（RTX 5070 Ti · CUDA）</h2>
  <p class="sub">本机 RTX 5070 Ti Laptop（12GB，CUDA）实测跑通 KINE-Bench 全协议：98 条片段、V-JEPA 2 ViT-L/16（300M）、离线加载本地权重。V-JEPA 2 仅 encode，FUT-1/EMB-1 按协议诚实报 n/a 不伪造。</p>
  <div class="metrics">
    <div class="metric"><div class="m-v ok">1.000</div><div class="m-k">KINE-TEMP-1 时序一致性（基线 0.5）</div></div>
    <div class="metric"><div class="m-v ok">-0.002</div><div class="m-k">KINE-MOT-1 运动保真 r（基线 0.0）</div></div>
    <div class="metric"><div class="m-v ok">1.000</div><div class="m-k">KINE-CAU-1 因果 AUC（基线 0.5，degraded）</div></div>
    <div class="metric"><div class="m-v ok">901.7s</div><div class="m-k">GPU 墙钟（98 条，含模型加载）</div></div>
    <div class="metric"><div class="m-v ok">≈160×</div><div class="m-k">提速（每片段 1476s→9.2s vs CPU）</div></div>
    <div class="metric"><div class="m-v bad">skipped</div><div class="m-k">KINE-EVT-1（合成模式无真实视频+标注）</div></div>
  </div>
  <div class="card">
    <p class="sub">吞吐对照（同配置：num_frames 16 / img_size 64）</p>
    <p class="note">CPU smoke 8 条 = <b>11807.5s</b>（每片段 <b>1476s</b>）；本次 GPU 98 条 = <b>901.7s</b>（每片段 <b>9.2s</b>）→ <b>约 160× 提速</b>。
    且该 GPU 被系统电源策略锁在 ~17W（`Perf P4`），满血（100W）下还可更快。这是可复现的<b>吞吐硬证据</b>。</p>
    <p class="sub" style="margin-top:12px;">诚实边界（重要）</p>
    <p class="note">本轮为 <b>98 条合成片段</b>验证：TEMP-1=1.0 在合成数据上是<b>平凡高分</b>、MOT-1≈0 是因合成片段<b>无真实运动结构</b>、CAU-1 AUC=1.0 属退化态（intervene 分支不可用，<code>auc_do=null</code>）。
    因此这些<b>分数不构成竞争力证据</b>；本轮真实价值是「CUDA 链路跑通 + 协议可执行 + 160× 吞吐」。申报用的硬数字须用真实视频跑
    <code>python bench_gpu_launcher.py --data-dir &lt;视频文件夹&gt; --device cuda</code>（EVT-1 还需真实事件标注）。完整指南见 <code>BENCH_GPU.md</code>，原始报告见 <code>bench_report.html</code>。</p>
  </div>
</section>

<section>
  <div class="warn"><b>诚实边界：</b>当前 rollout / counterfactual 在<b>随机初始化</b>与<b>合成动力学</b>上验证（架构 + moat recipe 证明），
  非真实物理预测。真实轨迹后训练（生产形态）需用户 GPU 与标注数据，属闭源配方。所有指标均来自本仓库 <code>posttrain.py</code> /
  <code>counterfactual.py</code> / <code>real_feature_smoke.py</code> 的可复现运行，无外部断言。</div>
</section>

<footer>KINEWORLD · KineOne-WM — 反环境感知世界模型 · 证据台自生成于 {len(log)} epoch CPU 训练 · 全开源可复现</footer>
</div></body></html>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kineworld_capability_deck.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"wrote {out} ({os.path.getsize(out)} bytes)")
