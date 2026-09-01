# Board 2026-09-01

## Shipped (code, CPU-verified)
- InterventionHead + frozen-encoder trainer on paired clips
- Grant tickets: missing / wrong action / high risk DENY
- Actuator sim only moves on ALLOW
- CAU encoder branch + CAU do-branch (auc_do)
- Bench loads EXP-001 and EXP-002 ckpts, keeps the head
- compare_arms + fill_exp002_html
- datapipe pairs + event windows
- 申报包发布闸门 `validate_package.py`（产物 / 引用 / 自包含 / 可编译，全绿）
- 证据台 10 支柱（含内嵌可交互反事实 demo + GPU 实测第 10 支柱）
- 申报入口 `kineworld_index.html`（8 张材料卡，双通道就绪度）

## GPU：已解阻塞（2026-09-01）
本机 **RTX 5070 Ti Laptop 12GB**，CUDA 链路打通，不再是阻塞项。
- 托管 venv 装了 **torch 2.11.0+cu128 + torchvision 0.26.0+cu128**（必须版本配对）。
- 负载实测 **98.8W / util 99%**（空闲读数 17W 是正常降频，**不是功耗墙**）。
- 98 条合成片段跑完 **901.7s**（≈9.2s 每条）vs CPU 11807.5s/8 条 → **≈160×**。
- 256px 原生分辨率可跑（2 条 144.8s）。三档分辨率吞吐扫描进行中。
- 已知限制：合成分数不可用作竞争力证据（TEMP/MOT 随样本量剧烈跳变，
  CAU 属 degraded `auc_do=null`）。**真实数字须 `--data-dir` 喂真实 98 条视频重跑。**

## Blocked on real data（不是 GPU）
- 真实 98 条视频 → KINE-Bench TEMP/MOT/**EVT** 硬数字（EVT-1 另需真实 events.json）
- 真实轨迹动作标注 → 后训练（EXP-002 A/B/C 2000 steps 从 ckpt-step10000 续跑）
- 真实轨迹后训练 → 物理可信 KineOne-WM（闭源配方落地）

## Blocked on external
- kineworld.com DNS 上线（apex → CNAME `kineworld-web.oss-cn-hongkong.aliyuncs.com`），
  由另一进程处理，本仓库不动。
- ⚠️ 安全待办：阿里云 AccessKey `LTAI5t9…` 曾以明文出现，建议 RAM 轮换 / 删除。

## Do not do
- More EXP-001 loss-only steps as the main line
- Hand-typed scores（所有数字必须来自脚本实测输出；本次 GPU 数字全部来自
  `bench_report.json` / `gpu_sweep.json`，无手打）
