# KineOne-WM · 反环境感知世界模型（底座仓库 kine-jepa）

勘境（Kineworld）的 **KineOne-WM** —— 基于 Meta **V-JEPA 2**（MIT / Apache-2.0，可商用）编码器的
**动作条件化、可规划、可反事实推演**的 latent 世界模型。本仓库是其开源底座：架构、接口、基准与
证明脚本全公开、可复现、无需 GPU；训练权重、轨迹动作标注与后训练配方闭源（见下方边界）。

> 正式模型名为 **KineOne-WM**。`KINE-JEPA` 仅保留为历史架构代号。本项目与第三方榜单中既有的 `KineWorld` 条目无关。
> Implementation inspired by V-JEPA 2-AC (arXiv:2506.09985, MIT) and KINE-EXP-002 causal.py; all code original.

## 它解决什么

纯编码器/基准（如白泽类）只能"看懂"当前画面。KineOne-WM 额外能：

- **预测未来**：给定场景与连续动作指令，rollout 未来 latent（动作条件化世界模型）；
- **规划**：LatentPlanner（CEM 无梯度）在 latent 空间搜动作序列抵达目标状态；
- **反事实推演**：离散 `do(x)`（撤支撑 / 断接触 / 随机）经因果干预头每步重条件化 latent，
  回答"如果当时撤了支撑会怎样"——这是白泽类纯编码器不具备的能力。

## 架构（开源）

```
kineworld_jepa/
  vit.py         3D tubelet ViT 编码器（clean-room）
  causal.py      因果干预头 InterventionHead(do(x)) + CausalKineJEPA（接口公开）
  rollout.py     ActionRollout（残差更新 + latent_clip 长程稳定）
                 MultiActionEmbedder（连续+离散混合多动作空间）
                 LatentPlanner（CEM 无梯度规划）
                 VJEPA2Projector（8192→256 token 对齐 V-JEPA 2 的 1024-d）
  counterfactual.py  CounterfactualRollout(ActionRollout 子类)
                     do(x) 重条件化 latent + arm 动作 token，因子分解无重复条件
tests/           test_core / test_rollout / test_counterfactual / test_posttrain（CPU 全绿）
```

编码器对齐基线 = **Meta V-JEPA 2**（ViT-L，1.3GB 权重，MIT/Apache-2.0 可商用）；
本地权重离线加载，输出真实 `(B, 1024, 1024)` 特征，直接进 rollout 空间。

## 证据（全开源、可复现、CPU）

| 证据 | 脚本 / 产物 | 关键数字 |
|---|---|---|
| 真实特征端到端 | `real_feature_smoke.py` | 真实 V-JEPA 2 编码 → 反事实分歧 0.0623 |
| 反事实推演 | `counterfactual.py` + `counterfactual_demo.html` | do(x) 分歧可测、确定性可复现 |
| 后训练配方（合成） | `posttrain.py` + `posttrain_demo.html` | rollout MSE ↓63.7%（6.41→2.33） |
| 后训练配方（真实特征） | `real_feature_posttrain.py` + `real_feature_posttrain.html` | 留一 rollout ↓30%（0.0815→0.0573） |
| **统一能力证据台** | `build_deck.py` → `kineworld_capability_deck.html` | 10 支柱聚合（内嵌可交互反事实 demo），服务申报 |
| **GPU 评测（CUDA 实测）** | `bench_gpu_launcher.py` / `gpu_resolution_sweep.py` → `BENCH_GPU.md` | 98 条 901.7s（≈160× vs CPU）；原生 256px 全量 926.5s（仅慢 2.7%）；64 帧满配零代价 |

复现（CPU）：

```bash
git clone https://github.com/zoahdev/kine-jepa.git
cd kine-jepa
python -m venv .venv && .venv/Scripts/pip install torch
python build_deck.py            # 重算后训练+反事实，生成 kineworld_capability_deck.html
python posttrain.py             # 合成动力学后训练证明
python real_feature_posttrain.py # 真实特征后训练证明（需本地 V-JEPA 2 权重）
python -m pytest tests/         # 回归测试
```

GPU 评测（可选，笔记本级即可）见 `BENCH_GPU.md`：一行命令跑 KINE-Bench 全协议，
实测 RTX 5070 Ti Laptop 上 64px ≈160× 提速、原生 256px 全量可行、fpc64 满配零吞吐代价。

## 开源 / 闭源边界（护城河）

| 类别 | 内容 | 边界 |
|---|---|---|
| 架构 / 接口 | CounterfactualRollout、ActionRollout、MultiActionEmbedder、InterventionHead、KINE-Bench 接口 | **开源** |
| 基准 | 评测协议；缺能力报 n/a 不伪造的诚实约定 | **开源** |
| 权重 | 特定本体后训练权重、动作标注 | **闭源** |
| 后训练配方 | 课程 / 数据配比 / ViT-g teacher 蒸馏 | **闭源** |

> 公开的是"能想象替代未来"的方法与可复核证据；壁垒在私有的、针对具体本体的训练产物与配方。

## 申报节点

- **9/20 引航陪跑创业营**：提交申请，本证据台作技术可行性佐证。
- **10/1 合肥国资 + 公司注册**：差异化定位（可规划/反事实 + 商用合规 + 单设备 ~12GB 部署）。

## 诚实边界

当前 rollout / counterfactual 在**随机初始化**与**合成动力学**上验证（架构 + moat recipe 证明），
真实特征后训练为**概念验证（21 对）**，非大规模物理预测。生产形态：真实轨迹 + 动作标注 + 私有权重/配方。
所有指标均来自本仓库可复现脚本，无外部断言。

## 许可证

MIT（代码全为自有原创实现）。
