# KineOne-WM-Latent 0.1（历史仓库：kine-jepa）

勘境（Kineworld）的 KineOne-WM 潜在表征底座：对 V-JEPA（arXiv:2404.08471）的 **clean-room 复现与改进**。

> 正式模型名为 **KineOne-WM**。`KINE-JEPA` 仅保留为 KINE-EXP-001 的历史架构代号；本项目与第三方榜单中既有的 `KineWorld` 条目无关。

> Implementation inspired by the V-JEPA paper (arXiv:2404.08471); all code original.
> 未复制任何 V-JEPA 官方代码与权重（其为 CC-BY-NC-4.0，禁止商用）。

## 这是什么

JEPA（联合嵌入预测架构）视频自监督模型：

- **编码器**：ViT-S/patch16，3D tubelet（2×16×16）把视频切成 token，只编码未被掩码的可见 token
- **掩码**：时空多块随机掩码（比例按余弦从 0.9 → 0.75 退火）
- **预测器**：小型 Transformer，在目标编码器的表示空间中预测被掩码区域的特征
- **目标编码器**：编码器的 EMA 副本，无梯度
- **损失**：预测特征与归一化目标特征之间的 L1

## 硬件约束

单张 RTX 5070 Ti（12GB VRAM）。bf16 混合精度，batch 8，16 帧 224×224。

## 用法

```bash
git clone https://github.com/zoahdev/kine-jepa.git
cd kine-jepa
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements.txt

# 冒烟测试（合成数据，无需视频）
.venv/Scripts/python -m kineworld_jepa.train --smoke --steps 30

# 正式训练（数据由 ../kine-datapipe 产出）
.venv/Scripts/python -m kineworld_jepa.train --data-dir ../kine-datapipe/data/clips --steps 20000
```

实验日志（jsonl 指标 + 配置 + 摘要 + 检查点）写入 `experiments/KINE-EXP-001/run-*/`。

## 公开检查点

- [Release `exp001-step5000-weights`](https://github.com/zoahdev/kine-jepa/releases/tag/exp001-step5000-weights)：KINE-EXP-001 第 5000 步中间检查点（fp16 在线编码器，112.6 MB，MIT），可被 kine-bench 的 `load_model` 直接加载。

## 测试与评测

```bash
# 核心模块单元测试（CPU 即可，无需 GPU）
.venv/Scripts/python tests/test_core.py
```

训练产出的检查点用 **KINE-Bench** 评测（时序理解 / 运动幅度 / 未来预测保真度，单卡可复核）：
[github.com/zoahdev/kine-bench](https://github.com/zoahdev/kine-bench)

## 结构

```
kineworld_jepa/
  vit.py       3D tubelet ViT 编码器
  masking.py   时空多块掩码
  jepa.py      KineOne-WM-Latent（编码器 + EMA 目标编码器 + 预测器）
  causal.py    因果干预头（do(x) token）+ CausalKineJEPA（接口公开，训练产物闭源）
  rollout.py   动作条件化世界模型（ActionEmbedder / ActionRollout / LatentPlanner，CEM）
  dataset.py   kine-datapipe 视频片段数据集 + 合成冒烟数据
  train.py     单卡训练循环与实验日志
```

> `causal.py` 与 `rollout.py` 公开的是**接口与算法结构**；其依赖的具体训练权重、轨迹动作标注与后训练配方属于技术壁垒，闭源（见 kine-bench 的 `OPEN_SOURCE_BOUNDARY.md`）。

## 路线图

复现（0-8 周）→ 改进（物理先验损失 / 动作条件化 / 长时程记忆压缩，均做消融并公开）→ 机器人"想象引擎"产品化。

## 许可证

MIT（代码全为自有原创实现）。
