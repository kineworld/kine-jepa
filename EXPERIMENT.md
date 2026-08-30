# KINE-EXP-001 · 单卡世界模型预训练（实验档案）

> 勘境 KINEWORLD 的第一个公开实验。所有配置、数据配方与指标随仓库公开，可被任何人复核或复现。

## 一句话

在单张 RTX 5070 Ti（12GB）上，用 98 条从公开视频切出的片段，预训练一个 clean-room 实现的 JEPA 世界模型（ViT-S/16），验证"消费级算力可复现"这条路线走得通。

## 为什么做

竞品普遍把"进展"停留在口径上。勘境选择把第一步做成可查证的东西：有代码、有数据配方、有损失曲线、有硬件与随机种子。哪怕模型很小，流程是完整的、可复现的。

## 数据配方（用 kine-datapipe v0.2 产出）

| 项 | 值 |
|---|---|
| 来源 | 公开视频站点关键词检索（机械臂抓取 / 人形机器人行走） |
| 下载 | 12 条，成功 10 条（≤720p，mp4） |
| 切片 | 场景切分（min 2s / max 10s）→ 98 条 clip |
| 过滤 | 运动强度 ≥ 0.35 → 保留 98/98 |

## 模型与训练配置

| 项 | 值 |
|---|---|
| 架构 | KINE-JEPA：ViT-S/16 tubelet(2×16×16) 编码器 + EMA 目标编码器 + 预测器 |
| 编码器 | depth 12 / dim 384 / heads 6（约 22.5M 参数） |
| 输入 | 16 帧 · 224×224 · ImageNet 归一化 |
| 掩码 | 时空多块掩码，比例 0.90 → 0.75 余弦退火 |
| 损失 | 预测特征与归一化目标特征的 L1 |
| 优化 | AdamW，lr 3e-4 → 3e-6 余弦，500 步预热，weight-decay 0.05，grad-clip 1.0 |
| batch | 8 |
| 精度 | bf16 混合精度（Blackwell 原生） |
| 种子 | 42 |
| 步数 | 25000（冒烟测试已先验证 15–40 步） |

## 硬件

- 单张 NVIDIA GeForce RTX 5070 Ti Laptop GPU（12GB）
- 峰值显存 < 2.4GB（batch 8，bf16）——远低于 12GB 上限，留足放大空间

## 结果（持续更新）

- 冒烟测试（合成数据，40 步）：loss 0.81 → 0.25，无 NaN，显存稳定。
- 冒烟测试（真实 98 clip，15 步）：loss 0.81 → 0.39，解码与训练管线打通。
- 正式训练：进行中（2026-08-30 03:56 启动），逐 100 步记录于 `experiments/KINE-EXP-001/run-*/metrics.jsonl`。步 10900 时损失约 0.010（自 0.799 起），全程无 NaN，峰值显存 2.1GB。
- 检查点每 5000 步保存；**step5000 权重已发 GitHub Release**（`exp001-step5000-weights`，fp16 约 112.6MB，MIT），任何人可下载并用 kine-bench 一条命令复评。
- KINE-Bench v0.3 纵向评测（同一运行、同一 98 条集）已公开 5k / 10k 两点：FUT-1 0.823→0.842（随机基线反降，边际扩大），线性探针任务（TEMP/MOT）与事件敏感度回落——表征向预测目标特化的已知权衡，原样记录于 kine-bench 仓库，训练完成后补齐 15k/20k/25k。

## 复现方式

```bash
cd kine-exp001
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m kineworld_jepa.train --data-dir ../kine-datapipe/data/clips \
    --steps 25000 --batch-size 8 --seed 42
```

## 已记录的问题与修复（失败也是公开的一部分）

1. 数据批次维度丢失（取数时误取 `[0]`）→ 已修复。
2. 掩码块超调导致批内掩码数不一致 → 增加超额裁剪，保证可堆叠。

## 许可证红线

实现为完全自研（clean-room），仅借鉴 JEPA 论文思想（arXiv:2404.08471）；未复制、未使用任何 V-JEPA 官方代码或权重（其为 CC-BY-NC，禁止商用）。
