# KINE-JEPA（kine-exp001）

勘境（KINEWORLD）世界模型技术底座：对 V-JEPA（arXiv:2404.08471）的 **clean-room 复现与改进**。

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
  jepa.py      KINE-JEPA（编码器 + EMA 目标编码器 + 预测器）
  dataset.py   kine-datapipe 视频片段数据集 + 合成冒烟数据
  train.py     单卡训练循环与实验日志
```

## 路线图

复现（0-8 周）→ 改进（物理先验损失 / 动作条件化 / 长时程记忆压缩，均做消融并公开）→ 机器人"想象引擎"产品化。

## 许可证

MIT（代码全为自有原创实现）。
