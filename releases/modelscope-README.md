# KineOne-WM-Latent 0.1 · KINE-EXP-001

勘境（Kineworld）KineOne-WM 系列的潜在表征模型，采用 clean-room JEPA 实现并在单张 RTX 5070 Ti（12GB）上训练。

> `KINE-JEPA` 是历史架构代号。本模型与第三方榜单中既有的 `KineWorld` 条目无关，未取得 WorldArena 官方成绩。

- 架构：ViT-S/16 编码器 + 3D tubelet（2×16×16）+ 时空多块掩码 + EMA 目标编码器 + Transformer 预测器
- 数据：98 条运动过滤机器人视频片段（kine-datapipe 端到端产出，下载→切分→过滤全日志可查）
- 训练：25,000 步，seed 42，bf16，峰值显存 2.1 GB
- 权重格式：{"model": fp16 state_dict（在线编码器）, "config", "step", "license": "MIT"}
- 复评：`pip install torch` 后克隆 [kine-bench](https://github.com/zoahdev/kine-bench)，`python -m kinebench run --ckpt 本文件`（五项任务 + 随机基线）

## 已公开分数（KINE-Bench v0.3，同一 98 条评测集）

| 检查点 | FUT-1 | MOT-1 | EVT-1 | EMB-1 | TEMP-1 |
|---|---|---|---|---|---|
| step5000 | 0.823（基线0.127） | 0.660（0.0） | 0.539（0.5） | 0.681（0.683） | 0.500（0.5） |
| step10000 | 0.842（0.074） | 0.554（0.0） | 0.515（0.5） | 0.501（0.569） | 0.431（0.5） |

低于基线的数字原样发布，不修饰。完整训练日志与曲线：https://kineworld.com/report.html
评测结果原始文件：https://github.com/zoahdev/kine-bench/tree/main/results
训练代码与实验档案：https://github.com/zoahdev/kine-jepa

许可证：MIT
