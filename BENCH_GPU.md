# KINE-Bench GPU 评测 · 一键指南

把 KineOne-WM（或对齐基线 V-JEPA 2）在**真实 98 条片段**上跑出可对比基准数字，
是和白泽/白皮书同台、把"原型"变成"可投"的关键一步。

## 前置（仅首次）
```bash
# 在 kine-jepa 的 venv 里装 kine-bench 依赖
.venv/Scripts/pip install torch torchvision transformers safetensors numpy opencv-python
# 确保 kine-bench 在路径上（与 kine-jepa 同级）
#   kine/
#     kine-jepa/   <- 本文件所在
#     kine-bench/  <- python -m kinebench 可用
```

## 数据管线（把你的视频整理成评测布局）
```bash
python prep_bench_data.py --src /path/to/your/videos --out bench_data
```
- 复制 `.mp4/.mkv/.webm` 到 `bench_data/raw/`
- 生成 `bench_data/events.json`（**启发式事件帧占位**；要可信的 KINE-EVT-1 请换成真实标注）

> 若你已有 `kine-datapipe/clips`（kineworld 的正式数据），直接指过去即可，跳过上面一步：
> `--data-dir ../kine-datapipe/clips`

## 一键评测（GPU）
```bash
python bench_gpu_launcher.py --data-dir bench_data --device cuda
# 或正式数据：
python bench_gpu_launcher.py --data-dir ../kine-datapipe/clips --device cuda
```
- 默认 `--max-clips 98`、`--model vjepa2-vitl-256`、`--num-frames 16`
- 产出 `bench_report.json` + `bench_report.html`（自包含报告，浏览器直接看）

## CPU 验证（无 GPU 时先确认链路）
```bash
python bench_gpu_launcher.py --smoke
```

## 你会拿到什么
| 任务 | 含义 | V-JEPA 2（编码器）状态 |
|---|---|---|
| KINE-TEMP-1 | 时序理解 | ✅ 可跑 |
| KINE-MOT-1 | 运动幅度 | ✅ 可跑 |
| KINE-EVT-1 | 物理事件偏移 | ✅ 可跑（需 events.json） |
| KINE-FUT-1 / EMB-1 / CAU-1 | 未来预测/具身想象/因果 | n/a（编码器无 predictor/intervention，协议诚实报 n/a） |

> 白泽类纯编码器同样只有 encode，所以 TEMP/MOT/EVT 是同台可比的硬数字；
> FUT/EMB/CAU 的差距正是 KineOne-WM「可规划 + 可反事实」的护城河所在——待真实轨迹后训练（闭源配方）补上。

## 拿到数字后
把 `bench_report.html` 并入 `kineworld_capability_deck.html` 作为「真实基准」一节，
直接服务 9/20 引航陪跑 + 10/1 合肥国资申报。
