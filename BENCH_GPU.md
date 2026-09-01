# KINE-Bench GPU 评测 · 一键指南（本机实测可用）

把 KineOne-WM（或对齐基线 V-JEPA 2）在 GPU 上跑出可对比基准数字，是和白泽同台、
把"原型"变成"可投"的关键一步。本指南基于 **本机 RTX 5070 Ti Laptop（12GB）** 实测跑通。

## 环境（本机）
- Python venv：`C:/Users/zoah/.workbuddy/binaries/python/envs/default/Scripts/python.exe`
- kine-bench：与 kine-jepa 同级 `kine/kine-bench`（`python -m kinebench` 可用；launcher 已自动加该路径）
- 本地 V-JEPA 2 权重：`C:/Users/zoah/AppData/Local/Temp/vjepa2`（config.json + model.safetensors 1.3GB + video_preprocessor_config.json）

## 首次：装 CUDA torch（关键坑）
托管 venv 默认是 **CPU 版 torch**，必须换 CUDA 构建，且 **torch 与 torchvision 版本必须匹配**，否则 `torchvision::nms` 注册失败、VJEPA2Model 导入崩。

```bash
# 1) 强制重装 CUDA torch（注意：普通 pip install 会因"已装 CPU 版"跳过，必须 --force-reinstall --no-deps）
cd C:/Users/zoah/.workbuddy/binaries/python/envs/default
Scripts/pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu128
#    -> 实测得到 torch 2.11.0+cu128 + torchvision 0.26.0+cu128（匹配，nms 正常）
# 2) 拉最新 kine-bench（含 transformers 5.x 兼容性修复）
cd C:/Users/zoah/WorkBuddy/2026-09-01-10-59-54/kine/kine-bench
git -c http.proxy=http://127.0.0.1:7897/ pull
```

> 踩坑记录（已修复，pull 即含）：
> - transformers 5.16.1 的 `AutoVideoProcessor` 是**惰性桩**，访问即抛 `ModuleNotFoundError`；原适配器 `try/except ImportError` 没兜住，连带把 `AutoModel` 置 None。已改为处理器可选 + 守卫只要求 `AutoModel`（提交 `ac448ef`，已推 `zoahdev/kine-bench`）。
> - 笔记本 GPU 被系统电源策略锁在 ~17W / `Perf P4`，`nvidia-smi -pl` 在沙箱内无权限改。评测照常跑，只是比满血慢；属正常现象。

## 一键评测（GPU）
```bash
cd C:/Users/zoah/WorkBuddy/2026-09-01-10-59-54/kine/kine-jepa

# A) 合成验证（无需视频文件，离线权重）：98 条合成片段走 CUDA
set KINE_VJEPA2_LOCAL=C:/Users/zoah/AppData/Local/Temp/vjepa2
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
python bench_gpu_launcher.py --synthetic --device cuda --max-clips 98

# B) 真实 98 条片段（把你的视频文件夹指过去；需真实标注才有可信 EVT-1）
python bench_gpu_launcher.py --data-dir <视频文件夹> --device cuda
#    或正式数据：python bench_gpu_launcher.py --data-dir ../kine-datapipe/clips --device cuda
```
- 默认 `--max-clips 98`、`--model vjepa2-vitl-256`、`--num-frames 16`、`--batch-size 2`（合成验证可用 4）
- 产出 `bench_report.json` + `bench_report.html`（自包含报告，浏览器直接看）
- `run_gpu_bench.sh` 已封装 A 的完整命令，双击/一键可复现

## 数据管线（真实片段时用）
```bash
python prep_bench_data.py --src /path/to/your/videos --out bench_data
```
- 复制 `.mp4/.mkv/.webm` 到 `bench_data/raw/`，生成 `bench_data/events.json`（事件帧为**启发式占位**，可信 EVT-1 需真实标注）

## CPU 验证（无 GPU 时确认链路）
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

> 白泽类纯编码器同样只有 encode，TEMP/MOT/EVT 是同台可比硬数字；
> FUT/EMB/CAU 差距正是 KineOne-WM「可规划 + 可反事实」的护城河——待真实轨迹后训练（闭源配方）补上。

## 拿到数字后
把 `bench_report.html` 并入 `kineworld_capability_deck.html` 第 10 支柱作为「真实基准」，
直接服务 9/20 引航陪跑 + 10/1 合肥国资申报。诚实边界：合成验证的数字仅证明 CUDA 路径
可行 + 给出 GPU 耗时对比（CPU smoke 8 条曾耗 11807s）；**真实申报数字须用真实 98 条片段跑 B**。
