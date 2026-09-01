#!/usr/bin/env bash
# 一键 GPU 评测（KINE-Bench · V-JEPA 2 on CUDA）
# 前置：托管 venv 已装 CUDA torch；本地 V-JEPA 2 权重在 KINE_VJEPA2_LOCAL。
#
# 用法（Git Bash）：
#   ./run_gpu_bench.sh            # 98 条合成片段，RTX 5070 CUDA 验证（无需视频文件）
# 真实片段评测（替换 <视频文件夹>，需真实 .mp4/.mkv/.webm）：
#   python bench_gpu_launcher.py --data-dir <视频文件夹> --device cuda
set -e

export KINE_VJEPA2_LOCAL="C:/Users/zoah/AppData/Local/Temp/vjepa2"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

PY="C:/Users/zoah/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "$(dirname "$0")"

echo "[run_gpu_bench] device=cuda model=vjepa2-vitl-256 mode=synthetic-98"
"$PY" bench_gpu_launcher.py --synthetic --device cuda --max-clips 98 --num-frames 16 --batch-size 4 --out bench_report.json
echo "[run_gpu_bench] done -> bench_report.json + bench_report.html"
