#!/usr/bin/env python
# Data pipeline for KINE-Bench: turn a folder of raw videos into the layout the
# harness expects, plus a heuristic events.json so KINE-EVT-1 can run.
#
# KINE-Bench (VideoClipDataset) reads a directory of .mp4/.mkv/.webm clips.
# KINE-EVT-1 additionally needs `<data-dir>/raw/` + `<data-dir>/events.json`.
#
# This script:
#   1. copies accepted video files into <out>/raw/
#   2. writes <out>/events.json with EVENLY-SPACED candidate event frames
#
# HONESTY NOTE: the event frames are heuristic placeholders (not mined from real
# physical events). For a credible KINE-EVT-1 number, replace events.json with
# annotations from kine-datapipe's `events` miner or your own labels.
#
# Usage:
#   python prep_bench_data.py --src /path/to/videos --out bench_data
import argparse
import json
import shutil
import sys
from pathlib import Path

ACCEPT = (".mp4", ".mkv", ".webm")


def frame_count(path):
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n or None


def main():
    ap = argparse.ArgumentParser(prog="prep_bench_data",
        description="把视频文件夹整理成 KINE-Bench 评测布局 + 启发式 events.json")
    ap.add_argument("--src", required=True, help="原始视频文件夹")
    ap.add_argument("--out", required=True, help="输出目录（生成 raw/ + events.json）")
    ap.add_argument("--events-per-clip", type=int, default=2,
                    help="每个片段放置的启发式候选事件帧数")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in src.iterdir()
                    if p.is_file() and p.suffix.lower() in ACCEPT)
    if not videos:
        print(f"[prep] 在 {src} 未找到 {ACCEPT} 视频；请确认格式")
        return 1

    events = {}
    skipped = []
    for v in videos:
        dst = raw / v.name
        if not dst.exists():
            shutil.copy2(v, dst)
        n = frame_count(dst)
        if n is None:
            # 无 cv2：用帧号占位，EVT-1 运行时再校验
            cand = [60, 120]
        else:
            if n < 300:
                skipped.append((v.name, n))
            cand = [max(30, int(n * (i + 1) / (args.events_per_clip + 1)))
                    for i in range(args.events_per_clip)]
        events[v.name] = {"events": [{"frame": f} for f in cand]}

    (out / "events.json").write_text(
        json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[prep] 复制 {len(videos)} 个片段 -> {raw}")
    print(f"[prep] 写 events.json（启发式，{args.events_per_clip} 帧/片段）-> {out/'events.json'}")
    if skipped:
        print(f"[prep] 警告：{len(skipped)} 个片段总帧<300，KINE-EVT-1 将跳过（需 >=300 帧）：")
        for name, n in skipped[:10]:
            print(f"        {name} ({n} 帧)")
    print(f"[prep] 下一步：python bench_gpu_launcher.py --data-dir {out} --device cuda")
    return 0


if __name__ == "__main__":
    sys.exit(main())
